#include "water_engine.h"

#define VMA_IMPLEMENTATION

#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
    do {                                                                                                               \
        printf((format), __VA_ARGS__);                                                                                 \
        printf("\n");                                                                                                  \
    } while (false)

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#include <vma/vk_mem_alloc.h>
#include <vma/vk_mem_alloc.hpp>

#include "vk_images.h"
#include "vk_initializers.h"
#include "vk_types.h"

#include <VkBootstrap.h>

#include <iostream>

#include "vk_pipelines.h"

constexpr bool bUseValidationLayers = true;

#define VK_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        VkResult err = x;                                                                                              \
        if (err) {                                                                                                     \
            std::cout << "Detected Vulkan error: " << err << std::endl;                                                \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

void WaterEngine::init() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow("Water Engine", _windowExtent.width, _windowExtent.height, window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_pipelines();

    init_imgui();

    _isInitialized = true;
}

void WaterEngine::cleanup() {
    if (_isInitialized) {
        _device.waitIdle();

        _device.destroyCommandPool(_commandPool);

        _device.destroyFence(_renderFence);
        _device.destroySemaphore(_renderSemaphore);
        _device.destroySemaphore(_presentSemaphore);

        _device.destroyRenderPass(_renderPass);

        for (int i = 0; i < _framebuffers.size(); i++) {
            _device.destroyFramebuffer(_framebuffers[i]);
        }

        for (int i = 0; i < FRAME_OVERLAP; i++) {
            _device.destroyCommandPool(_frames[i]._commandPool);
            _device.destroyFence(_frames[i]._renderFence);
            _device.destroySemaphore(_frames[i]._renderSemaphore);
            _device.destroySemaphore(_frames[i]._swapchainSemaphore);

            _frames[i]._deletionQueue.flush();
        }

        _mainDeletionQueue.flush();

        destroy_swapchain();

        _instance.destroySurfaceKHR(_surface);
        _device.destroy();

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        _instance.destroy();
        SDL_DestroyWindow(_window);
    }
}

void WaterEngine::draw() {
    try {
        vk::Result waitResult = _device.waitForFences(1, &get_current_frame()._renderFence, true, 1000000000);

        get_current_frame()._deletionQueue.flush();

        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for a frame");
        }

        vk::Result resetResult = _device.resetFences(1, &get_current_frame()._renderFence);

        if (resetResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to reset fences");
        }

        auto acquireResult =
                _device.acquireNextImageKHR(_swapchain, 1000000000, get_current_frame()._swapchainSemaphore, nullptr);

        if (acquireResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        uint32_t swapchainImageIndex = acquireResult.value;

        vk::CommandBuffer cmd = get_current_frame()._mainCommandBuffer;

        cmd.reset(vk::CommandBufferResetFlags());

        vk::CommandBufferBeginInfo cmdBeginInfo =
                vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        _drawExtent.width = _drawImage.imageExtent.width;
        _drawExtent.height = _drawImage.imageExtent.height;

        cmd.begin(cmdBeginInfo);

        // Transition our main draw image to general layout for writing
        // Old layout does not matter due to overwrite
        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        draw_background(cmd);

        // Transition the draw image and the swapchain image to their correct transfer layouts
        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eGeneral,
                                 vk::ImageLayout::eTransferSrcOptimal);
        vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferDstOptimal);

        // Execute a copy from the draw image into the swapchain
        vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent,
                                    _swapchainExtent);

        // Set the swapchain image layout to Attachment Optimal so we can draw it
        vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eTransferDstOptimal,
                                 vk::ImageLayout::eColorAttachmentOptimal);

        draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

        // Set the swapchain image layout to Present so we can draw it
        vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                                 vk::ImageLayout::ePresentSrcKHR);

        // Finalize the command buffer (can no longer commands, but it can now be executed)
        cmd.end();

        // Prepare submission to the queue
        // Wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
        // We will signal the _renderSemaphore, to signal that rendering was finished
        vk::CommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);

        vk::SemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
                vk::PipelineStageFlagBits2::eColorAttachmentOutput, get_current_frame()._swapchainSemaphore);
        vk::SemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(vk::PipelineStageFlagBits2::eAllGraphics,
                                                                           get_current_frame()._renderSemaphore);

        vk::SubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

        vk::Result submitResult = _graphicsQueue.submit2(1, &submit, get_current_frame()._renderFence);

        if (submitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to submit a frame: " +
                                     std::string(make_error_code(submitResult).message()));
        }

        // Prepare present
        // This will put the image we just rendered into the window
        // We must wait for the _renderSemaphore
        // Drawing commands must finish before the image is displayed
        vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
                                                 .setPNext(nullptr)
                                                 .setPSwapchains(&_swapchain)
                                                 .setSwapchainCount(1)
                                                 .setPWaitSemaphores(&get_current_frame()._renderSemaphore)
                                                 .setWaitSemaphoreCount(1)
                                                 .setPImageIndices(&swapchainImageIndex);

        vk::Result presentResult = _graphicsQueue.presentKHR(presentInfo);

        if (presentResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present a frame");
        }

        _frameNumber++;
    } catch (const vk::SystemError &e) {
        throw std::runtime_error("Error drawing frame: " + std::string(e.what()));
    }
}

void WaterEngine::run() {
    SDL_Event e;
    bool bQuit = false;

    while (!bQuit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_EVENT_QUIT)
                bQuit = true;

            if (e.window.type == SDL_WINDOW_MINIMIZED) {
                stop_rendering = true;
            }

            if (e.window.type == SDL_EVENT_WINDOW_RESTORED) {
                stop_rendering = false;
            }

            ImGui_ImplSDL3_ProcessEvent(&e);
        }

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("spectrum")) {
            ComputeOceanEffect &selected = oceanEffects[currentOceanEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::InputFloat2("Wind Direction", (float *) &selected.parameters.windDirection);
            ImGui::SliderFloat("Wind Speed", &selected.parameters.windSpeed, 0, 100);
            ImGui::SliderFloat("Gravity", &selected.parameters.g, 0, 100);
            ImGui::SliderFloat("Wind Amplitude", &selected.parameters.waveAmplitude, 0, 100);

            selected.parameters.t = static_cast<float>(SDL_GetTicks() / 1000.0f);
        }
        ImGui::End();

        ImGui::Render();

        draw();
    }
}

void WaterEngine::init_vulkan() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Water Engine")
                            .request_validation_layers(bUseValidationLayers)
                            .use_default_debug_messenger()
                            .require_api_version(1, 3, 0)
                            .build();

    if (!inst_ret) {
        throw std::runtime_error("Failed to create instance: " + inst_ret.error().message());
    }

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    VULKAN_HPP_DEFAULT_DISPATCHER.init(_instance);

    VkSurfaceKHR surface;
    SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &surface);
    _surface = surface, surface = nullptr;

    vk::PhysicalDeviceVulkan13Features features =
            vk::PhysicalDeviceVulkan13Features().setDynamicRendering(true).setSynchronization2(true);

    vk::PhysicalDeviceVulkan12Features features12 =
            vk::PhysicalDeviceVulkan12Features().setBufferDeviceAddress(true).setDescriptorIndexing(true);

    vkb::PhysicalDeviceSelector selector{vkb_inst};
    auto phys_ret = selector.set_minimum_version(1, 3)
                            .set_required_features_13(features)
                            .set_required_features_12(features12)
                            .set_surface(_surface)
                            .select();

    if (!phys_ret) {
        throw std::runtime_error("Failed to create physical device: " + std::string(phys_ret.error().message()));
    }

    vkb::PhysicalDevice physicalDevice = phys_ret.value();

    vkb::DeviceBuilder deviceBuilder{physicalDevice};

    vkb::Device vkbDevice = deviceBuilder.build().value();

    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    VULKAN_HPP_DEFAULT_DISPATCHER.init(_device);

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    vma::AllocatorCreateInfo allocatorInfo = vma::AllocatorCreateInfo()
                                                     .setPhysicalDevice(_chosenGPU)
                                                     .setDevice(_device)
                                                     .setInstance(_instance)
                                                     .setFlags(vma::AllocatorCreateFlagBits::eBufferDeviceAddress);

    _allocator = vma::createAllocator(allocatorInfo);

    _mainDeletionQueue.push_function([&]() { _allocator.destroy(); });
}

void WaterEngine::create_swapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};

    _swapchainImageFormat = vk::Format::eB8G8R8A8Unorm;

    auto swapchain_ret =
            swapchainBuilder
                    .set_desired_format(vk::SurfaceFormatKHR(_swapchainImageFormat, vk::ColorSpaceKHR::eSrgbNonlinear))
                    .set_desired_present_mode(static_cast<VkPresentModeKHR>(vk::PresentModeKHR::eMailbox))
                    .set_desired_extent(width, height)
                    .add_image_usage_flags(static_cast<VkImageUsageFlags>(vk::ImageUsageFlagBits::eTransferDst))
                    .build();

    if (!swapchain_ret) {
        std::cout << "Failed to create swapchain: " << swapchain_ret.error().message();
    }

    vkb::Swapchain vkbSwapchain = swapchain_ret.value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;

    auto vkb_images = vkbSwapchain.get_images().value();
    _swapchainImages = {vkb_images.begin(), vkb_images.end()};

    auto vkb_image_views = vkbSwapchain.get_image_views().value();
    _swapchainImageViews = {vkb_image_views.begin(), vkb_image_views.end()};
}


void WaterEngine::init_swapchain() {
    create_swapchain(_windowExtent.width, _windowExtent.height);

    vk::Extent3D drawImageExtent = {_windowExtent.width, _windowExtent.height, 1};

    // _drawImage.imageFormat = vk::Format::eR16G16B16A16Sfloat;
    _drawImage.imageFormat = vk::Format::eR16G16Sfloat;
    _drawImage.imageExtent = drawImageExtent;

    vk::ImageUsageFlags drawImageUsages = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                          vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment;

    vk::ImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    vma::AllocationCreateInfo rimg_allocinfo =
            vma::AllocationCreateInfo()
                    .setUsage(vma::MemoryUsage::eGpuOnly)
                    .setRequiredFlags(vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal));

    vk::Result allocResult =
            _allocator.createImage(&rimg_info, &rimg_allocinfo, &_drawImage.image, &_drawImage.allocation, nullptr);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create draw image");
    }

    vk::ImageViewCreateInfo rview_info =
            vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, vk::ImageAspectFlagBits::eColor);

    vk::Result createImgViewResult = _device.createImageView(&rview_info, nullptr, &_drawImage.imageView);

    if (createImgViewResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create draw image view");
    }

    _mainDeletionQueue.push_function([=, this]() {
        _device.destroyImageView(_drawImage.imageView);
        _allocator.destroyImage(_drawImage.image, _drawImage.allocation);
    });
}


void WaterEngine::destroy_swapchain() {
    _device.destroySwapchainKHR(_swapchain);

    for (auto &_swapchainImageView: _swapchainImageViews) {
        _device.destroyImageView(_swapchainImageView);
    }
}

void WaterEngine::init_commands() {
    vk::CommandPoolCreateInfo commandPoolInfo =
            vkinit::command_pool_create_info(_graphicsQueueFamily, vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        try {
            _frames[i]._commandPool = _device.createCommandPool(commandPoolInfo);

            vk::CommandBufferAllocateInfo cmdAllocInfo = vk::CommandBufferAllocateInfo()
                                                                 .setPNext(nullptr)
                                                                 .setCommandPool(_frames[i]._commandPool)
                                                                 .setCommandBufferCount(1)
                                                                 .setLevel(vk::CommandBufferLevel::ePrimary);

            _frames[i]._mainCommandBuffer = _device.allocateCommandBuffers(cmdAllocInfo).front();
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create command pool/buffer: " + std::string(e.what()));
        }
    }

    // Imm commands
    _immCommandPool = _device.createCommandPool(commandPoolInfo);

    vk::CommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);

    _immCommandBuffer = _device.allocateCommandBuffers(cmdAllocInfo).front();

    _mainDeletionQueue.push_function([=, this]() { _device.destroyCommandPool(_immCommandPool); });
}

void WaterEngine::init_sync_structures() {
    vk::FenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(vk::FenceCreateFlagBits::eSignaled);
    vk::SemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    for (auto &_frame: _frames) {
        try {
            _frame._renderFence = _device.createFence(fenceCreateInfo);

            _frame._swapchainSemaphore = _device.createSemaphore(semaphoreCreateInfo);
            _frame._renderSemaphore = _device.createSemaphore(semaphoreCreateInfo);
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create fence/semaphore: " + std::string(e.what()));
        }
    }

    _immFence = _device.createFence(fenceCreateInfo);
    _mainDeletionQueue.push_function([=, this]() { _device.destroyFence(_immFence); });
}

void WaterEngine::init_descriptors() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {{vk::DescriptorType::eStorageImage, 1}};

    globalDescriptorAllocator.init_pool(_device, 10, sizes);

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eStorageImage);
        _drawImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eCompute);
    }

    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    vk::DescriptorImageInfo imgInfo =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_drawImage.imageView);

    vk::WriteDescriptorSet drawImageWrite = vk::WriteDescriptorSet()
                                                    .setPNext(nullptr)
                                                    .setDstBinding(0)
                                                    .setDstSet(_drawImageDescriptors)
                                                    .setDescriptorCount(1)
                                                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                                                    .setPImageInfo(&imgInfo);

    _device.updateDescriptorSets(1, &drawImageWrite, 0, nullptr);

    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pool(_device);
        _device.destroyDescriptorSetLayout(_drawImageDescriptorLayout);
    });
}


void WaterEngine::draw_background(vk::CommandBuffer cmd) {
    ComputeOceanEffect &effect = oceanEffects[currentOceanEffect];

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, effect.pipeline);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, _spectrumPipelineLayout, 0, 1, &_drawImageDescriptors, 0,
                           nullptr);

    cmd.pushConstants(_spectrumPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(OceanParameters),
                      &effect.parameters);

    cmd.dispatch(std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);
}

void WaterEngine::init_pipelines() { init_background_pipelines(); }

void WaterEngine::init_background_pipelines() {
    vk::PipelineLayoutCreateInfo computeLayout = vk::PipelineLayoutCreateInfo()
                                                         .setPNext(nullptr)
                                                         .setPSetLayouts(&_drawImageDescriptorLayout)
                                                         .setSetLayoutCount(1);

    vk::PushConstantRange pushConstant = vk::PushConstantRange()
                                                 .setOffset(0)
                                                 .setSize(sizeof(OceanParameters))
                                                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    computeLayout.setPPushConstantRanges(&pushConstant);
    computeLayout.setPushConstantRangeCount(1);

    _spectrumPipelineLayout = _device.createPipelineLayout(computeLayout);

    vk::ShaderModule computeDrawShader;
    if (!vkutil::load_shader_module("../shaders/spectrum2.comp.spv", _device, &computeDrawShader)) {
        fmt::print("Error when building the compute shader \n");
    }

    vk::PipelineShaderStageCreateInfo stageInfo = vk::PipelineShaderStageCreateInfo()
                                                          .setPNext(nullptr)
                                                          .setStage(vk::ShaderStageFlagBits::eCompute)
                                                          .setModule(computeDrawShader)
                                                          .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_spectrumPipelineLayout).setStage(stageInfo);

    ComputeOceanEffect ocean;
    ocean.layout = _spectrumPipelineLayout;
    ocean.name = "ocean";

    vk::Result createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &ocean.pipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create compute pipeline");
    }

    oceanEffects.push_back(ocean);

    _device.destroyShaderModule(computeDrawShader);

    _mainDeletionQueue.push_function([=, this]() {
        _device.destroyPipelineLayout(_spectrumPipelineLayout);
        _device.destroyPipeline(ocean.pipeline);
    });
}

void WaterEngine::immediate_submit(std::function<void(vk::CommandBuffer)> &&function) {
    _device.resetFences(_immFence);
    _immCommandBuffer.reset();

    vk::CommandBuffer cmd = _immCommandBuffer;
    vk::CommandBufferBeginInfo cmdBeginInfo =
            vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    cmd.begin(cmdBeginInfo);

    function(cmd);

    cmd.end();

    vk::CommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    vk::SubmitInfo2 submit = vkinit::submit_info(&cmdInfo, nullptr, nullptr);

    vk::Result submitResult = _graphicsQueue.submit2(1, &submit, _immFence);
    if (submitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to submit a frame: " + std::string(make_error_code(submitResult).message()));
    }

    vk::Result waitResult = _device.waitForFences(_immFence, true, 9999999999);
    if (waitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence: " + std::string(make_error_code(waitResult).message()));
    }
}

void WaterEngine::init_imgui() {
    // Create descriptor pool for imgui
    vk::DescriptorPoolSize poolSizes[] = {
            {vk::DescriptorType::eSampler, 1000},
            {vk::DescriptorType::eCombinedImageSampler, 1000},
            {vk::DescriptorType::eSampledImage, 1000},
            {vk::DescriptorType::eStorageImage, 1000},
            {vk::DescriptorType::eUniformTexelBuffer, 1000},
            {vk::DescriptorType::eStorageTexelBuffer, 1000},
            {vk::DescriptorType::eUniformBuffer, 1000},
            {vk::DescriptorType::eStorageBuffer, 1000},
            {vk::DescriptorType::eUniformBufferDynamic, 1000},
            {vk::DescriptorType::eStorageBufferDynamic, 1000},
            {vk::DescriptorType::eInputAttachment, 1000},
    };

    vk::DescriptorPoolCreateInfo pool_info = vk::DescriptorPoolCreateInfo()
                                                     .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                                                     .setMaxSets(1000)
                                                     .setPoolSizeCount((uint32_t) std::size(poolSizes))
                                                     .setPPoolSizes(poolSizes);

    vk::DescriptorPool imguiPool = _device.createDescriptorPool(pool_info);

    // Init imgui library

    ImGui::CreateContext();

    ImGui_ImplSDL3_InitForVulkan(_window);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _instance;
    init_info.PhysicalDevice = _chosenGPU;
    init_info.Device = _device;
    init_info.Queue = _graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.UseDynamicRendering = true;

    // Dynamic rendering parameters for imgui
    init_info.PipelineRenderingCreateInfo =
            vk::PipelineRenderingCreateInfo().setColorAttachmentCount(1).setPColorAttachmentFormats(
                    &_swapchainImageFormat);

    init_info.MSAASamples = static_cast<VkSampleCountFlagBits>(vk::SampleCountFlagBits::e1);

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();

    _mainDeletionQueue.push_function([=, this]() {
        ImGui_ImplVulkan_Shutdown();
        _device.destroyDescriptorPool(imguiPool);
    });
}

void WaterEngine::draw_imgui(vk::CommandBuffer cmd, vk::ImageView targetImageView) {
    vk::RenderingAttachmentInfo colorAttachment =
            vkinit::attachment_info(targetImageView, nullptr, vk::ImageLayout::eColorAttachmentOptimal);
    vk::RenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    cmd.beginRendering(renderInfo);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    cmd.endRendering();
}
