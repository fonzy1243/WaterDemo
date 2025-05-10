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

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

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

    SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow("Water Engine", _windowExtent.width, _windowExtent.height, window_flags);

    init_vulkan();

    init_swapchain();

    init_commands();

    init_sync_structures();

    init_descriptors();

    init_compute_images();
    init_compute_descriptors();

    init_pipelines();

    init_imgui();

    init_default_data();

    mainCamera.velocity = glm::vec3(0.f);
    mainCamera.position = glm::vec3(30.f, -00.f, -085.f);

    mainCamera.pitch = 0;
    mainCamera.yaw = 0;

    // Delete when done
    std::string structurePath = {"..\\assets\\structure.glb"};
    auto structureFile = loadGltf(this, structurePath);

    assert(structureFile.has_value());

    loadScenes["structure"] = *structureFile;

    _isInitialized = true;
}

void WaterEngine::cleanup() {
    if (_isInitialized) {
        _device.waitIdle();

        loadScenes.clear();

        for (int i = 0; i < FRAME_OVERLAP; i++) {
            _device.destroyCommandPool(_frames[i]._commandPool);
            _device.destroyFence(_frames[i]._renderFence);
            _device.destroySemaphore(_frames[i]._renderSemaphore);
            _device.destroySemaphore(_frames[i]._swapchainSemaphore);

            _frames[i]._deletionQueue.flush();
        }

        for (auto &mesh: testMeshes) {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }

        metalRoughMaterial.clear_resources(_device);

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
        update_scene();

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

        if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
            resize_requested = true;
            return;
        }

        uint32_t swapchainImageIndex = acquireResult.value;

        vk::CommandBuffer cmd = get_current_frame()._mainCommandBuffer;

        cmd.reset(vk::CommandBufferResetFlags());

        vk::CommandBufferBeginInfo cmdBeginInfo =
                vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        _drawExtent.width = std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * renderScale;
        _drawExtent.height = std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * renderScale;

        _spectrumExtent.width = _spectrumImage.imageExtent.width;
        _spectrumExtent.height = _spectrumImage.imageExtent.height;

        _butterflyExtent.width = _butterflyImage.imageExtent.width;
        _butterflyExtent.height = _butterflyImage.imageExtent.height;

        cmd.begin(cmdBeginInfo);

        // Transition our FFT images to general layout for writing
        // Old layout does not matter
        vkutil::transition_image(cmd, _spectrumImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        vkutil::transition_image(cmd, _spectrumNormalImage.image, vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eGeneral);
        vkutil::transition_image(cmd, _normalMapImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        // Transition our main draw image to general layout for writing
        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

        draw_background(cmd);

        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eGeneral,
                                 vk::ImageLayout::eColorAttachmentOptimal);
        vkutil::transition_image(cmd, _depthImage.image, vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eDepthAttachmentOptimal);

        draw_geometry(cmd);

        // Transition the draw image and the swapchain image to their correct transfer layouts
        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eColorAttachmentOptimal,
                                 vk::ImageLayout::eTransferSrcOptimal);
        vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferDstOptimal);

        // Execute a copy from the draw image into the swapchain
        vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent,
                                    _swapchainExtent);

        vkutil::transition_image(cmd, _drawImage.image, vk::ImageLayout::eTransferSrcOptimal,
                                 vk::ImageLayout::eGeneral);

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

        if (presentResult == vk::Result::eErrorOutOfDateKHR) {
            resize_requested = true;
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

            if (e.window.type == SDL_EVENT_WINDOW_RESIZED) {
                resize_requested = true;
            }

            mainCamera.processSDLEvent(e);
            ImGui_ImplSDL3_ProcessEvent(&e);
        }

        if (stop_rendering) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (resize_requested) {
            resize_swapchain();
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        if (ImGui::Begin("spectrum")) {
            ComputeOceanEffect &selected = oceanEffects[currentOceanEffect];

            ImGui::Text("Selected effect: ", selected.name);

            ImGui::Checkbox("Show Heightmap", &showHeightmap);

            ImGui::InputFloat2("Wind Direction", reinterpret_cast<float *>(&selected.parameters.windDirection));
            ImGui::SliderFloat("Wind Speed", &selected.parameters.windSpeed, 0, 100);
            ImGui::SliderFloat("Gravity", &selected.parameters.g, 0, 100);
            ImGui::SliderFloat("Wave Amplitude", &selected.parameters.waveAmplitude, 0, 30);
        }
        ImGui::End();

        ImGui::Render();

        oceanEffects[currentOceanEffect].parameters.t = static_cast<double>(SDL_GetTicks() / 1000.0);

        draw();
    }
}

void WaterEngine::init_vulkan() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    vkb::InstanceBuilder builder;

    auto inst_ret = builder.set_app_name("Water Engine")
                            .request_validation_layers(bUseValidationLayers)
                            .use_default_debug_messenger()
                            .require_api_version(1, 4, 0)
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
    auto phys_ret = selector.set_minimum_version(1, 4)
                            .set_required_features_13(features)
                            .set_required_features_12(features12)
                            .set_surface(_surface)
                            .add_required_extension(vk::KHRPushDescriptorExtensionName)
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

    // Draw image

    _drawImage.imageFormat = vk::Format::eR32G32B32A32Sfloat;
    _drawImage.imageExtent = drawImageExtent;

    vk::ImageUsageFlags drawImageUsages = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                          vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment;

    vk::ImageCreateInfo draw_img_info =
            vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);

    vma::AllocationCreateInfo draw_img_allocinfo =
            vma::AllocationCreateInfo()
                    .setUsage(vma::MemoryUsage::eGpuOnly)
                    .setRequiredFlags(vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal));

    vk::Result allocResult = _allocator.createImage(&draw_img_info, &draw_img_allocinfo, &_drawImage.image,
                                                    &_drawImage.allocation, nullptr);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create draw image");
    }

    vk::ImageViewCreateInfo draw_view_info =
            vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, vk::ImageAspectFlagBits::eColor);

    vk::Result createImgViewResult = _device.createImageView(&draw_view_info, nullptr, &_drawImage.imageView);

    if (createImgViewResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create draw image view");
    }

    // Depth image

    _depthImage.imageFormat = vk::Format::eD32Sfloat;
    _depthImage.imageExtent = drawImageExtent;

    vk::ImageUsageFlags depthImageUsages = vk::ImageUsageFlagBits::eDepthStencilAttachment;

    vk::ImageCreateInfo dimg_info =
            vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    allocResult = _allocator.createImage(&dimg_info, &draw_img_allocinfo, &_depthImage.image, &_depthImage.allocation,
                                         nullptr);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create depth image");
    }

    vk::ImageViewCreateInfo dimg_view_info =
            vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, vk::ImageAspectFlagBits::eDepth);

    createImgViewResult = _device.createImageView(&dimg_view_info, nullptr, &_depthImage.imageView);

    if (createImgViewResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create depth image view");
    }

    _mainDeletionQueue.push_function([=, this]() {
        // Draw img
        _device.destroyImageView(_drawImage.imageView);
        _allocator.destroyImage(_drawImage.image, _drawImage.allocation);
        // Depth img
        _device.destroyImageView(_depthImage.imageView);
        _allocator.destroyImage(_depthImage.image, _depthImage.allocation);
    });
}

void WaterEngine::resize_swapchain() {
    _device.waitIdle();

    destroy_swapchain();

    int w, h;
    SDL_GetWindowSize(_window, &w, &h);
    _windowExtent.width = w;
    _windowExtent.height = h;

    create_swapchain(_windowExtent.width, _windowExtent.height);

    resize_requested = false;
}


void WaterEngine::destroy_swapchain() {
    _device.destroySwapchainKHR(_swapchain);

    for (auto &_swapchainImageView: _swapchainImageViews) {
        _device.destroyImageView(_swapchainImageView);
    }
}

AllocatedBuffer WaterEngine::create_buffer(size_t allocSize, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage) {
    vk::BufferCreateInfo bufferInfo = vk::BufferCreateInfo().setPNext(nullptr).setSize(allocSize).setUsage(usage);

    vma::AllocationCreateInfo vmaallocInfo =
            vma::AllocationCreateInfo().setUsage(memoryUsage).setFlags(vma::AllocationCreateFlagBits::eMapped);

    AllocatedBuffer newBuffer;

    vk::Result allocResult = _allocator.createBuffer(&bufferInfo, &vmaallocInfo, &newBuffer.buffer,
                                                     &newBuffer.allocation, &newBuffer.info);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create buffer");
    }

    return newBuffer;
}

AllocatedImage WaterEngine::create_image(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage,
                                         bool mipmapped) {
    AllocatedImage newImage;
    newImage.imageFormat = format;
    newImage.imageExtent = size;

    vk::ImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.setMipLevels(static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1);
    }

    vma::AllocationCreateInfo allocinfo = {};
    allocinfo.setUsage(vma::MemoryUsage::eGpuOnly);
    allocinfo.setRequiredFlags(vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal));

    vk::Result allocResult =
            _allocator.createImage(&img_info, &allocinfo, &newImage.image, &newImage.allocation, nullptr);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create image");
    }

    vk::ImageAspectFlags aspectFlag = vk::ImageAspectFlagBits::eColor;
    if (format == vk::Format::eD32Sfloat) {
        aspectFlag = vk::ImageAspectFlagBits::eDepth;
    }

    vk::ImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlag);
    view_info.subresourceRange.setLevelCount(img_info.mipLevels);

    vk::Result createImgViewResult = _device.createImageView(&view_info, nullptr, &newImage.imageView);

    if (createImgViewResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create image view");
    }

    return newImage;
}

AllocatedImage WaterEngine::create_image(void *data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage,
                                         bool mipmapped) {
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadbuffer =
            create_buffer(data_size, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eCpuToGpu);

    memcpy(uploadbuffer.info.pMappedData, data, data_size);

    AllocatedImage new_image = create_image(
            size, format, usage | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            mipmapped);

    immediate_submit([&](vk::CommandBuffer cmd) {
        vkutil::transition_image(cmd, new_image.image, vk::ImageLayout::eUndefined,
                                 vk::ImageLayout::eTransferDstOptimal);

        vk::BufferImageCopy copyRegion =
                vk::BufferImageCopy()
                        .setBufferOffset(0)
                        .setBufferRowLength(0)
                        .setBufferImageHeight(0)

                        .setImageSubresource(vk::ImageSubresourceLayers()
                                                     .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                     .setMipLevel(0)
                                                     .setBaseArrayLayer(0)
                                                     .setLayerCount(1))
                        .setImageExtent(size);

        cmd.copyBufferToImage(uploadbuffer.buffer, new_image.image, vk::ImageLayout::eTransferDstOptimal, 1,
                              &copyRegion);

        vkutil::transition_image(cmd, new_image.image, vk::ImageLayout::eTransferDstOptimal,
                                 vk::ImageLayout::eShaderReadOnlyOptimal);
    });

    destroy_buffer(uploadbuffer);

    return new_image;
}


void WaterEngine::destroy_buffer(const AllocatedBuffer &buffer) {
    _allocator.destroyBuffer(buffer.buffer, buffer.allocation);
}

void WaterEngine::destroy_image(const AllocatedImage &img) {
    _device.destroyImageView(img.imageView);
    _allocator.destroyImage(img.image, img.allocation);
}


GPUMeshBuffers WaterEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices) {
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;

    newSurface.vertexBuffer =
            create_buffer(vertexBufferSize,
                          vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                                  vk::BufferUsageFlagBits::eShaderDeviceAddress,
                          vma::MemoryUsage::eGpuOnly);

    vk::BufferDeviceAddressInfo deviceAddressInfo =
            vk::BufferDeviceAddressInfo().setBuffer(newSurface.vertexBuffer.buffer);
    newSurface.vertexBufferAddress = _device.getBufferAddress(deviceAddressInfo);

    newSurface.indexBuffer = create_buffer(
            indexBufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vma::MemoryUsage::eGpuOnly);

    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                            vma::MemoryUsage::eCpuOnly);

    void *data = static_cast<VmaAllocation>(staging.allocation)->GetMappedData();

    // copy vertex buffer
    memcpy(data, vertices.data(), vertexBufferSize);
    // copy index buffer
    memcpy((char *) data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy vertexCopy{0};
        vertexCopy.setDstOffset(0);
        vertexCopy.setSrcOffset(0);
        vertexCopy.setSize(vertexBufferSize);

        cmd.copyBuffer(staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);

        vk::BufferCopy indexCopy{0};
        indexCopy.setDstOffset(0);
        indexCopy.setSrcOffset(vertexBufferSize);
        indexCopy.setSize(indexBufferSize);

        cmd.copyBuffer(staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
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

void WaterEngine::init_compute_images() {
    vk::Extent3D oceanSize = vk::Extent3D{OCEAN_SIZE_INT, OCEAN_SIZE_INT, 1};
    vk::Extent3D butterflySize = vk::Extent3D{static_cast<uint32_t>(log2(OCEAN_SIZE_INT)), OCEAN_SIZE_INT, 1};
    vk::ImageUsageFlags imageUsages = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment;

    _spectrumImage = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _spectrumNormalImage = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _butterflyImage = create_image(butterflySize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _pingPong0Image = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _pingPong1Image = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _heightmapImage = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);
    _normalMapImage = create_image(oceanSize, vk::Format::eR32G32B32A32Sfloat, imageUsages, false);

    _mainDeletionQueue.push_function([=, this]() {
        destroy_image(_spectrumImage);
        destroy_image(_spectrumNormalImage);
        destroy_image(_butterflyImage);
        destroy_image(_pingPong0Image);
        destroy_image(_pingPong1Image);
        destroy_image(_heightmapImage);
        destroy_image(_normalMapImage);
    });
}

void WaterEngine::init_default_data() {
    // std::array<Vertex, 4> rect_vertices;
    //
    // rect_vertices[0].position = {0.5, -0.5, 0};
    // rect_vertices[1].position = {0.5, 0.5, 0};
    // rect_vertices[2].position = {-0.5, -0.5, 0};
    // rect_vertices[3].position = {-0.5, 0.5, 0};
    //
    // rect_vertices[0].color = {0, 0, 0, 1};
    // rect_vertices[1].color = {0.5, 0.5, 0.5, 1};
    // rect_vertices[2].color = {1, 0, 0, 1};
    // rect_vertices[3].color = {0, 1, 0, 1};
    //
    // rect_vertices[0].uv_x = 1;
    // rect_vertices[0].uv_y = 0;
    // rect_vertices[1].uv_x = 0;
    // rect_vertices[1].uv_y = 0;
    // rect_vertices[2].uv_x = 1;
    // rect_vertices[2].uv_y = 1;
    // rect_vertices[3].uv_x = 0;
    // rect_vertices[3].uv_y = 1;
    //
    // std::array<uint32_t, 6> rect_indices;
    //
    // rect_indices[0] = 0;
    // rect_indices[1] = 1;
    // rect_indices[2] = 2;
    //
    // rect_indices[3] = 2;
    // rect_indices[4] = 1;
    // rect_indices[5] = 3;
    //
    // rectangle = uploadMesh(rect_indices, rect_vertices);
    //
    uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
    _whiteImage = create_image((void *) &white, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm,
                               vk::ImageUsageFlagBits::eSampled);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
    _greyImage = create_image((void *) &grey, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm,
                              vk::ImageUsageFlagBits::eSampled);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
    _blackImage = create_image((void *) &black, vk::Extent3D{1, 1, 1}, vk::Format::eR8G8B8A8Unorm,
                               vk::ImageUsageFlagBits::eSampled);

    // Checkerboard image
    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1, 0, 1, 1));
    std::array<uint32_t, 16 * 16> pixels;
    for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), vk::Extent3D{16, 16, 1}, vk::Format::eR8G8B8A8Unorm,
                                           vk::ImageUsageFlagBits::eSampled);

    vk::SamplerCreateInfo sample =
            vk::SamplerCreateInfo().setMagFilter(vk::Filter::eNearest).setMinFilter(vk::Filter::eNearest);

    _defaultSamplerNearest = _device.createSampler(sample, nullptr);

    sample.setMagFilter(vk::Filter::eLinear);
    sample.setMinFilter(vk::Filter::eLinear);

    _defaultSamplerLinear = _device.createSampler(sample, nullptr);

    _mainDeletionQueue.push_function([&, this]() {
        _device.destroySampler(_defaultSamplerNearest);
        _device.destroySampler(_defaultSamplerLinear);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
    });

    GLTFMetallic_Roughness::MaterialResources materialResources;
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;

    AllocatedBuffer materialConstants =
            create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), vk::BufferUsageFlagBits::eUniformBuffer,
                          vma::MemoryUsage::eCpuToGpu);

    GLTFMetallic_Roughness::MaterialConstants *sceneUniformData =
            (GLTFMetallic_Roughness::MaterialConstants *) static_cast<VmaAllocation>(
                    (*materialConstants.allocation).GetMappedData());
    sceneUniformData->colorFactors = glm::vec4{1, 1, 1, 1};
    sceneUniformData->metal_rough_factors = glm::vec4{1, 0.5, 0, 0};

    _mainDeletionQueue.push_function([=, this]() { destroy_buffer(materialConstants); });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    defaultData = metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources,
                                                    globalDescriptorAllocator);

    testMeshes = loadGltfMeshes(this, "..\\assets\\basicmesh.glb").value();

    // for (auto &m: testMeshes) {
    //     std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>();
    //     newNode->mesh = m;
    //
    //     newNode->localTransform = glm::mat4{1.f};
    //     newNode->worldTransform = glm::mat4{1.f};
    //
    //     for (auto &s: newNode->mesh->surfaces) {
    //         s.material = std::make_shared<GLTFMaterial>(defaultData);
    //     }
    //
    //     loadedNodes[m->name] = std::move(newNode);
    // }
}

void WaterEngine::init_compute_descriptors() {
    std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {{vk::DescriptorType::eStorageImage, 1}};

    // Spectrum descriptor
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eStorageImage);
        builder.add_binding(1, vk::DescriptorType::eStorageImage);
        _spectrumImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eCompute);
    }

    // Butterfly descriptor
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eStorageImage);
        _butterflyImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eCompute);
    }

    // FFT descriptor
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eStorageImage);
        builder.add_binding(1, vk::DescriptorType::eStorageImage);
        builder.add_binding(2, vk::DescriptorType::eStorageImage);
        _fftImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eCompute, nullptr,
                                                  vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR);
    }

    _spectrumImageDescriptors = globalDescriptorAllocator.allocate(_device, _spectrumImageDescriptorLayout);
    _butterflyImageDescriptors = globalDescriptorAllocator.allocate(_device, _butterflyImageDescriptorLayout);

    vk::DescriptorImageInfo spectrumImgInfo =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_spectrumImage.imageView);
    vk::DescriptorImageInfo spectrumNormalImgInfo = vk::DescriptorImageInfo()
                                                            .setImageLayout(vk::ImageLayout::eGeneral)
                                                            .setImageView(_spectrumNormalImage.imageView);
    vk::DescriptorImageInfo butterflyImgInfo =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_butterflyImage.imageView);

    std::array spectrumImageWrites = {
            vk::WriteDescriptorSet()
                    .setPNext(nullptr)
                    .setDstBinding(0)
                    .setDstSet(_spectrumImageDescriptors)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&spectrumImgInfo),
            vk::WriteDescriptorSet()
                    .setPNext(nullptr)
                    .setDstBinding(1)
                    .setDstSet(_spectrumImageDescriptors)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&spectrumNormalImgInfo),
    };

    vk::WriteDescriptorSet butterflyImageWrite = vk::WriteDescriptorSet()
                                                         .setPNext(nullptr)
                                                         .setDstBinding(0)
                                                         .setDstSet(_butterflyImageDescriptors)
                                                         .setDescriptorCount(1)
                                                         .setDescriptorType(vk::DescriptorType::eStorageImage)
                                                         .setPImageInfo(&butterflyImgInfo);

    _device.updateDescriptorSets(spectrumImageWrites.size(), spectrumImageWrites.data(), 0, nullptr);
    _device.updateDescriptorSets(1, &butterflyImageWrite, 0, nullptr);

    _mainDeletionQueue.push_function([&]() {
        _device.destroyDescriptorSetLayout(_spectrumImageDescriptorLayout);
        _device.destroyDescriptorSetLayout(_butterflyImageDescriptorLayout);
        _device.destroyDescriptorSetLayout(_fftImageDescriptorLayout);
    });
}

void WaterEngine::init_descriptors() {
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
            {vk::DescriptorType::eStorageImage, 3},
            {vk::DescriptorType::eUniformBuffer, 3},
    };

    globalDescriptorAllocator.init(_device, 10, sizes);

    // Scene descriptor
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eUniformBuffer);
        _gpuSceneDataDescriptorLayout =
                builder.build(_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    }

    // Single image
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, vk::DescriptorType::eCombinedImageSampler);
        _singleImageDescriptorLayout = builder.build(_device, vk::ShaderStageFlagBits::eFragment);
    }

    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);
        _device.destroyDescriptorSetLayout(_singleImageDescriptorLayout);
        _device.destroyDescriptorSetLayout(_gpuSceneDataDescriptorLayout);
    });

    for (int i = 0; i < FRAME_OVERLAP; i++) {
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
                {vk::DescriptorType::eStorageImage, 3},
                {vk::DescriptorType::eStorageBuffer, 3},
                {vk::DescriptorType::eUniformBuffer, 3},
                {vk::DescriptorType::eCombinedImageSampler, 4},
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() { _frames[i]._frameDescriptors.destroy_pools(_device); });
    }
}


void WaterEngine::draw_background(vk::CommandBuffer cmd) {
    run_spectrum(cmd);

    run_butterfly(cmd);

    run_ifft(cmd, _spectrumImage.image, &_heightmapImage.image);

    run_ifft(cmd, _spectrumNormalImage.image, &_normalMapImage.image);
}

void WaterEngine::draw_geometry(vk::CommandBuffer cmd) {
    vk::RenderingAttachmentInfo colorAttachment =
            vkinit::attachment_info(_drawImage.imageView, nullptr, vk::ImageLayout::eColorAttachmentOptimal);
    vk::RenderingAttachmentInfo depthAttachment =
            vkinit::depth_attachment_info(_depthImage.imageView, vk::ImageLayout::eDepthAttachmentOptimal);

    vk::RenderingInfo renderInfo = vkinit::rendering_info(_windowExtent, &colorAttachment, &depthAttachment);
    cmd.beginRendering(renderInfo);

    vk::Viewport viewport = vk::Viewport()
                                    .setX(0.0f)
                                    .setY(0.0f)
                                    .setWidth(static_cast<float>(_drawExtent.width))
                                    .setHeight(static_cast<float>(_drawExtent.height))
                                    .setMinDepth(0.0f)
                                    .setMaxDepth(1.0f);

    cmd.setViewport(0, 1, &viewport);

    vk::Rect2D scissor = vk::Rect2D().setOffset(vk::Offset2D(0, 0)).setExtent(_drawExtent);

    cmd.setScissor(0, 1, &scissor);

    AllocatedBuffer gpuSceneDataBuffer =
            create_buffer(sizeof(GPUSceneData), vk::BufferUsageFlagBits::eUniformBuffer, vma::MemoryUsage::eCpuToGpu);

    get_current_frame()._deletionQueue.push_function([=, this]() { destroy_buffer(gpuSceneDataBuffer); });

    GPUSceneData *sceneUniformData =
            static_cast<GPUSceneData *>(static_cast<VmaAllocation>(gpuSceneDataBuffer.allocation)->GetMappedData());
    *sceneUniformData = sceneData;

    vk::DescriptorSet globalDescriptor =
            get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, vk::DescriptorType::eUniformBuffer);
    writer.update_set(_device, globalDescriptor);

    // for (const RenderObject &draw: mainDrawContext.OpaqueSurfaces) {
    //     cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->pipeline);
    //     cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->layout, 0, 1,
    //                            &globalDescriptor, 0, nullptr);
    //     cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->layout, 1, 1,
    //                            &draw.material->materialSet, 0, nullptr);
    //
    //     cmd.bindIndexBuffer(draw.indexBuffer, 0, vk::IndexType::eUint32);
    //
    //     GPUDrawPushConstants pushConstants{};
    //     pushConstants.vertexBuffer = draw.vertexBufferAddress;
    //     pushConstants.worldMatrix = draw.transform;
    //     cmd.pushConstants(draw.material->pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0,
    //                       sizeof(GPUDrawPushConstants), &pushConstants);
    //
    //     cmd.drawIndexed(draw.indexCount, 1, draw.firstIndex, 0, 0);
    // }
    auto draw = [&](const RenderObject &draw) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->layout, 0, 1,
                               &globalDescriptor, 0, nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, draw.material->pipeline->layout, 1, 1,
                               &draw.material->materialSet, 0, nullptr);

        cmd.bindIndexBuffer(draw.indexBuffer, 0, vk::IndexType::eUint32);

        GPUDrawPushConstants pushConstants{};
        pushConstants.vertexBuffer = draw.vertexBufferAddress;
        pushConstants.worldMatrix = draw.transform;
        cmd.pushConstants(draw.material->pipeline->layout, vk::ShaderStageFlagBits::eVertex, 0,
                          sizeof(GPUDrawPushConstants), &pushConstants);

        cmd.drawIndexed(draw.indexCount, 1, draw.firstIndex, 0, 0);
    };

    for (auto &r: mainDrawContext.OpaqueSurfaces) {
        draw(r);
    }

    for (auto &r: mainDrawContext.TransparentSurfaces) {
        draw(r);
    }

    cmd.endRendering();
}


void WaterEngine::init_pipelines() {
    // Compute pipelines
    init_background_pipelines();
    // Graphics pipelines

    metalRoughMaterial.build_pipelines(this);
}

void WaterEngine::init_background_pipelines() {
    // Spectrum init
    vk::PipelineLayoutCreateInfo computeLayout = vk::PipelineLayoutCreateInfo()
                                                         .setPNext(nullptr)
                                                         .setPSetLayouts(&_spectrumImageDescriptorLayout)
                                                         .setSetLayoutCount(1);

    vk::PushConstantRange pushConstant = vk::PushConstantRange()
                                                 .setOffset(0)
                                                 .setSize(sizeof(OceanParameters))
                                                 .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    computeLayout.setPPushConstantRanges(&pushConstant);
    computeLayout.setPushConstantRangeCount(1);

    _spectrumPipelineLayout = _device.createPipelineLayout(computeLayout);

    vk::ShaderModule spectrumShader;
    if (!vkutil::load_shader_module("../shaders/spectrum2.comp.spv", _device, &spectrumShader)) {
        fmt::print("Error when building the spectrum shader \n");
    }


    vk::PipelineShaderStageCreateInfo stageInfo = vk::PipelineShaderStageCreateInfo()
                                                          .setPNext(nullptr)
                                                          .setStage(vk::ShaderStageFlagBits::eCompute)
                                                          .setModule(spectrumShader)
                                                          .setPName("main");

    vk::ComputePipelineCreateInfo computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_spectrumPipelineLayout).setStage(stageInfo);

    ComputeOceanEffect ocean;
    ocean.layout = _spectrumPipelineLayout;
    ocean.name = "ocean";

    vk::Result createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &ocean.pipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create spectrum pipeline");
    }

    oceanEffects.push_back(ocean);

    // Butterfly init

    vk::PipelineLayoutCreateInfo butterflyLayout = vk::PipelineLayoutCreateInfo()
                                                           .setPNext(nullptr)
                                                           .setPSetLayouts(&_butterflyImageDescriptorLayout)
                                                           .setSetLayoutCount(1);

    vk::PushConstantRange butterflyPushConstant = vk::PushConstantRange()
                                                          .setOffset(0)
                                                          .setSize(sizeof(ButterflyParameters))
                                                          .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    butterflyLayout.setPPushConstantRanges(&butterflyPushConstant);
    butterflyLayout.setPushConstantRangeCount(1);

    _butterflyPipelineLayout = _device.createPipelineLayout(butterflyLayout);

    vk::ShaderModule butterflyShader;
    if (!vkutil::load_shader_module("../shaders/butterfly.comp.spv", _device, &butterflyShader)) {
        fmt::print("Error when building the butterfly shader \n");
    }

    stageInfo = vk::PipelineShaderStageCreateInfo()
                        .setPNext(nullptr)
                        .setStage(vk::ShaderStageFlagBits::eCompute)
                        .setModule(butterflyShader)
                        .setPName("main");

    computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_butterflyPipelineLayout).setStage(stageInfo);

    createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &_butterflyPipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create butterfly pipeline");
    }

    // FFT init

    vk::PipelineLayoutCreateInfo fftLayout = vk::PipelineLayoutCreateInfo()
                                                     .setPNext(nullptr)
                                                     .setPSetLayouts(&_fftImageDescriptorLayout)
                                                     .setSetLayoutCount(1);

    vk::PushConstantRange fftPushConstant = vk::PushConstantRange()
                                                    .setOffset(0)
                                                    .setSize(sizeof(IFFTParameters))
                                                    .setStageFlags(vk::ShaderStageFlagBits::eCompute);

    fftLayout.setPPushConstantRanges(&fftPushConstant);
    fftLayout.setPushConstantRangeCount(1);

    _fftPipelineLayout = _device.createPipelineLayout(fftLayout);

    vk::ShaderModule fftShader;
    if (!vkutil::load_shader_module("../shaders/fft.comp.spv", _device, &fftShader)) {
        fmt::print("Error when building the fft shader \n");
    }

    stageInfo = vk::PipelineShaderStageCreateInfo()
                        .setPNext(nullptr)
                        .setStage(vk::ShaderStageFlagBits::eCompute)
                        .setModule(fftShader)
                        .setPName("main");

    computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_fftPipelineLayout).setStage(stageInfo);

    createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &_fftPipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create fft pipeline");
    }

    // Copy init

    vk::PipelineLayoutCreateInfo copyLayout = vk::PipelineLayoutCreateInfo()
                                                      .setPNext(nullptr)
                                                      .setPSetLayouts(&_fftImageDescriptorLayout)
                                                      .setSetLayoutCount(1);

    _copyPipelineLayout = _device.createPipelineLayout(copyLayout);

    vk::ShaderModule copyShader;
    if (!vkutil::load_shader_module("../shaders/copy_ppong.comp.spv", _device, &copyShader)) {
        fmt::print("Error when building the copy shader \n");
    }

    stageInfo = vk::PipelineShaderStageCreateInfo()
                        .setPNext(nullptr)
                        .setStage(vk::ShaderStageFlagBits::eCompute)
                        .setModule(copyShader)
                        .setPName("main");

    computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_copyPipelineLayout).setStage(stageInfo);

    createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &_copyPipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create copy pipeline");
    }

    // Permute init

    vk::PipelineLayoutCreateInfo permuteLayout = vk::PipelineLayoutCreateInfo()
                                                         .setPNext(nullptr)
                                                         .setPSetLayouts(&_fftImageDescriptorLayout)
                                                         .setSetLayoutCount(1);

    permuteLayout.setPPushConstantRanges(&butterflyPushConstant);
    permuteLayout.setPushConstantRangeCount(1);

    _permutePipelineLayout = _device.createPipelineLayout(permuteLayout);

    vk::ShaderModule permuteShader;
    if (!vkutil::load_shader_module("../shaders/permute.comp.spv", _device, &permuteShader)) {
        fmt::print("Error when building the permute shader \n");
    }

    stageInfo = vk::PipelineShaderStageCreateInfo()
                        .setPNext(nullptr)
                        .setStage(vk::ShaderStageFlagBits::eCompute)
                        .setModule(permuteShader)
                        .setPName("main");

    computePipelineCreateInfo =
            vk::ComputePipelineCreateInfo().setPNext(nullptr).setLayout(_permutePipelineLayout).setStage(stageInfo);

    createPipelineResult =
            _device.createComputePipelines(VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &_permutePipeline);
    if (createPipelineResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create permute pipeline");
    }

    // Cleanup

    _device.destroyShaderModule(spectrumShader);
    _device.destroyShaderModule(butterflyShader);
    _device.destroyShaderModule(fftShader);
    _device.destroyShaderModule(copyShader);
    _device.destroyShaderModule(permuteShader);

    _mainDeletionQueue.push_function([=, this]() {
        // Spectrum
        _device.destroyPipelineLayout(_spectrumPipelineLayout);
        _device.destroyPipeline(ocean.pipeline);
        // Butterfly
        _device.destroyPipelineLayout(_butterflyPipelineLayout);
        _device.destroyPipeline(_butterflyPipeline);
        // FFT
        _device.destroyPipelineLayout(_fftPipelineLayout);
        _device.destroyPipeline(_fftPipeline);
        // Copy
        _device.destroyPipelineLayout(_copyPipelineLayout);
        _device.destroyPipeline(_copyPipeline);
        // Permute
        _device.destroyPipelineLayout(_permutePipelineLayout);
        _device.destroyPipeline(_permutePipeline);
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

void WaterEngine::run_spectrum(vk::CommandBuffer cmd) {
    ComputeOceanEffect &effect = oceanEffects[currentOceanEffect];

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, effect.pipeline);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, _spectrumPipelineLayout, 0, 1, &_spectrumImageDescriptors,
                           0, nullptr);

    cmd.pushConstants(_spectrumPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(OceanParameters),
                      &effect.parameters);

    cmd.dispatch(std::ceil(_spectrumExtent.width / 32.0), std::ceil(_spectrumExtent.height / 32.0), 1);

    vk::MemoryBarrier2 memoryBarrier = vk::MemoryBarrier2()
                                               .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                                               .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
                                               .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                                               .setDstAccessMask(vk::AccessFlagBits2::eShaderRead);

    vk::DependencyInfo dependencyInfo = vk::DependencyInfo()
                                                .setDependencyFlags(vk::DependencyFlagBits::eByRegion)
                                                .setMemoryBarrierCount(1)
                                                .setPMemoryBarriers(&memoryBarrier);

    cmd.pipelineBarrier2(&dependencyInfo);
}

void WaterEngine::run_butterfly(vk::CommandBuffer cmd) {
    vkutil::transition_image(cmd, _spectrumImage.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, _butterflyImage.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    uint32_t resolution = oceanEffects[currentOceanEffect].parameters.resolution.x;
    _butterflyParameters.resolution = resolution;


    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, _butterflyPipeline);

    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, _butterflyPipelineLayout, 0, 1, &_butterflyImageDescriptors,
                           0, nullptr);

    cmd.pushConstants(_butterflyPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ButterflyParameters),
                      &_butterflyParameters);

    cmd.dispatch(std::ceil(log2(resolution) / 16.0), std::ceil(resolution / 16.0), 1);

    vk::MemoryBarrier2 memoryBarrier = vk::MemoryBarrier2()
                                               .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                                               .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
                                               .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                                               .setDstAccessMask(vk::AccessFlagBits2::eShaderRead);

    vk::DependencyInfo dependencyInfo = vk::DependencyInfo()
                                                .setDependencyFlags(vk::DependencyFlagBits::eByRegion)
                                                .setMemoryBarrierCount(1)
                                                .setPMemoryBarriers(&memoryBarrier);

    cmd.pipelineBarrier2(&dependencyInfo);
}

void WaterEngine::run_ifft(vk::CommandBuffer cmd, vk::Image &input, vk::Image *output) {
    int pingPong = 0;
    vkutil::transition_image(cmd, input, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
    vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eUndefined,
                             vk::ImageLayout::eTransferDstOptimal);

    vkutil::copy_image_to_image(cmd, input, _pingPong0Image.image, vk::Extent2D{OCEAN_SIZE_INT, OCEAN_SIZE_INT},
                                vk::Extent2D{OCEAN_SIZE_INT, OCEAN_SIZE_INT});

    vkutil::transition_image(cmd, input, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eTransferDstOptimal,
                             vk::ImageLayout::eGeneral);

    // Ensure images are in general layout
    vkutil::transition_image(cmd, *output, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, _pingPong1Image.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

    if (output != nullptr) {
        run_copy(cmd);

        // Clear the image (making it blank)
        vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eGeneral,
                                 vk::ImageLayout::eTransferDstOptimal);

        // vk::ClearColorValue clearColor(std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f});
        // vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        // cmd.clearColorImage(_pingPong0Image.image, vk::ImageLayout::eTransferDstOptimal, clearColor, range);

        vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eTransferDstOptimal,
                                 vk::ImageLayout::eGeneral);
    }

    vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, _pingPong1Image.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);

    // IFFT passes
    int resolution = OCEAN_SIZE_INT;
    int log2N = static_cast<int>(log2(resolution));

    // log2(N) horizontal passes
    for (int stage = 0; stage < log2N; stage++) {
        ifft_pass(cmd, stage, 0, resolution, pingPong);
        pingPong = (pingPong + 1) % 2;
    }

    // log2(N) vertical passes
    for (int stage = 0; stage < log2N; stage++) {
        ifft_pass(cmd, stage, 1, resolution, pingPong);
        pingPong = (pingPong + 1) % 2;
    }

    run_copy(cmd);

    run_permute(cmd);

    vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eGeneral,
                             vk::ImageLayout::eTransferSrcOptimal);
    vkutil::transition_image(cmd, *output, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal);

    vkutil::copy_image_to_image(cmd, _pingPong0Image.image, *output, vk::Extent2D{OCEAN_SIZE_INT, OCEAN_SIZE_INT},
                                vk::Extent2D{OCEAN_SIZE_INT, OCEAN_SIZE_INT});

    vkutil::transition_image(cmd, _pingPong0Image.image, vk::ImageLayout::eTransferSrcOptimal,
                             vk::ImageLayout::eGeneral);
    vkutil::transition_image(cmd, *output, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
}

void WaterEngine::ifft_pass(vk::CommandBuffer cmd, uint32_t stage, uint32_t direction, uint32_t resolution,
                            uint32_t pingPong) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, _fftPipeline);

    vk::DescriptorImageInfo pingPong0Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong0Image.imageView);

    vk::DescriptorImageInfo pingPong1Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong1Image.imageView);

    vk::DescriptorImageInfo butterflyInfo =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_butterflyImage.imageView);

    vk::DescriptorImageInfo *sourceInfo = &pingPong0Info;
    vk::DescriptorImageInfo *destinationInfo = &pingPong1Info;

    std::array descriptorWrites = {
            vk::WriteDescriptorSet()
                    .setDstBinding(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(sourceInfo),
            vk::WriteDescriptorSet()
                    .setDstBinding(1)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(destinationInfo),
            vk::WriteDescriptorSet()
                    .setDstBinding(2)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&butterflyInfo),
    };

    vk::PushDescriptorSetInfo pushDescriptorSetsInfo = vk::PushDescriptorSetInfo()
                                                               .setStageFlags(vk::ShaderStageFlagBits::eCompute)
                                                               .setLayout(_fftPipelineLayout)
                                                               .setDescriptorWriteCount(descriptorWrites.size())
                                                               .setPDescriptorWrites(descriptorWrites.data());

    cmd.pushDescriptorSet2(pushDescriptorSetsInfo);

    _fftParameters.stage = stage;
    _fftParameters.direction = direction;
    _fftParameters.resolution = resolution;
    _fftParameters.pingPong = pingPong;

    cmd.pushConstants(_fftPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(IFFTParameters),
                      &_fftParameters);

    cmd.dispatch(std::ceil(resolution / 16.0f), std::ceil(resolution / 16.0f), 1);

    vk::MemoryBarrier2 memoryBarrier =
            vk::MemoryBarrier2()
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead)
                    .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead);

    vk::DependencyInfo dependencyInfo = vk::DependencyInfo()
                                                .setDependencyFlags(vk::DependencyFlagBits::eByRegion) // If applicable
                                                .setMemoryBarrierCount(1)
                                                .setPMemoryBarriers(&memoryBarrier);

    cmd.pipelineBarrier2(dependencyInfo);
}

void WaterEngine::run_copy(vk::CommandBuffer cmd) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, _copyPipeline);

    vk::DescriptorImageInfo pingPong0Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong0Image.imageView);

    vk::DescriptorImageInfo pingPong1Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong1Image.imageView);

    std::array descriptorWrites = {
            vk::WriteDescriptorSet()
                    .setDstBinding(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&pingPong0Info),
            vk::WriteDescriptorSet()
                    .setDstBinding(1)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&pingPong1Info),
    };

    vk::PushDescriptorSetInfo pushDescriptorSetsInfo = vk::PushDescriptorSetInfo()
                                                               .setStageFlags(vk::ShaderStageFlagBits::eCompute)
                                                               .setLayout(_copyPipelineLayout)
                                                               .setDescriptorWriteCount(descriptorWrites.size())
                                                               .setPDescriptorWrites(descriptorWrites.data());

    cmd.pushDescriptorSet2(pushDescriptorSetsInfo);

    int resolution = OCEAN_SIZE_INT;
    cmd.dispatch(std::ceil(resolution / 16.0f), std::ceil(resolution / 16.0f), 1);

    vk::MemoryBarrier2 memoryBarrier =
            vk::MemoryBarrier2()
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead)
                    .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead);

    vk::DependencyInfo dependencyInfo = vk::DependencyInfo()
                                                .setDependencyFlags(vk::DependencyFlagBits::eByRegion) // If applicable
                                                .setMemoryBarrierCount(1)
                                                .setPMemoryBarriers(&memoryBarrier);

    cmd.pipelineBarrier2(dependencyInfo);
}

void WaterEngine::run_permute(vk::CommandBuffer cmd) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, _permutePipeline);

    vk::DescriptorImageInfo pingPong0Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong0Image.imageView);

    vk::DescriptorImageInfo pingPong1Info =
            vk::DescriptorImageInfo().setImageLayout(vk::ImageLayout::eGeneral).setImageView(_pingPong1Image.imageView);

    std::array descriptorWrites = {
            vk::WriteDescriptorSet()
                    .setDstBinding(0)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&pingPong0Info),
            vk::WriteDescriptorSet()
                    .setDstBinding(1)
                    .setDescriptorCount(1)
                    .setDescriptorType(vk::DescriptorType::eStorageImage)
                    .setPImageInfo(&pingPong1Info),
    };

    vk::PushDescriptorSetInfo pushDescriptorSetsInfo = vk::PushDescriptorSetInfo()
                                                               .setStageFlags(vk::ShaderStageFlagBits::eCompute)
                                                               .setLayout(_permutePipelineLayout)
                                                               .setDescriptorWriteCount(descriptorWrites.size())
                                                               .setPDescriptorWrites(descriptorWrites.data());

    cmd.pushDescriptorSet2(pushDescriptorSetsInfo);

    cmd.pushConstants(_permutePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ButterflyParameters),
                      &_butterflyParameters);

    int resolution = OCEAN_SIZE_INT;
    cmd.dispatch(std::ceil(resolution / 16.0f), std::ceil(resolution / 16.0f), 1);

    vk::MemoryBarrier2 memoryBarrier =
            vk::MemoryBarrier2()
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead)
                    .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
                    .setDstAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead);

    vk::DependencyInfo dependencyInfo = vk::DependencyInfo()
                                                .setDependencyFlags(vk::DependencyFlagBits::eByRegion) // If applicable
                                                .setMemoryBarrierCount(1)
                                                .setPMemoryBarriers(&memoryBarrier);

    cmd.pipelineBarrier2(dependencyInfo);
}

void WaterEngine::update_scene() {
    mainCamera.update();
    mainDrawContext.OpaqueSurfaces.clear();
    mainDrawContext.TransparentSurfaces.clear();

    // for (auto &m: loadedNodes) {
    //     m.second->Draw(glm::mat4{1.f}, mainDrawContext);
    // }
    //
    // for (int x = -3; x < 3; x++) {
    //
    //     glm::mat4 scale = glm::scale(glm::vec3{0.2});
    //     glm::mat4 translation = glm::translate(glm::vec3{x, 1, 0});
    //
    //     loadedNodes["Cube"]->Draw(translation * scale, mainDrawContext);
    // }


    glm::mat4 view = mainCamera.getViewMatrix();

    // camera projection
    glm::mat4 projection = glm::perspective(glm::radians(70.f),
                                            (float) _windowExtent.width / (float) _windowExtent.height, 10000.f, 0.1f);

    projection[1][1] *= -1;

    sceneData.view = view;
    sceneData.proj = projection;
    sceneData.viewproj = projection * view;

    sceneData.ambientColor = glm::vec4(.1f);
    sceneData.sunlightColor = glm::vec4(1.f);
    sceneData.sunlightDirection = glm::vec4(0, 1, 0.5, 1.f);

    loadScenes["structure"]->Draw(glm::mat4{1.f}, mainDrawContext);
}


void GLTFMetallic_Roughness::build_pipelines(WaterEngine *engine) {
    vk::ShaderModule meshFragShader;
    if (!vkutil::load_shader_module("..\\shaders\\mesh.frag.spv", engine->_device, &meshFragShader)) {
        throw std::runtime_error("Failed to load mesh.frag.spv");
    }

    vk::ShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("..\\shaders\\mesh.vert.spv", engine->_device, &meshVertexShader)) {
        throw std::runtime_error("Failed to load mesh.vert.spv");
    }

    vk::PushConstantRange matrixRange = vk::PushConstantRange()
                                                .setOffset(0)
                                                .setSize(sizeof(GPUDrawPushConstants))
                                                .setStageFlags(vk::ShaderStageFlagBits::eVertex);

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, vk::DescriptorType::eUniformBuffer);
    layoutBuilder.add_binding(1, vk::DescriptorType::eCombinedImageSampler);
    layoutBuilder.add_binding(2, vk::DescriptorType::eCombinedImageSampler);

    materialLayout =
            layoutBuilder.build(engine->_device, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);

    vk::DescriptorSetLayout layouts[] = {engine->_gpuSceneDataDescriptorLayout, materialLayout};

    vk::PipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setSetLayoutCount(2);
    mesh_layout_info.setPSetLayouts(layouts).setPPushConstantRanges(&matrixRange).setPushConstantRangeCount(1);

    vk::PipelineLayout newLayout = engine->_device.createPipelineLayout(mesh_layout_info, nullptr);

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(vk::PrimitiveTopology::eTriangleList);
    pipelineBuilder.set_polygon_mode(vk::PolygonMode::eFill);
    pipelineBuilder.set_cull_mode(vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, vk::CompareOp::eGreaterOrEqual);

    // render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout
    pipelineBuilder._pipelineLayout = newLayout;

    // build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // create transparent variant
    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(false, vk::CompareOp::eGreaterOrEqual);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    engine->_device.destroyShaderModule(meshFragShader);
    engine->_device.destroyShaderModule(meshVertexShader);
}

MaterialInstance GLTFMetallic_Roughness::write_material(vk::Device device, MaterialPass pass,
                                                        const MaterialResources &resources,
                                                        DescriptorAllocatorGrowable &descriptorAllocator) {
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    } else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset,
                        vk::DescriptorType::eUniformBuffer);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler,
                       vk::ImageLayout::eShaderReadOnlyOptimal, vk::DescriptorType::eCombinedImageSampler);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler,
                       vk::ImageLayout::eShaderReadOnlyOptimal, vk::DescriptorType::eCombinedImageSampler);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void GLTFMetallic_Roughness::clear_resources(vk::Device device) {
    device.destroyDescriptorSetLayout(materialLayout);
    device.destroyPipelineLayout(transparentPipeline.layout);

    device.destroyPipeline(transparentPipeline.pipeline);
    device.destroyPipeline(opaquePipeline.pipeline);
}

void MeshNode::Draw(const glm::mat4 &topMatrix, DrawContext &ctx) {
    glm::mat4 nodeMatrix = topMatrix * worldTransform;

    for (auto &s: mesh->surfaces) {
        RenderObject def;
        def.indexCount = s.count;
        def.firstIndex = s.startIndex;
        def.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
        def.material = &s.material->data;

        def.transform = nodeMatrix;
        def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

        ctx.OpaqueSurfaces.push_back(def);
    }

    Node::Draw(topMatrix, ctx);
}
