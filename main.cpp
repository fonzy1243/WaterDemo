#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// For C++ Vulkan bindings
#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

// To help with Vulkan boilerplate
#include <VkBootstrap.h>
#include <VkBootstrapDispatch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#if defined(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
#endif


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::string MODEL_PATH = "models/base.obj";
const std::string TEXTURE_PATH = "textures/texture_diffuse.png";

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription()
                .setBinding(0)
                .setStride(sizeof(Vertex))
                .setInputRate(vk::VertexInputRate::eVertex);
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {vk::VertexInputAttributeDescription()
                        .setBinding(0)
                        .setLocation(0)
                        .setFormat(vk::Format::eR32G32B32Sfloat)
                        .setOffset(offsetof(Vertex, pos)),
                vk::VertexInputAttributeDescription()
                        .setBinding(0)
                        .setLocation(1)
                        .setFormat(vk::Format::eR32G32B32Sfloat)
                        .setOffset(offsetof(Vertex, color)),
                vk::VertexInputAttributeDescription()
                        .setBinding(0)
                        .setLocation(2)
                        .setFormat(vk::Format::eR32G32Sfloat)
                        .setOffset(offsetof(Vertex, texCoord))};
    }

    bool operator==(const Vertex &other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
} // namespace std

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class WaterDemoApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow *window;

    vkb::Instance vkbInstance;
    vkb::Device vkbDevice;
    vkb::PhysicalDevice vkbPhysicalDevice;
    vkb::Swapchain vkbSwapchain;

    vk::Instance instance;
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceProperties physicalDeviceProperties;
    std::vector<VkQueueFamilyProperties> queueFamilyProperties;
    vk::SurfaceKHR surface;

    vk::DebugUtilsMessengerEXT debugMessenger;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    vk::RenderPass renderPass;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;

    vk::CommandPool commandPool;

    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;

    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "WaterDemo", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
        auto app = reinterpret_cast<WaterDemoApplication *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan() {
        createInstance();
        // setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();
    }

    void cleanupSwapChain() {
        device.destroyImageView(depthImageView);
        device.destroyImage(depthImage);
        device.freeMemory(depthImageMemory);

        for (auto framebuffer: swapChainFramebuffers) {
            device.destroyFramebuffer(framebuffer);
        }

        for (auto imageView: swapChainImageViews) {
            device.destroyImageView(imageView);
        }

        device.destroySwapchainKHR(swapChain);
    }

    void cleanup() {
        cleanupSwapChain();

        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroyBuffer(uniformBuffers[i]);
            device.freeMemory(uniformBuffersMemory[i]);
        }

        device.destroyDescriptorPool(descriptorPool);

        device.destroySampler(textureSampler);
        device.destroyImageView(textureImageView);

        device.destroyImage(textureImage);
        device.freeMemory(textureImageMemory);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        device.destroyBuffer(indexBuffer);
        device.freeMemory(indexBufferMemory);

        device.destroyBuffer(vertexBuffer);
        device.freeMemory(vertexBufferMemory);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }

        device.destroyCommandPool(commandPool);

        device = nullptr;

        if (enableValidationLayers) {
            instance.destroyDebugUtilsMessengerEXT(debugMessenger);
        }

        instance.destroySurfaceKHR(surface);
        instance = nullptr;

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();

        cleanupSwapChain();

        createSwapChain();
        createDepthResources();
        createFramebuffers();
    }

    void createInstance() {
        vkb::InstanceBuilder builder;

        auto inst_ret = builder.set_app_name("WaterDemo")
                                .set_engine_name("No Engine")
                                .set_app_version(1, 0, 0)
                                .set_engine_version(1, 0, 0)
                                .request_validation_layers(enableValidationLayers)
                                .use_default_debug_messenger()
                                .require_api_version(1, 3, 0)
                                .build();

        if (!inst_ret) {
            throw std::runtime_error("Failed to create instance: " + std::string(inst_ret.error().message()));
        }

        vkbInstance = inst_ret.value();
        instance = vkbInstance.instance;

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

        if (enableValidationLayers) {
            debugMessenger = vkbInstance.debug_messenger;
        }
    }

    void createSurface() {
        VkSurfaceKHR rawSurface;
        if (glfwCreateWindowSurface(instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }

        surface = rawSurface;
    }

    void pickPhysicalDevice() {
        vkb::PhysicalDeviceSelector selector{vkbInstance};

        vk::PhysicalDeviceFeatures features;
        features.samplerAnisotropy = VK_TRUE;

        auto phys_ret = selector.set_surface(surface)
                                .add_required_extensions(deviceExtensions)
                                .set_minimum_version(1, 0)
                                .set_required_features(features)
                                .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                                .require_present()
                                .select();

        if (!phys_ret) {
            throw std::runtime_error("Failed to create physical device: " + std::string(phys_ret.error().message()));
        }

        vkbPhysicalDevice = phys_ret.value();
        physicalDevice = phys_ret.value().physical_device;
        // physicalDeviceProperties = physicalDevice.getProperties();

        queueFamilyProperties = vkbPhysicalDevice.get_queue_families();
    }

    void createLogicalDevice() {
        vkb::DeviceBuilder builder{vkbPhysicalDevice};

        auto dev_ret = builder.build();

        if (!dev_ret) {
            throw std::runtime_error("Failed to create logical device: " + std::string(dev_ret.error().message()));
        }

        vkbDevice = dev_ret.value();
        device = vkbDevice.device;

        VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

        auto gq_ret = vkbDevice.get_queue(vkb::QueueType::graphics);
        if (!gq_ret) {
            throw std::runtime_error("Failed to create graphics queue: " + std::string(gq_ret.error().message()));
        }
        graphicsQueue = gq_ret.value();

        auto pq_ret = vkbDevice.get_queue(vkb::QueueType::present);
        if (!pq_ret) {
            throw std::runtime_error("Failed to create graphics queue: " + std::string(gq_ret.error().message()));
        }
        presentQueue = pq_ret.value();
    }

    void createSwapChain() {
        vkb::SwapchainBuilder builder{vkbDevice};

        auto swapchain_ret =
                builder.set_desired_format(
                               vk::SurfaceFormatKHR(vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear))
                        .set_desired_present_mode(static_cast<VkPresentModeKHR>(vk::PresentModeKHR::eMailbox))
                        .set_image_usage_flags(static_cast<VkImageUsageFlags>(vk::ImageUsageFlagBits::eColorAttachment))
                        .build();

        if (!swapchain_ret) {
            throw std::runtime_error("Failed to create swapchain: " + std::string(swapchain_ret.error().message()));
        }

        vkbSwapchain = swapchain_ret.value();
        swapChain = vkbSwapchain.swapchain;

        auto vkb_images = vkbSwapchain.get_images().value();
        swapChainImages = std::vector<vk::Image>(vkb_images.begin(), vkb_images.end());

        auto vkb_image_views = vkbSwapchain.get_image_views().value();
        swapChainImageViews = std::vector<vk::ImageView>(vkb_image_views.begin(), vkb_image_views.end());

        swapChainImageFormat = static_cast<vk::Format>(vkbSwapchain.image_format);
        swapChainExtent = vkbSwapchain.extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] =
                    createImageView(swapChainImages[i], swapChainImageFormat, vk::ImageAspectFlagBits::eColor);
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment({}, swapChainImageFormat, vk::SampleCountFlagBits::e1,
                                                  vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                                                  vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
                                                  vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);

        vk::Format depthFormat = findDepthFormat();
        vk::AttachmentDescription depthAttachment(
                {}, depthFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
                vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
                vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentReference depthAttachmentRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &colorAttachmentRef,
                                       nullptr, &depthAttachmentRef);

        vk::SubpassDependency dependency(
                VK_SUBPASS_EXTERNAL, 0,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
                vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests, {},
                vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite);

        std::array<vk::AttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        renderPass = device.createRenderPass(vk::RenderPassCreateInfo({}, attachments, subpass, dependency));
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                                        vk::ShaderStageFlagBits::eVertex);

        vk::DescriptorSetLayoutBinding samplerLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1,
                                                            vk::ShaderStageFlagBits::eFragment);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        descriptorSetLayout = device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/scene.vert.spv");
        auto fragShaderCode = readFile("shaders/scene.frag.spv");

        vk::ShaderModuleCreateInfo vertShaderModuleInfo({}, vertShaderCode.size(),
                                                        reinterpret_cast<const uint32_t *>(vertShaderCode.data()));
        vk::ShaderModuleCreateInfo fragShaderModuleInfo({}, fragShaderCode.size(),
                                                        reinterpret_cast<const uint32_t *>(fragShaderCode.data()));

        vk::ShaderModule vertShaderModule = device.createShaderModule(vertShaderModuleInfo);
        vk::ShaderModule fragShaderModule = device.createShaderModule(fragShaderModuleInfo);

        vk::PipelineShaderStageCreateInfo vertShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                                                    vertShaderModule, "main");
        vk::PipelineShaderStageCreateInfo fragShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                                                    fragShaderModule, "main");

        std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {vertShaderStageCreateInfo,
                                                                   fragShaderStageCreateInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo(
                {}, 1, &bindingDescription, static_cast<uint32_t>(attributeDescriptions.size()),
                attributeDescriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, false);

        vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

        vk::PipelineRasterizationStateCreateInfo rasterizer(
                {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
                false, 0.0f, 0.0f, 0.0f, 1.0f);

        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, false, 1.0f, nullptr,
                                                             false, false);

        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, true, true, vk::CompareOp::eLess, false, false, {}, {},
                                                             0.0f, 1.0f);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment(
                false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne,
                vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                        vk::ColorComponentFlagBits::eA);

        vk::PipelineColorBlendStateCreateInfo colorBlending({}, false, vk::LogicOp::eCopy, 1, &colorBlendAttachment,
                                                            {0.0f, 0.0f, 0.0f, 0.0f});

        std::array<vk::DynamicState, 2> dynamicStates = {
                vk::DynamicState::eViewport,
                vk::DynamicState::eScissor,
        };

        vk::PipelineDynamicStateCreateInfo dynamicState({}, static_cast<uint32_t>(dynamicStates.size()),
                                                        dynamicStates.data());

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descriptorSetLayout, 0, nullptr);

        try {
            pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create pipeline layout: " + std::string(e.what()));
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo(
                {}, static_cast<uint32_t>(stages.size()), stages.data(), &vertexInputStateCreateInfo, &inputAssembly,
                nullptr, &viewportState, &rasterizer, &multisampling, &depthStencil, &colorBlending, &dynamicState,
                pipelineLayout, renderPass, 0, nullptr, -1);

        try {
            graphicsPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create graphics pipeline: " + std::string(e.what()));
        }

        device.destroyShaderModule(vertShaderModule);
        device.destroyShaderModule(fragShaderModule);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 2> attachments = {swapChainImageViews[i], depthImageView};

            vk::FramebufferCreateInfo frameBufferInfo(vk::FramebufferCreateFlags(), renderPass,
                                                      static_cast<uint32_t>(attachments.size()), attachments.data(),
                                                      swapChainExtent.width, swapChainExtent.height, 1);


            try {
                swapChainFramebuffers[i] = device.createFramebuffer(frameBufferInfo);
            } catch (vk::SystemError &e) {
                throw std::runtime_error("Failed to create framebuffer: " + std::string(e.what()));
            }
        }
    }

    void createCommandPool() {
        uint32_t graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

        vk::CommandPoolCreateInfo commandPoolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                  graphicsQueueFamily);

        try {
            commandPool = device.createCommandPool(commandPoolInfo);
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create command pool: " + std::string(e.what()));
        }
    }

    void createDepthResources() {
        vk::Format depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal,
                    depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    vk::Format findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features) {
        for (vk::Format format: candidates) {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);
            if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    vk::Format findDepthFormat() {
        std::vector<vk::Format> depthFormats = {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
                                                vk::Format::eD24UnormS8Uint};

        return findSupportedFormat(depthFormats, vk::ImageTiling::eOptimal,
                                   vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        auto stagingBufferCreateInfo = vk::BufferCreateInfo()
                                               .setSize(imageSize)
                                               .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                                               .setSharingMode(vk::SharingMode::eExclusive);
        vk::DeviceMemory stagingBufferMemory;

        vk::Buffer stagingBuffer;
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *data = device.mapMemory(stagingBufferMemory, 0, imageSize);
        std::memcpy(data, pixels, static_cast<size_t>(imageSize));
        device.unmapMemory(stagingBufferMemory);

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth),
                          static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        try {
            textureSampler = device.createSampler(samplerInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to create texture sampler: " + std::string(e.what()));
        }
    }

    vk::ImageView createImageView(VkImage image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
        vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {},
                                         vk::ImageSubresourceRange(aspectFlags, 0, 1, 0, 1));

        try {
            return device.createImageView(viewInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to create image view: " + std::string(e.what()));
        }
    }

    void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
                     vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image &image,
                     vk::DeviceMemory &imageMemory) {
        vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format, vk::Extent3D(width, height, 1), 1, 1,
                                      vk::SampleCountFlagBits::e1, tiling, usage, vk::SharingMode::eExclusive, 0,
                                      nullptr, vk::ImageLayout::eUndefined);

        try {
            image = device.createImage(imageInfo);
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to create image: " + std::string(e.what()));
        }

        vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

        vk::MemoryAllocateInfo allocInfo(memRequirements.size,
                                         findMemoryType(memRequirements.memoryTypeBits, properties));

        try {
            imageMemory = device.allocateMemory(allocInfo);
        } catch (vk::SystemError &e) {
            throw std::runtime_error("Failed to allocate memory: " + std::string(e.what()));
        }

        device.bindImageMemory(image, imageMemory, 0);
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
        vk::ImageMemoryBarrier barrier =
                vk::ImageMemoryBarrier{}
                        .setOldLayout(oldLayout)
                        .setNewLayout(newLayout)
                        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .setImage(image)
                        .setSubresourceRange(vk::ImageSubresourceRange{}
                                                     .setAspectMask(vk::ImageAspectFlagBits::eColor)
                                                     .setBaseMipLevel(0)
                                                     .setLevelCount(1)
                                                     .setBaseArrayLayer(0)
                                                     .setLayerCount(1));

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eNone;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, vk::DependencyFlags{}, // No dependency flags
                                      {}, // No memory barriers
                                      {}, // No buffer memory barriers
                                      {barrier} // Image memory barriers
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = vk::Offset3D{0, 0, 0};
        region.imageExtent = vk::Extent3D{width, height, 1};

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    void loadModel() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto &shape: shapes) {
            for (const auto &index: shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1],
                              attrib.vertices[3 * index.vertex_index + 2]};

                vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                                   1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};

                vertex.color = {1.0f, 1.0f, 1.0f};

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void createVertexBuffer() {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        device.unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
        memcpy(data, indices.data(), (size_t) bufferSize);
        device.unmapMemory(stagingBufferMemory);

        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.freeMemory(stagingBufferMemory);
    }

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         uniformBuffers[i], uniformBuffersMemory[i]);

            uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
        }
    }

    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        try {
            descriptorPool = device.createDescriptorPool(poolInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to create descriptor pool: " + std::string(e.what()));
        }
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        try {
            descriptorSets = device.allocateDescriptorSets(allocInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to allocate descriptor sets: " + std::string(e.what()));
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0,
                                        nullptr);
        }
    }

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                      vk::Buffer &buffer, vk::DeviceMemory &bufferMemory) {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;

        try {
            buffer = device.createBuffer(bufferInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to create buffer: " + std::string(e.what()));
        }

        vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        try {
            bufferMemory = device.allocateMemory(allocInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to allocate buffer memory: " + std::string(e.what()));
        }

        device.bindBufferMemory(buffer, bufferMemory, 0);
    }

    vk::CommandBuffer beginSingleTimeCommands() {
        auto allocInfo = vk::CommandBufferAllocateInfo{}
                                 .setLevel(vk::CommandBufferLevel::ePrimary)
                                 .setCommandPool(commandPool)
                                 .setCommandBufferCount(1);

        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

        auto beginInfo = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
        commandBuffer.end();

        auto submitInfo = vk::SubmitInfo{}.setCommandBufferCount(1).setPCommandBuffers(&commandBuffer);

        vk::Result submitResult = graphicsQueue.submit(1, &submitInfo, VK_NULL_HANDLE);
        if (submitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to submit command buffer command buffer submission");
        }
        graphicsQueue.waitIdle();

        device.freeCommandBuffers(commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        try {
            commandBuffers = device.allocateCommandBuffers(allocInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to allocate command buffers: " + std::string(e.what()));
        }
    }

    void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex) {
        vk::CommandBufferBeginInfo beginInfo{};

        try {
            commandBuffer.begin(beginInfo);
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to begin command buffer: " + std::string(e.what()));
        }

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = vk::Offset2D{0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<vk::ClearValue, 2> clearValues{};
        clearValues[0].color = vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        ;

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        commandBuffer.setViewport(0, 1, &viewport);

        vk::Rect2D scissor{};
        scissor.offset = vk::Offset2D{0, 0};
        scissor.extent = swapChainExtent;
        commandBuffer.setScissor(0, 1, &scissor);

        vk::Buffer vertexBuffers[] = {vertexBuffer};
        vk::DeviceSize offsets[] = {0};

        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

        commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                         {descriptorSets[currentFrame]}, nullptr);

        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer.endRenderPass();

        try {
            commandBuffer.end();
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to render command buffers: " + std::string(e.what()));
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semaphoreInfo{};

        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            try {
                imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
                renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                inFlightFences[i] = device.createFence(fenceInfo);
            } catch (const vk::SystemError &e) {
                throw std::runtime_error("Failed to create synchronization objects for a frame: " +
                                         std::string(e.what()));
            }
        }
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f,
                                    10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        vk::Result waitResult = device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        if (waitResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for fences!");
        }

        uint32_t imageIndex;
        try {
            auto result = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
                                                     VK_NULL_HANDLE);
            imageIndex = result.value;
        } catch (const vk::OutOfDateKHRError &) {
            recreateSwapChain();
            return;
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to acquire swap chain image: " + std::string(e.what()));
        }

        updateUniformBuffer(currentFrame);

        vk::Result resetResult = device.resetFences(1, &inFlightFences[currentFrame]);
        if (resetResult != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to reset fence!");
        }

        commandBuffers[currentFrame].reset(vk::CommandBufferResetFlags());

        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        vk::Semaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};

        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        vk::Semaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        try {
            vk::Result submitResult = graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]);
            if (submitResult != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to submit command buffer to present!");
            }
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to submit command buffers for a frame: " + std::string(e.what()));
        }

        vk::SwapchainKHR swapChains[] = {swapChain};
        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        try {
            auto presentResult = presentQueue.presentKHR(presentInfo);

            if (presentResult == vk::Result::eSuboptimalKHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapChain();
            }
        } catch (const vk::OutOfDateKHRError &e) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        } catch (const vk::SystemError &e) {
            throw std::runtime_error("Failed to present: " + std::string(e.what()));
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    static std::vector<char> readFile(const std::string &filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                        void *pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

int main() {
    WaterDemoApplication app;
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
