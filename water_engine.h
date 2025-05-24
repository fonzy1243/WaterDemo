#pragma once

#include <deque>
#include <functional>
#include <vector>

#include "VkBootstrap.h"
#include "camera.h"
#include "simulation/FFTOcean.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "vk_types.h"

#ifndef WATER_ENGINE_H
#define WATER_ENGINE_H

struct ComputeOceanEffect {
    const char *name;

    vk::Pipeline pipeline;
    vk::PipelineLayout layout;

    OceanParameters parameters;
};

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()> &&function) { deletors.push_back(function); }

    void flush() {
        // Reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
            (*it)();
        }

        deletors.clear();
    }
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    vk::DescriptorSetLayout materialLayout;

    struct MaterialConstants {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // padding
        glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage;
        vk::Sampler colorSampler;
        AllocatedImage metalRoughImage;
        vk::Sampler metalRoughSampler;
        vk::Buffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(WaterEngine *engine);
    void clear_resources(vk::Device device);

    MaterialInstance write_material(vk::Device device, MaterialPass pass, const MaterialResources &resources,
                                    DescriptorAllocatorGrowable &descriptorAllocator);
};

struct FrameData {
    vk::Semaphore _swapchainSemaphore, _renderSemaphore;
    vk::Fence _renderFence;

    vk::CommandPool _commandPool;
    vk::CommandBuffer _mainCommandBuffer;

    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
    DescriptorAllocatorGrowable _oceanDescriptors;
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    vk::Buffer indexBuffer;

    MaterialInstance *material;

    glm::mat4 transform;
    vk::DeviceAddress vertexBufferAddress;
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct MeshNode : public Node {
    std::shared_ptr<MeshAsset> mesh;

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class WaterEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    bool resize_requested{false};
    int GRID_SIZE{4096};
    float GRID_SCALE{150};
    vk::Extent2D _windowExtent{1600, 900};

    struct SDL_Window *_window{nullptr};

    vk::Instance _instance;
    vk::DebugUtilsMessengerEXT _debug_messenger;
    vk::PhysicalDevice _chosenGPU;
    vk::Device _device;
    vk::SurfaceKHR _surface;

    vk::Semaphore _presentSemaphore, _renderSemaphore;
    vk::Fence _renderFence;

    FrameData _frames[FRAME_OVERLAP];

    FrameData &get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

    vk::Queue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    vk::CommandPool _commandPool;
    vk::CommandBuffer _commandBuffer;

    vk::RenderPass _renderPass;

    vk::SwapchainKHR _swapchain;
    vk::Format _swapchainImageFormat;

    std::vector<vk::Image> _swapchainImages;
    std::vector<vk::ImageView> _swapchainImageViews;
    vk::Extent2D _swapchainExtent;

    std::vector<vk::Framebuffer> _framebuffers;

    DeletionQueue _mainDeletionQueue;

    vma::Allocator _allocator = nullptr;

    DescriptorAllocatorGrowable globalDescriptorAllocator;

    AllocatedImage _drawImage;
    vk::Extent2D _drawExtent;
    AllocatedImage _depthImage;
    float renderScale = 1.f;

    // Spectrum shader
    AllocatedImage _spectrumImage;
    AllocatedImage _spectrumNormalImage;
    vk::Extent2D _spectrumExtent;
    vk::DescriptorSet _spectrumImageDescriptors;
    vk::DescriptorSetLayout _spectrumImageDescriptorLayout;
    // Spectrum shader pipeline
    vk::Pipeline _spectrumPipeline;
    vk::PipelineLayout _spectrumPipelineLayout;

    // Butterfly shader
    AllocatedImage _butterflyImage;
    vk::Extent2D _butterflyExtent;
    vk::DescriptorSet _butterflyImageDescriptors;
    vk::DescriptorSetLayout _butterflyImageDescriptorLayout;
    vk::Pipeline _butterflyPipeline;
    vk::PipelineLayout _butterflyPipelineLayout;
    ButterflyParameters _butterflyParameters;
    bool showHeightmap{true};

    // FFT shader
    AllocatedImage _pingPong0Image;
    AllocatedImage _pingPong1Image;
    AllocatedImage _heightmapImage;
    AllocatedImage _normalMapImage;
    vk::Extent2D _fftExtent;
    vk::DescriptorSet _fftImageDescriptors;
    vk::DescriptorSetLayout _fftImageDescriptorLayout;
    vk::Pipeline _fftPipeline;
    vk::PipelineLayout _fftPipelineLayout;
    IFFTParameters _fftParameters;

    // Copy shader
    vk::Pipeline _copyPipeline;
    vk::PipelineLayout _copyPipelineLayout;

    // Permute shader
    vk::Pipeline _permutePipeline;
    vk::PipelineLayout _permutePipelineLayout;

    vk::Fence _immFence;
    vk::CommandBuffer _immCommandBuffer;
    vk::CommandPool _immCommandPool;

    // Constants for ocean simulation
    std::vector<ComputeOceanEffect> oceanEffects;
    int currentOceanEffect{0};

    // Ocean vertex and fragment shaders
    vk::DescriptorSetLayout _oceanLayout;
    vk::PipelineLayout _oceanPipelineLayout;
    vk::Pipeline _oceanPipeline;
    vk::DescriptorSet _oceanDescriptors;

    // Mes pipeline and layout
    vk::PipelineLayout _meshPipelineLayout;
    vk::Pipeline _meshPipeline;

    GPUMeshBuffers rectangle;
    GPUSceneData sceneData;

    vk::DescriptorSetLayout _gpuSceneDataDescriptorLayout;

    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    // Test images
    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    vk::Sampler _defaultSamplerLinear;
    vk::Sampler _defaultSamplerNearest;

    vk::DescriptorSetLayout _singleImageDescriptorLayout;

    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadScenes;

    // Camera
    Camera mainCamera;

    // Initialize engine
    void init();

    // Shutdown engine
    void cleanup();

    // Draw loop
    void draw();

    // Run main loop
    void run();

    void immediate_submit(std::function<void(vk::CommandBuffer)> &&function);

    AllocatedBuffer create_buffer(size_t allocSize, vk::BufferUsageFlags usage, vma::MemoryUsage memoryUsage);
    AllocatedImage create_image(vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage,
                                bool mipmapped = false);
    AllocatedImage create_image(void *data, vk::Extent3D size, vk::Format format, vk::ImageUsageFlags usage,
                                bool mipmapped = false);

    void destroy_buffer(const AllocatedBuffer &buffer);
    void destroy_image(const AllocatedImage &image);

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);


private:
    void init_imgui();

    void init_pipelines();
    void init_background_pipelines();
    void init_ocean_pipeline();

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    void init_vulkan();

    void init_swapchain();
    void resize_swapchain();

    void init_commands();

    void init_sync_structures();
    void init_compute_images();

    void init_default_data();
    void init_compute_descriptors();

    void init_descriptors();
    void run_spectrum(vk::CommandBuffer cmd);

    void draw_background(vk::CommandBuffer cmd);
    void draw_geometry(vk::CommandBuffer cmd);

    void draw_imgui(vk::CommandBuffer cmd, vk::ImageView targetImageView);

    void run_butterfly(vk::CommandBuffer cmd);

    void run_ifft(vk::CommandBuffer cmd, vk::Image &input, vk::Image *output = nullptr);

    void ifft_pass(vk::CommandBuffer cmd, uint32_t stage, uint32_t direction, uint32_t resolution, uint32_t pingPong);

    void run_copy(vk::CommandBuffer cmd);

    void run_permute(vk::CommandBuffer cmd);

    void update_scene();
};

#endif // WATER_ENGINE_H
