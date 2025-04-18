#pragma once

#include <deque>
#include <functional>
#include <vector>

#include "VkBootstrap.h"
#include "simulation/FFTOcean.h"
#include "vk_descriptors.h"
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

struct FrameData {
    vk::CommandPool _commandPool;
    vk::CommandBuffer _mainCommandBuffer;
    vk::Semaphore _swapchainSemaphore, _renderSemaphore;
    vk::Fence _renderFence;
    DeletionQueue _deletionQueue;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class WaterEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    vk::Extent2D _windowExtent{1700, 900};

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

    AllocatedImage _drawImage;
    vk::Extent2D _drawExtent;

    DescriptorAllocator globalDescriptorAllocator;

    vk::DescriptorSet _drawImageDescriptors;
    vk::DescriptorSetLayout _drawImageDescriptorLayout;

    // Compute shader pipeline
    vk::Pipeline _spectrumPipeline;
    vk::PipelineLayout _spectrumPipelineLayout;

    vk::Fence _immFence;
    vk::CommandBuffer _immCommandBuffer;
    vk::CommandPool _immCommandPool;

    std::vector<ComputeOceanEffect> oceanEffects;
    int currentOceanEffect{0};

    // Initialize engine
    void init();

    // Shutdown engine
    void cleanup();

    // Draw loop
    void draw();

    // Run main loop
    void run();

    void immediate_submit(std::function<void(vk::CommandBuffer)> &&function);

private:
    void init_imgui();

    void init_pipelines();
    void init_background_pipelines();

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    void init_vulkan();

    void init_swapchain();

    void init_commands();

    void init_sync_structures();

    void init_descriptors();

    void draw_background(vk::CommandBuffer cmd);

    void draw_imgui(vk::CommandBuffer cmd, vk::ImageView targetImageView);
};

#endif // WATER_ENGINE_H
