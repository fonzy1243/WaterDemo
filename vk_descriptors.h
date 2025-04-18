#pragma once

#include <deque>
#include <span>
#include <vector>
#include "vk_types.h"

#ifndef VK_DESCRIPTORS_H
#define VK_DESCRIPTORS_H

struct DescriptorLayoutBuilder {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;

    void add_binding(uint32_t binding, vk::DescriptorType type);
    void clear();
    vk::DescriptorSetLayout build(vk::Device device, vk::ShaderStageFlags shaderStages, void *pNext = nullptr,
                                  vk::DescriptorSetLayoutCreateFlags flags = vk::DescriptorSetLayoutCreateFlags());
};

struct DescriptorAllocator {
    struct PoolSizeRatio {
        vk::DescriptorType type;
        float ratio;
    };

    vk::DescriptorPool pool;

    void init_pool(vk::Device device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
    void clear_descriptors(vk::Device device);
    void destroy_pool(vk::Device device);

    vk::DescriptorSet allocate(vk::Device device, vk::DescriptorSetLayout layout);
};

#endif // VK_DESCRIPTORS_H
