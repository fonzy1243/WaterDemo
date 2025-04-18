//
// Created by Alfon on 4/16/2025.
//

#include "vk_descriptors.h"

void DescriptorLayoutBuilder::add_binding(uint32_t binding, vk::DescriptorType type) {
    vk::DescriptorSetLayoutBinding newbind =
            vk::DescriptorSetLayoutBinding().setBinding(binding).setDescriptorCount(1).setDescriptorType(type);

    bindings.push_back(newbind);
}

void DescriptorLayoutBuilder::clear() { bindings.clear(); }

vk::DescriptorSetLayout DescriptorLayoutBuilder::build(vk::Device device, vk::ShaderStageFlags shaderStages,
                                                       void *pNext, vk::DescriptorSetLayoutCreateFlags flags) {
    for (auto &b: bindings) {
        b.stageFlags |= shaderStages;
    }

    vk::DescriptorSetLayoutCreateInfo info = vk::DescriptorSetLayoutCreateInfo()
                                                     .setPNext(pNext)
                                                     .setPBindings(bindings.data())
                                                     .setBindingCount(static_cast<uint32_t>(bindings.size()))
                                                     .setFlags(flags);

    vk::DescriptorSetLayout set = device.createDescriptorSetLayout(info, nullptr);

    return set;
}

void DescriptorAllocator::init_pool(vk::Device device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio: poolRatios) {
        poolSizes.push_back(vk::DescriptorPoolSize()
                                    .setType(ratio.type)
                                    .setDescriptorCount(static_cast<uint32_t>(ratio.ratio * maxSets)));
    }

    vk::DescriptorPoolCreateInfo pool_info = vk::DescriptorPoolCreateInfo()
                                                     .setFlags(vk::DescriptorPoolCreateFlags())
                                                     .setMaxSets(maxSets)
                                                     .setPoolSizeCount((uint32_t) poolSizes.size())
                                                     .setPPoolSizes(poolSizes.data());

    pool = device.createDescriptorPool(pool_info, nullptr);
}

void DescriptorAllocator::clear_descriptors(vk::Device device) { device.resetDescriptorPool(pool); }

void DescriptorAllocator::destroy_pool(vk::Device device) { device.destroyDescriptorPool(pool); }

vk::DescriptorSet DescriptorAllocator::allocate(vk::Device device, vk::DescriptorSetLayout layout) {
    vk::DescriptorSetAllocateInfo allocInfo = vk::DescriptorSetAllocateInfo()
                                                      .setPNext(nullptr)
                                                      .setDescriptorPool(pool)
                                                      .setDescriptorSetCount(1)
                                                      .setPSetLayouts(&layout);

    vk::DescriptorSet ds = {};
    vk::Result allocResult = device.allocateDescriptorSets(&allocInfo, &ds);

    if (allocResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    return ds;
}
