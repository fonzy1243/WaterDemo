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

void DescriptorWriter::write_image(int binding, vk::ImageView image, vk::Sampler sampler, vk::ImageLayout layout,
                                   vk::DescriptorType type) {
    vk::DescriptorImageInfo &info = imageInfos.emplace_back(
            vk::DescriptorImageInfo().setSampler(sampler).setImageView(image).setImageLayout(layout));

    vk::WriteDescriptorSet write = vk::WriteDescriptorSet()
                                           .setDstBinding(binding)
                                           .setDstSet(VK_NULL_HANDLE)
                                           .setDescriptorCount(1)
                                           .setDescriptorType(type)
                                           .setPImageInfo(&info);

    writes.push_back(write);
}

void DescriptorWriter::write_buffer(int binding, vk::Buffer buffer, size_t size, size_t offset,
                                    vk::DescriptorType type) {
    vk::DescriptorBufferInfo &info =
            bufferInfos.emplace_back(vk::DescriptorBufferInfo().setBuffer(buffer).setOffset(offset).setRange(size));

    vk::WriteDescriptorSet write = vk::WriteDescriptorSet()
                                           .setDstBinding(binding)
                                           .setDstSet(VK_NULL_HANDLE)
                                           .setDescriptorCount(1)
                                           .setDescriptorType(type)
                                           .setPBufferInfo(&info);

    writes.push_back(write);
}

void DescriptorWriter::clear() {
    imageInfos.clear();
    writes.clear();
    bufferInfos.clear();
}

void DescriptorWriter::update_set(vk::Device device, vk::DescriptorSet set) {
    for (vk::WriteDescriptorSet &write: writes) {
        write.setDstSet(set);
    }

    device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void DescriptorAllocatorGrowable::init(vk::Device device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios) {
    ratios.clear();

    for (auto r: poolRatios) {
        ratios.push_back(r);
    }

    vk::DescriptorPool newPool = create_pool(device, maxSets, poolRatios);

    setsPerPool = maxSets * 1.5;

    readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clear_pools(vk::Device device) {
    for (auto p: readyPools) {
        device.resetDescriptorPool(p);
    }
    for (auto p: fullPools) {
        device.resetDescriptorPool(p);
        readyPools.push_back(p);
    }
    fullPools.clear();
}

void DescriptorAllocatorGrowable::destroy_pools(vk::Device device) {
    for (auto p: readyPools) {
        device.destroyDescriptorPool(p);
    }
    readyPools.clear();
    for (auto p: fullPools) {
        device.destroyDescriptorPool(p);
    }
    fullPools.clear();
}

vk::DescriptorPool DescriptorAllocatorGrowable::get_pool(vk::Device device) {
    vk::DescriptorPool newPool;
    if (readyPools.size() != 0) {
        newPool = readyPools.back();
        readyPools.pop_back();
    } else {
        newPool = create_pool(device, setsPerPool, ratios);

        setsPerPool = setsPerPool * 1.5;
        if (setsPerPool > 4092) {
            setsPerPool = 4092;
        }
    }

    return newPool;
}

vk::DescriptorPool DescriptorAllocatorGrowable::create_pool(vk::Device device, uint32_t setCount,
                                                            std::span<PoolSizeRatio> poolRatios) {
    std::vector<vk::DescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio: poolRatios) {
        poolSizes.push_back(vk::DescriptorPoolSize()
                                    .setType(ratio.type)
                                    .setDescriptorCount(static_cast<uint32_t>(ratio.ratio * setCount)));
    }

    vk::DescriptorPoolCreateInfo pool_info = vk::DescriptorPoolCreateInfo()
                                                     .setFlags(vk::DescriptorPoolCreateFlags())
                                                     .setMaxSets(setCount)
                                                     .setPoolSizeCount(static_cast<uint32_t>(poolSizes.size()))
                                                     .setPPoolSizes(poolSizes.data());

    vk::DescriptorPool newPool = device.createDescriptorPool(pool_info, nullptr);

    return newPool;
}

vk::DescriptorSet DescriptorAllocatorGrowable::allocate(vk::Device device, vk::DescriptorSetLayout layout,
                                                        void *pNext) {
    vk::DescriptorPool poolToUse = get_pool(device);

    vk::DescriptorSetAllocateInfo allocInfo = vk::DescriptorSetAllocateInfo()
                                                      .setPNext(pNext)
                                                      .setDescriptorPool(poolToUse)
                                                      .setDescriptorSetCount(1)
                                                      .setPSetLayouts(&layout);

    vk::DescriptorSet ds;
    vk::Result result = device.allocateDescriptorSets(&allocInfo, &ds);

    if (result == vk::Result::eErrorOutOfPoolMemory || result == vk::Result::eErrorFragmentedPool) {
        fullPools.push_back(poolToUse);

        poolToUse = get_pool(device);
        allocInfo.setDescriptorPool(poolToUse);

        vk::Result result2 = device.allocateDescriptorSets(&allocInfo, &ds);
        if (result2 != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to allocate descriptor set");
        }
    }

    readyPools.push_back(poolToUse);
    return ds;
}
