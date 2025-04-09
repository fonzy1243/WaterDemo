//
// Created by Alfon on 4/8/2025.
//

#include "VulkanUtils.h"

namespace std {
    size_t hash<VulkanUtils::Vertex>::operator()(VulkanUtils::Vertex const &vertex) const {
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
               (hash<glm::vec2>()(vertex.texCoord) << 1);
    }
} // namespace std

// IMAGE AND VIEW UTILS

vk::ImageView VulkanUtils::createImageView(vk::Device device, vk::Image image, vk::Format format,
                                           vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {},
                                     vk::ImageSubresourceRange(aspectFlags, 0, 1, 0, 1));

    try {
        return device.createImageView(viewInfo);
    } catch (const vk::SystemError &e) {
        throw std::runtime_error("Failed to create image view: " + std::string(e.what()));
    }
}

void VulkanUtils::createImage(vk::Device device, vk::PhysicalDevice physicalDevice, uint32_t width, uint32_t height,
                              vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                              vk::MemoryPropertyFlags properties, vk::Image &image, vk::DeviceMemory &imageMemory) {
    vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format, vk::Extent3D(width, height, 1), 1, 1,
                                  vk::SampleCountFlagBits::e1, tiling, usage, vk::SharingMode::eExclusive, 0, nullptr,
                                  vk::ImageLayout::eUndefined);

    try {
        image = device.createImage(imageInfo);
    } catch (vk::SystemError &e) {
        throw std::runtime_error("Failed to create image: " + std::string(e.what()));
    }

    vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo allocInfo(
            memRequirements.size,
            VulkanUtils::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties));

    try {
        imageMemory = device.allocateMemory(allocInfo);
    } catch (vk::SystemError &e) {
        throw std::runtime_error("Failed to allocate memory: " + std::string(e.what()));
    }

    device.bindImageMemory(image, imageMemory, 0);
}

vk::Format VulkanUtils::findSupportedFormat(vk::PhysicalDevice physicalDevice,
                                            const std::vector<vk::Format> &candidates, vk::ImageTiling tiling,
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

vk::Format VulkanUtils::findDepthFormat(vk::PhysicalDevice physicalDevice) {
    std::vector<vk::Format> depthFormats = {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
                                            vk::Format::eD24UnormS8Uint};

    return findSupportedFormat(physicalDevice, depthFormats, vk::ImageTiling::eOptimal,
                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

void VulkanUtils::transitionImageLayout(vk::Device device, vk::CommandPool commandPool, vk::Queue queue,
                                        vk::Image image, vk::Format format, vk::ImageLayout oldLayout,
                                        vk::ImageLayout newLayout) {

    vk::CommandBuffer commandBuffer = VulkanUtils::beginSingleTimeCommands(device, commandPool);
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

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

void VulkanUtils::copyBufferToImage(vk::Device device, vk::CommandPool commandPool, vk::Queue queue, vk::Buffer buffer,
                                    vk::Image image, uint32_t width, uint32_t height) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

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

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

// BUFFER UTILS

void VulkanUtils::createBuffer(vk::Device device, vk::PhysicalDevice physicalDevice, vk::DeviceSize size,
                               vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                               vk::DeviceMemory &bufferMemory) {

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
    allocInfo.memoryTypeIndex = VulkanUtils::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    try {
        bufferMemory = device.allocateMemory(allocInfo);
    } catch (const vk::SystemError &e) {
        throw std::runtime_error("Failed to allocate buffer memory: " + std::string(e.what()));
    }

    device.bindBufferMemory(buffer, bufferMemory, 0);
}

void VulkanUtils::copyBuffer(vk::Device device, vk::CommandPool commandPool, vk::Queue queue, vk::Buffer srcBuffer,
                             vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands(device, commandPool);

    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    endSingleTimeCommands(device, commandPool, queue, commandBuffer);
}

uint32_t VulkanUtils::findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter,
                                     vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}


// COMMAND BUFFER UTILS

vk::CommandBuffer VulkanUtils::beginSingleTimeCommands(vk::Device device, vk::CommandPool commandPool) {
    auto allocInfo = vk::CommandBufferAllocateInfo{}
                             .setLevel(vk::CommandBufferLevel::ePrimary)
                             .setCommandPool(commandPool)
                             .setCommandBufferCount(1);

    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

    auto beginInfo = vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

void VulkanUtils::endSingleTimeCommands(vk::Device device, vk::CommandPool commandPool, vk::Queue queue,
                                        vk::CommandBuffer commandBuffer) {
    commandBuffer.end();

    auto submitInfo = vk::SubmitInfo{}.setCommandBufferCount(1).setPCommandBuffers(&commandBuffer);

    vk::Result submitResult = queue.submit(1, &submitInfo, VK_NULL_HANDLE);
    if (submitResult != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to submit command buffer command buffer submission");
    }
    queue.waitIdle();

    device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

// SHADER UTILS

std::vector<char> VulkanUtils::readFile(const std::string &filename) {
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

// MODEL LOADING

void VulkanUtils::loadModel(const std::string &modelPath, std::vector<Vertex> &vertices,
                            std::vector<uint32_t> &indices) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
        throw std::runtime_error(warn + err);
    }

    std::unordered_map<VulkanUtils::Vertex, uint32_t> uniqueVertices{};

    for (const auto &shape: shapes) {
        for (const auto &index: shape.mesh.indices) {
            VulkanUtils::Vertex vertex{};

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
