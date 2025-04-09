#ifndef VULKANUTILS_H
#define VULKANUTILS_H

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#include <tiny_obj_loader.h>

namespace VulkanUtils {
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

    // IMAGE AND VIEW UTILS

    vk::ImageView createImageView(vk::Device device, vk::Image image, vk::Format format,
                                  vk::ImageAspectFlags aspectFlags);
    void createImage(vk::Device device, vk::PhysicalDevice physicalDevice, uint32_t width, uint32_t height,
                     vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::MemoryPropertyFlags properties, vk::Image &image, vk::DeviceMemory &imageMemory);
    vk::Format findSupportedFormat(vk::PhysicalDevice physicalDevice, const std::vector<vk::Format> &candidates,
                                   vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format findDepthFormat(vk::PhysicalDevice physicalDevice);
    void transitionImageLayout(vk::Device device, vk::CommandPool commandPool, vk::Queue queue, vk::Image image,
                               vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
    void copyBufferToImage(vk::Device device, vk::CommandPool commandPool, vk::Queue queue, vk::Buffer buffer,
                           vk::Image image, uint32_t width, uint32_t height);

    // BUFFER UTILS

    void createBuffer(vk::Device device, vk::PhysicalDevice physicalDevice, vk::DeviceSize size,
                      vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                      vk::DeviceMemory &bufferMemory);
    void copyBuffer(vk::Device device, vk::CommandPool commandPool, vk::Queue queue, vk::Buffer srcBuffer,
                    vk::Buffer dstBuffer, vk::DeviceSize size);
    uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    // COMMAND BUFFER UTILS

    vk::CommandBuffer beginSingleTimeCommands(vk::Device device, vk::CommandPool commandPool);
    void endSingleTimeCommands(vk::Device device, vk::CommandPool commandPool, vk::Queue queue,
                               vk::CommandBuffer commandBuffer);

    // SHADER UTILS

    std::vector<char> readFile(const std::string &filename);

    // MODEL LOADING

    void loadModel(const std::string &modelPath, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices);
}; // namespace VulkanUtils

namespace std {
    template<>
    struct hash<VulkanUtils::Vertex> {
        size_t operator()(VulkanUtils::Vertex const &vertex) const;
    };
} // namespace std

#endif // VULKANUTILS_H
