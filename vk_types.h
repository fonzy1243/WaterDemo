#pragma once

#include <vk_include.h>
#include <vma/vk_mem_alloc.h>
#include <vma/vk_mem_alloc.hpp>

#include <fmt/core.h>

#ifndef VK_TYPES_H
#define VK_TYPES_H

struct AllocatedImage {
    vk::Image image;
    vk::ImageView imageView;
    vma::Allocation allocation;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;
};

struct AllocatedBuffer {
    vk::Buffer buffer;
    vma::Allocation allocation;
    vma::AllocationInfo info;
};

struct GPUGLTFMaterial {
    glm::vec4 colorFactors;
    glm::vec4 metal_rough_factors;
    glm::vec4 extra[14];
};

static_assert(sizeof(GPUGLTFMaterial) == 256);

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
};

enum class MaterialPass : uint8_t { MainColor, Transparent, Other };

struct MaterialPipeline {
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
};

struct MaterialInstance {
    MaterialPipeline *pipeline;
    vk::DescriptorSet materialSet;
    MaterialPass passType;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    vk::DeviceAddress vertexBufferAddress;
};

struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    vk::DeviceAddress vertexBuffer;
};

struct OceanFragConstats {
    alignas(16) glm::vec4 ambientColor;
    alignas(16) glm::vec4 sunlightDirection;
    alignas(16) glm::vec4 sunlightColor;
};

struct OceanDrawPushConstants {
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;
    glm::vec4 sunlightColor;
    glm::mat4 worldMatrix;
    vk::DeviceAddress vertexBuffer;
};

struct DrawContext;

class IRenderable {
    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) = 0;
};

struct Node : public IRenderable {
    std::weak_ptr<Node> parent;
    std::vector<std::shared_ptr<Node>> children;

    glm::mat4 localTransform;
    glm::mat4 worldTransform;

    void refreshTransform(const glm::mat4 &parentMatrix) {
        worldTransform = parentMatrix * localTransform;
        for (auto c: children) {
            c->refreshTransform(worldTransform);
        }
    }

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) {
        // draw children
        for (auto &c: children) {
            c->Draw(topMatrix, ctx);
        }
    }
};

#endif // VK_TYPES_H
