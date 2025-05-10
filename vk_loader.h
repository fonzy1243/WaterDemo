//
// Created by Alfon on 4/30/2025.
//
#pragma once

#ifndef VK_LOADER_H
#define VK_LOADER_H

#include <filesystem>
#include <unordered_map>

#include "vk_descriptors.h"
#include "vk_types.h"
#include "water_engine.h"

class WaterEngine;

struct Bounds {
    glm::vec3 origin;
    float sphereRadius;
    glm::vec3 extents;
};

struct GLTFMaterial {
    MaterialInstance data;
};

struct GeoSurface {
    uint32_t startIndex;
    uint32_t count;
    Bounds bounds;
    std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
    std::string name;

    std::vector<GeoSurface> surfaces;
    GPUMeshBuffers meshBuffers;
};

struct LoadedGLTF : public IRenderable {
    std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
    std::unordered_map<std::string, std::shared_ptr<AllocatedImage>> images;
    std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

    std::vector<std::shared_ptr<Node>> topNodes;

    std::vector<vk::Sampler> samplers;

    DescriptorAllocatorGrowable descriptorPool;

    AllocatedBuffer materialDataBuffer;

    WaterEngine *creator;

    ~LoadedGLTF() { clearAll(); };

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx);

private:
    void clearAll();
};


std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(WaterEngine *engine,
                                                                      std::filesystem::path filePath);

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(WaterEngine *engine, std::string_view filePath);

#endif // VK_LOADER_H
