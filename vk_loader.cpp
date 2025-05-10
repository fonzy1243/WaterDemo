//
// Created by Alfon on 4/30/2025.
//

#include <iostream>
#include "water_engine.h"

#include "vk_loader.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include "vk_initializers.h"
#include "vk_types.h"

#define STB_IMAGE_IMPLEMENTATION
#include <memory>
#include <stb/stb_image.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

vk::Filter extract_filter(fastgltf::Filter filter) {
    switch (filter) {
        // nearest samplers
        case fastgltf::Filter::Nearest:
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::LinearMipMapNearest:
            return vk::Filter::eNearest;

            // linear samplers
        case fastgltf::Filter::Linear:
        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return vk::Filter::eLinear;
    }
}
vk::SamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter) {
    switch (filter) {
        // nearest samplers
        case fastgltf::Filter::NearestMipMapNearest:
        case fastgltf::Filter::LinearMipMapNearest:
            return vk::SamplerMipmapMode::eNearest;

            // linear samplers
        case fastgltf::Filter::NearestMipMapLinear:
        case fastgltf::Filter::LinearMipMapLinear:
        default:
            return vk::SamplerMipmapMode::eLinear;
    }
}

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(WaterEngine *engine,
                                                                      std::filesystem::path filePath) {
    std::cout << "Loading GLTF: " << filePath << std::endl;

    auto dataRes = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!dataRes) {
        std::cout << "Failed to load GLTF: " << filePath << std::endl;
        return {};
    }

    fastgltf::GltfDataBuffer &data = dataRes.get();

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser{};

    auto load = parser.loadGltfBinary(data, filePath.parent_path(), gltfOptions);
    if (load) {
        gltf = std::move(load.get());
    } else {
        fmt::print("Failed to load glTF: {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh: gltf.meshes) {
        MeshAsset newmesh;

        newmesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto &&p: mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = static_cast<uint32_t>(indices.size());
            newSurface.count = static_cast<uint32_t>(gltf.accessors[p.indicesAccessor.value()].count);

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor &indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(
                        gltf, indexaccessor, [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
            }

            // load vertex positions
            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newvtx;
                    newvtx.position = v;
                    newvtx.normal = {1, 0, 0};
                    newvtx.color = glm::vec4{1.f};
                    newvtx.uv_x = 0;
                    newvtx.uv_y = 0;
                    vertices[initial_vtx + index] = newvtx;
                });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                        gltf, gltf.accessors[(*normals).accessorIndex],
                        [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).accessorIndex],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                        gltf, gltf.accessors[(*colors).accessorIndex],
                        [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
            }
            newmesh.surfaces.push_back(newSurface);
        }

        constexpr bool OverrideColors = true;
        if (OverrideColors) {
            for (Vertex &vtx: vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }
        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
    }

    return meshes;
}

std::optional<AllocatedImage> load_image(WaterEngine *engine, fastgltf::Asset &asset, fastgltf::Image &image) {
    AllocatedImage newImage{};

    int width, height, nrChannels;

    std::visit(fastgltf::visitor{
                       [](auto &arg) {},
                       [&](fastgltf::sources::URI &filePath) {
                           assert(filePath.fileByteOffset == 0);
                           assert(filePath.uri.isLocalPath());

                           const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
                           unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
                           if (data) {
                               vk::Extent3D imagesize;
                               imagesize.width = width;
                               imagesize.height = height;
                               imagesize.depth = 1;

                               newImage = engine->create_image(data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                               vk::ImageUsageFlagBits::eSampled, false);

                               stbi_image_free(data);
                           }
                       },
                       [&](fastgltf::sources::Vector &vector) {
                           unsigned char *data = stbi_load_from_memory(
                                   reinterpret_cast<unsigned char *>(vector.bytes.data()),
                                   static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
                           if (data) {
                               vk::Extent3D imagesize;
                               imagesize.width = width;
                               imagesize.height = height;
                               imagesize.depth = 1;

                               newImage = engine->create_image(data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                               vk::ImageUsageFlagBits::eSampled, false);

                               stbi_image_free(data);
                           }
                       },
                       [&](fastgltf::sources::BufferView &view) {
                           auto &bufferView = asset.bufferViews[view.bufferViewIndex];
                           auto &buffer = asset.buffers[bufferView.bufferIndex];
                           std::visit(
                                   fastgltf::visitor{[](auto &arg) { fmt::println("Unexpected buffer source type!"); },
                                                     [&](fastgltf::sources::Array &blob) {
                                                         unsigned char *data = stbi_load_from_memory(
                                                                 reinterpret_cast<stbi_uc const *>(
                                                                         blob.bytes.data() + bufferView.byteOffset),
                                                                 static_cast<int>(bufferView.byteLength), &width,
                                                                 &height, &nrChannels, 4);
                                                         if (data) {
                                                             vk::Extent3D imagesize;
                                                             imagesize.width = width;
                                                             imagesize.height = height;
                                                             imagesize.depth = 1;

                                                             newImage = engine->create_image(
                                                                     data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                                     vk::ImageUsageFlagBits::eSampled, false);

                                                             stbi_image_free(data);
                                                         }
                                                     },
                                                     [&](fastgltf::sources::Vector &vector) {
                                                         unsigned char *data = stbi_load_from_memory(
                                                                 reinterpret_cast<unsigned char *>(
                                                                         vector.bytes.data() + bufferView.byteOffset),
                                                                 static_cast<int>(bufferView.byteLength), &width,
                                                                 &height, &nrChannels, 4);
                                                         if (data) {
                                                             vk::Extent3D imagesize;
                                                             imagesize.width = width;
                                                             imagesize.height = height;
                                                             imagesize.depth = 1;

                                                             newImage = engine->create_image(
                                                                     data, imagesize, vk::Format::eR8G8B8A8Unorm,
                                                                     vk::ImageUsageFlagBits::eSampled, false);

                                                             stbi_image_free(data);
                                                         }
                                                     }},
                                   buffer.data);
                       },
               },
               image.data);

    if (newImage.image == VK_NULL_HANDLE) {
        return {};
    }

    return newImage;
}

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(WaterEngine *engine, std::string_view filePath) {
    fmt::print("Loading GLTF: {}", filePath);

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF &file = *scene.get();

    fastgltf::Parser parser{};

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble |
                                 fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages;

    auto dataRes = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (!dataRes) {
        std::cout << "Failed to load GLTF: " << filePath << std::endl;
        return {};
    }

    fastgltf::GltfDataBuffer data;
    data = std::move(dataRes.get());

    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(data);
    if (type == fastgltf::GltfType::glTF) {
        auto load = parser.loadGltf(data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        } else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    } else if (type == fastgltf::GltfType::GLB) {
        auto load = parser.loadGltfBinary(data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        } else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    } else {
        std::cerr << "Failed to determine glTF container" << std::endl;
        return {};
    }

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = {
            {vk::DescriptorType::eCombinedImageSampler, 3},
            {vk::DescriptorType::eUniformBuffer, 3},
            {vk::DescriptorType::eStorageBuffer, 1},
    };

    file.descriptorPool.init(engine->_device, gltf.materials.size(), sizes);

    for (fastgltf::Sampler &sampler: gltf.samplers) {
        vk::SamplerCreateInfo sampl =
                vk::SamplerCreateInfo()
                        .setMaxLod(vk::LodClampNone)
                        .setMinLod(0.f)
                        .setMagFilter(extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest)))
                        .setMinFilter(extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)))
                        .setMipmapMode(extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest)));

        vk::Sampler newSampler = engine->_device.createSampler(sampl);

        file.samplers.push_back(newSampler);
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    for (fastgltf::Image &image: gltf.images) {
        std::optional<AllocatedImage> img = load_image(engine, gltf, image);

        if (img.has_value()) {
            images.push_back(*img);
            file.images[image.name.c_str()] = std::make_shared<AllocatedImage>(*img);
        } else {
            images.push_back(engine->_errorCheckerboardImage);
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
    }

    file.materialDataBuffer =
            engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(),
                                  vk::BufferUsageFlagBits::eUniformBuffer, vma::MemoryUsage::eCpuToGpu);
    int data_index = 0;
    GLTFMetallic_Roughness::MaterialConstants *sceneMaterialConstants =
            (GLTFMetallic_Roughness::MaterialConstants *) file.materialDataBuffer.info.pMappedData;

    for (fastgltf::Material &mat: gltf.materials) {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        GLTFMetallic_Roughness::MaterialConstants constants;
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
        constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;
        // write material parameters to buffer
        sceneMaterialConstants[data_index] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend) {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_Roughness::MaterialResources materialResources;
        // default the material textures
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        // set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(GLTFMetallic_Roughness::MaterialConstants);
        // grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value()) {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }
        // build material
        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources,
                                                                 file.descriptorPool);

        data_index++;
    }

    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh &mesh: gltf.meshes) {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto &&p: mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = static_cast<uint32_t>(indices.size());
            newSurface.count = static_cast<uint32_t>(gltf.accessors[p.indicesAccessor.value()].count);

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor &indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(
                        gltf, indexaccessor, [&](std::uint32_t idx) { indices.push_back(idx + initial_vtx); });
            }

            // load vertex positions
            {
                fastgltf::Accessor &posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor, [&](glm::vec3 v, size_t index) {
                    Vertex newvtx;
                    newvtx.position = v;
                    newvtx.normal = {1, 0, 0};
                    newvtx.color = glm::vec4{1.f};
                    newvtx.uv_x = 0;
                    newvtx.uv_y = 0;
                    vertices[initial_vtx + index] = newvtx;
                });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(
                        gltf, gltf.accessors[(*normals).accessorIndex],
                        [&](glm::vec3 v, size_t index) { vertices[initial_vtx + index].normal = v; });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).accessorIndex],
                                                              [&](glm::vec2 v, size_t index) {
                                                                  vertices[initial_vtx + index].uv_x = v.x;
                                                                  vertices[initial_vtx + index].uv_y = v.y;
                                                              });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(
                        gltf, gltf.accessors[(*colors).accessorIndex],
                        [&](glm::vec4 v, size_t index) { vertices[initial_vtx + index].color = v; });
            }

            if (p.materialIndex.has_value()) {
                newSurface.material = materials[p.materialIndex.value()];
            } else {
                newSurface.material = materials[0];
            }

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }

    for (fastgltf::Node &node: gltf.nodes) {
        std::shared_ptr<Node> newNode;

        if (node.meshIndex.has_value()) {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode *>(newNode.get())->mesh = meshes[*node.meshIndex];
        } else {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(fastgltf::visitor{[&](fastgltf::math::fmat4x4 matrix) {
                                         memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
                                     },
                                     [&](fastgltf::TRS transform) {
                                         glm::vec3 tl(transform.translation[0], transform.translation[1],
                                                      transform.translation[2]);
                                         glm::quat rot(transform.rotation[3], transform.rotation[0],
                                                       transform.rotation[1], transform.rotation[2]);
                                         glm::vec3 sc(transform.scale[0], transform.scale[1], transform.scale[2]);

                                         glm::mat4 tm = glm::translate(glm::mat4(1.f), tl);
                                         glm::mat4 rm = glm::toMat4(rot);
                                         glm::mat4 sm = glm::scale(glm::mat4(1.f), sc);

                                         newNode->localTransform = tm * rm * sm;
                                     }},
                   node.transform);
    }

    for (int i = 0; i < gltf.nodes.size(); i++) {
        fastgltf::Node &node = gltf.nodes[i];
        std::shared_ptr<Node> &sceneNode = nodes[i];

        for (auto &c: node.children) {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    for (auto &node: nodes) {
        if (node->parent.lock() == nullptr) {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{1.f});
        }
    }

    return scene;
}

void LoadedGLTF::Draw(const glm::mat4 &topMatrix, DrawContext &ctx) {
    for (auto &n: topNodes) {
        n->Draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll() {
    vk::Device dv = creator->_device;

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);

    for (auto &[k, v]: meshes) {
        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto &[k, v]: images) {
        if (v->image == creator->_errorCheckerboardImage.image) {
            continue;
        }
        creator->destroy_image(*v);
    }

    for (auto &sampler: samplers) {
        dv.destroySampler(sampler);
    }
}
