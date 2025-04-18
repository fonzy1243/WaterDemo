//
// Created by Alfon on 4/16/2025.
//

#include "vk_pipelines.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "vk_initializers.h"

bool vkutil::load_shader_module(const char *filePath, vk::Device device, vk::ShaderModule *outShaderModule) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        std::cout << "Current path: " << std::filesystem::current_path() << std::endl;
        return false;
    }

    size_t fileSize = (size_t) file.tellg();

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);

    file.read((char *) buffer.data(), fileSize);

    file.close();

    vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
                                                    .setPNext(nullptr)
                                                    .setCodeSize(buffer.size() * sizeof(uint32_t))
                                                    .setPCode(buffer.data());

    vk::ShaderModule shaderModule = {};

    auto loadResult = device.createShaderModule(&createInfo, nullptr, &shaderModule);
    if (loadResult != vk::Result::eSuccess) {
        fmt::print("Failed to create shader module:\n", static_cast<long long>(loadResult));
        return false;
    }

    *outShaderModule = shaderModule;
    return true;
}
