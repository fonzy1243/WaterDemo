#pragma once

#include "vk_types.h"

#ifndef VK_PIPELINES_H
#define VK_PIPELINES_H

namespace vkutil {
    bool load_shader_module(const char *filePath, vk::Device device, vk::ShaderModule *shaderModule);
}

#endif // VK_PIPELINES_H
