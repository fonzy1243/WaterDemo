#pragma once

#include "vk_types.h"

#ifndef VK_PIPELINES_H
#define VK_PIPELINES_H

class PipelineBuilder {
public:
    std::vector<vk::PipelineShaderStageCreateInfo> _shaderStages;

    vk::PipelineInputAssemblyStateCreateInfo _inputAssembly;
    vk::PipelineRasterizationStateCreateInfo _rasterizer;
    vk::PipelineColorBlendAttachmentState _colorBlendAttachment;
    vk::PipelineMultisampleStateCreateInfo _multisampling;
    vk::PipelineLayout _pipelineLayout;
    vk::PipelineDepthStencilStateCreateInfo _depthStencil;
    vk::PipelineRenderingCreateInfo _renderInfo;
    vk::Format _colorAttachmentFormat;

    PipelineBuilder() { clear(); }

    void clear();

    vk::Pipeline build_pipeline(vk::Device device);

    void set_shaders(vk::ShaderModule vertexShader, vk::ShaderModule fragmentShader);
    void set_input_topology(vk::PrimitiveTopology topology);
    void set_polygon_mode(vk::PolygonMode mode);
    void set_cull_mode(vk::CullModeFlags cullMode, vk::FrontFace frontFace);
    void set_multisampling_none();
    void disable_blending();
    void enable_blending_additive();
    void enable_blending_alphablend();

    void set_color_attachment_format(vk::Format format);
    void set_depth_format(vk::Format format);
    void disable_depthtest();
    void enable_depthtest(bool depthWriteEnable, vk::CompareOp op);
};

namespace vkutil {
    bool load_shader_module(const char *filePath, vk::Device device, vk::ShaderModule *shaderModule);
}

#endif // VK_PIPELINES_H
