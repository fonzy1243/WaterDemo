//
// Created by Alfon on 4/16/2025.
//

#include "vk_pipelines.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "vk_initializers.h"

void PipelineBuilder::clear() {
    _inputAssembly = vk::PipelineInputAssemblyStateCreateInfo();

    _rasterizer = vk::PipelineRasterizationStateCreateInfo();

    _colorBlendAttachment = vk::PipelineColorBlendAttachmentState();

    _multisampling = vk::PipelineMultisampleStateCreateInfo();

    _pipelineLayout = vk::PipelineLayout();

    _depthStencil = vk::PipelineDepthStencilStateCreateInfo();

    _renderInfo = vk::PipelineRenderingCreateInfo();

    _shaderStages.clear();
}

vk::Pipeline PipelineBuilder::build_pipeline(vk::Device device) {
    vk::PipelineViewportStateCreateInfo viewportState =
            vk::PipelineViewportStateCreateInfo().setPNext(nullptr).setViewportCount(1).setScissorCount(1);

    vk::PipelineColorBlendStateCreateInfo colorBlending = vk::PipelineColorBlendStateCreateInfo()
                                                                  .setPNext(nullptr)
                                                                  .setLogicOpEnable(vk::False)
                                                                  .setLogicOp(vk::LogicOp::eCopy)
                                                                  .setAttachmentCount(1)
                                                                  .setPAttachments(&_colorBlendAttachment);

    vk::PipelineVertexInputStateCreateInfo _vertexInputInfo = vk::PipelineVertexInputStateCreateInfo();

    vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
                                                          .setPNext(&_renderInfo)
                                                          .setStageCount((uint32_t) _shaderStages.size())
                                                          .setPStages(_shaderStages.data())
                                                          .setPVertexInputState(&_vertexInputInfo)
                                                          .setPInputAssemblyState(&_inputAssembly)
                                                          .setPViewportState(&viewportState)
                                                          .setPRasterizationState(&_rasterizer)
                                                          .setPMultisampleState(&_multisampling)
                                                          .setPColorBlendState(&colorBlending)
                                                          .setPDepthStencilState(&_depthStencil)
                                                          .setLayout(_pipelineLayout);

    vk::DynamicState state[] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicInfo =
            vk::PipelineDynamicStateCreateInfo().setPDynamicStates(&state[0]).setDynamicStateCount(2);

    pipelineInfo.setPDynamicState(&dynamicInfo);

    vk::Pipeline newPipeline;

    vk::Result createResult = device.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline);

    if (createResult != vk::Result::eSuccess) {
        fmt::println("Failed to create graphics pipeline: {}", static_cast<long long>(createResult));
        return VK_NULL_HANDLE;
    } else {
        return newPipeline;
    }
}

void PipelineBuilder::set_shaders(vk::ShaderModule vertexShader, vk::ShaderModule fragmentShader) {
    _shaderStages.clear();

    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::eVertex, vertexShader));

    _shaderStages.push_back(
            vkinit::pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::eFragment, fragmentShader));
}

void PipelineBuilder::set_input_topology(vk::PrimitiveTopology topology) {
    _inputAssembly.setTopology(topology);
    _inputAssembly.setPrimitiveRestartEnable(vk::False);
}

void PipelineBuilder::set_polygon_mode(vk::PolygonMode mode) {
    _rasterizer.setPolygonMode(mode);
    _rasterizer.setLineWidth(1.f);
}

void PipelineBuilder::set_cull_mode(vk::CullModeFlags cullMode, vk::FrontFace frontFace) {
    _rasterizer.setCullMode(cullMode);
    _rasterizer.setFrontFace(frontFace);
}

void PipelineBuilder::set_multisampling_none() {
    _multisampling.setSampleShadingEnable(vk::False);
    // defaulted to no multisampling
    _multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
    _multisampling.setMinSampleShading(1.f);
    _multisampling.setPSampleMask(nullptr);
    // no alpha to converge
    _multisampling.setAlphaToCoverageEnable(vk::False);
    _multisampling.setAlphaToOneEnable(vk::False);
}

void PipelineBuilder::disable_blending() {
    // default write mask
    _colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    // no blending
    _colorBlendAttachment.setBlendEnable(vk::False);
}

void PipelineBuilder::enable_blending_additive() {
    _colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    _colorBlendAttachment.setBlendEnable(vk::True);
    _colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha);
    _colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eOne);
    _colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);
    _colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
    _colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
    _colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);
}

void PipelineBuilder::enable_blending_alphablend() {
    _colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    _colorBlendAttachment.setBlendEnable(vk::True);
    _colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha);
    _colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha);
    _colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);
    _colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
    _colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
    _colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);
}

void PipelineBuilder::set_color_attachment_format(vk::Format format) {
    _colorAttachmentFormat = format;
    _renderInfo.setColorAttachmentCount(1);
    _renderInfo.setPColorAttachmentFormats(&_colorAttachmentFormat);
}

void PipelineBuilder::set_depth_format(vk::Format format) { _renderInfo.setDepthAttachmentFormat(format); }

void PipelineBuilder::disable_depthtest() {
    _depthStencil.setDepthTestEnable(vk::False);
    _depthStencil.setDepthWriteEnable(vk::False);
    _depthStencil.setDepthCompareOp(vk::CompareOp::eNever);
    _depthStencil.setDepthBoundsTestEnable(vk::False);
    _depthStencil.setStencilTestEnable(vk::False);
    _depthStencil.setFront(vk::StencilOpState());
    _depthStencil.setBack(vk::StencilOpState());
    _depthStencil.setMinDepthBounds(0.f);
    _depthStencil.setMaxDepthBounds(1.f);
}

void PipelineBuilder::enable_depthtest(bool depthWriteEnable, vk::CompareOp op) {
    _depthStencil.setDepthTestEnable(vk::True);
    _depthStencil.setDepthWriteEnable(depthWriteEnable);
    _depthStencil.setDepthCompareOp(op);
    _depthStencil.setDepthBoundsTestEnable(vk::False);
    _depthStencil.setStencilTestEnable(vk::False);
    _depthStencil.setFront(vk::StencilOpState());
    _depthStencil.setBack(vk::StencilOpState());
    _depthStencil.setMinDepthBounds(0.f);
    _depthStencil.setMaxDepthBounds(1.f);
}


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
