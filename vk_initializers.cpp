//
// Created by Alfon on 4/14/2025.
//

#include "vk_initializers.h"

vk::CommandPoolCreateInfo vkinit::command_pool_create_info(uint32_t queueFamilyIndex,
                                                           vk::CommandPoolCreateFlags flags) {
    vk::CommandPoolCreateInfo info =
            vk::CommandPoolCreateInfo().setPNext(nullptr).setFlags(flags).setQueueFamilyIndex(queueFamilyIndex);

    return info;
}

vk::CommandBufferAllocateInfo vkinit::command_buffer_allocate_info(vk::CommandPool pool, uint32_t count,
                                                                   vk::CommandBufferLevel level) {
    vk::CommandBufferAllocateInfo info = vk::CommandBufferAllocateInfo()
                                                 .setPNext(nullptr)
                                                 .setCommandPool(pool)
                                                 .setCommandBufferCount(count)
                                                 .setLevel(level);

    return info;
}

vk::CommandBufferBeginInfo vkinit::command_buffer_begin_info(vk::CommandBufferUsageFlags flags) {
    vk::CommandBufferBeginInfo info =
            vk::CommandBufferBeginInfo().setPNext(nullptr).setPInheritanceInfo(nullptr).setFlags(flags);

    return info;
}


vk::FenceCreateInfo vkinit::fence_create_info(vk::FenceCreateFlags flags) {
    vk::FenceCreateInfo info = vk::FenceCreateInfo().setPNext(nullptr).setFlags(flags);

    return info;
}


vk::SemaphoreCreateInfo vkinit::semaphore_create_info(vk::SemaphoreCreateFlags flags) {
    vk::SemaphoreCreateInfo info = vk::SemaphoreCreateInfo().setPNext(nullptr).setFlags(flags);

    return info;
}

vk::SemaphoreSubmitInfo vkinit::semaphore_submit_info(vk::PipelineStageFlags2 stageMask, vk::Semaphore semaphore) {
    vk::SemaphoreSubmitInfo submitInfo = vk::SemaphoreSubmitInfo()
                                                 .setPNext(nullptr)
                                                 .setSemaphore(semaphore)
                                                 .setStageMask(stageMask)
                                                 .setDeviceIndex(0)
                                                 .setValue(1);

    return submitInfo;
}

vk::CommandBufferSubmitInfo vkinit::command_buffer_submit_info(vk::CommandBuffer cmd) {
    vk::CommandBufferSubmitInfo info =
            vk::CommandBufferSubmitInfo().setPNext(nullptr).setCommandBuffer(cmd).setDeviceMask(0);

    return info;
}

vk::SubmitInfo2 vkinit::submit_info(vk::CommandBufferSubmitInfo *cmd, vk::SemaphoreSubmitInfo *signalSemaphoreInfo,
                                    vk::SemaphoreSubmitInfo *waitSemaphoreInfo) {
    vk::SubmitInfo2 info = vk::SubmitInfo2()
                                   .setPNext(nullptr)
                                   .setWaitSemaphoreInfoCount(waitSemaphoreInfo == nullptr ? 0 : 1)
                                   .setPWaitSemaphoreInfos(waitSemaphoreInfo)
                                   .setSignalSemaphoreInfoCount(signalSemaphoreInfo == nullptr ? 0 : 1)
                                   .setPSignalSemaphoreInfos(signalSemaphoreInfo)
                                   .setCommandBufferInfoCount(1)
                                   .setPCommandBufferInfos(cmd);

    return info;
}

vk::PresentInfoKHR vkinit::present_info() {
    vk::PresentInfoKHR info = vk::PresentInfoKHR()
                                      .setPNext(nullptr)
                                      .setSwapchainCount(0)
                                      .setPSwapchains(nullptr)
                                      .setPWaitSemaphores(nullptr)
                                      .setWaitSemaphoreCount(0)
                                      .setPImageIndices(nullptr);

    return info;
}

vk::RenderingAttachmentInfo vkinit::attachment_info(vk::ImageView view, vk::ClearValue *clear, vk::ImageLayout layout) {
    vk::RenderingAttachmentInfo colorAttachment =
            vk::RenderingAttachmentInfo()
                    .setPNext(nullptr)
                    .setImageView(view)
                    .setImageLayout(layout)
                    .setLoadOp(clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad)
                    .setStoreOp(vk::AttachmentStoreOp::eStore);

    if (clear) {
        colorAttachment.setClearValue(*clear);
    }

    return colorAttachment;
}

vk::RenderingAttachmentInfo vkinit::depth_attachment_info(vk::ImageView view, vk::ImageLayout layout) {
    vk::RenderingAttachmentInfo depthAttachment = vk::RenderingAttachmentInfo()
                                                          .setPNext(nullptr)
                                                          .setImageView(view)
                                                          .setImageLayout(layout)
                                                          .setLoadOp(vk::AttachmentLoadOp::eClear)
                                                          .setStoreOp(vk::AttachmentStoreOp::eStore);

    depthAttachment.clearValue.depthStencil.depth = 0.0f;

    return depthAttachment;
}

vk::RenderingInfo vkinit::rendering_info(vk::Extent2D renderExtent, vk::RenderingAttachmentInfo *colorAttachment,
                                         vk::RenderingAttachmentInfo *depthAttachment) {
    vk::RenderingInfo renderInfo = vk::RenderingInfo()
                                           .setPNext(nullptr)
                                           .setRenderArea(vk::Rect2D(vk::Offset2D(0, 0), renderExtent))
                                           .setLayerCount(1)
                                           .setColorAttachmentCount(1)
                                           .setPColorAttachments(colorAttachment)
                                           .setPDepthAttachment(depthAttachment)
                                           .setPStencilAttachment(nullptr);

    return renderInfo;
}

vk::ImageSubresourceRange vkinit::image_subresource_range(vk::ImageAspectFlags aspectMask) {
    vk::ImageSubresourceRange subImage = vk::ImageSubresourceRange()
                                                 .setAspectMask(aspectMask)
                                                 .setBaseMipLevel(0)
                                                 .setLevelCount(vk::RemainingMipLevels)
                                                 .setBaseArrayLayer(0)
                                                 .setLayerCount(vk::RemainingArrayLayers);

    return subImage;
}

vk::DescriptorSetLayoutBinding vkinit::descriptorset_layout_binding(vk::DescriptorType type,
                                                                    vk::ShaderStageFlags stageFlags, uint32_t binding) {
    vk::DescriptorSetLayoutBinding setbind = vk::DescriptorSetLayoutBinding()
                                                     .setBinding(binding)
                                                     .setDescriptorCount(1)
                                                     .setDescriptorType(type)
                                                     .setPImmutableSamplers(nullptr)
                                                     .setStageFlags(stageFlags);

    return setbind;
}

vk::WriteDescriptorSet vkinit::write_descriptor_image(vk::DescriptorType type, vk::DescriptorSet dstSet,
                                                      vk::DescriptorImageInfo *imageInfo, uint32_t binding) {
    vk::WriteDescriptorSet write = vk::WriteDescriptorSet()
                                           .setPNext(nullptr)
                                           .setDstBinding(binding)
                                           .setDstSet(dstSet)
                                           .setDescriptorCount(1)
                                           .setDescriptorType(type)
                                           .setPImageInfo(imageInfo);

    return write;
}

vk::DescriptorBufferInfo vkinit::buffer_info(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize range) {
    vk::DescriptorBufferInfo binfo = vk::DescriptorBufferInfo().setBuffer(buffer).setOffset(offset).setRange(range);

    return binfo;
}


vk::ImageCreateInfo vkinit::image_create_info(vk::Format format, vk::ImageUsageFlags usageFlags, vk::Extent3D extent) {
    vk::ImageCreateInfo info = vk::ImageCreateInfo()
                                       .setPNext(nullptr)
                                       .setImageType(vk::ImageType::e2D)
                                       .setFormat(format)
                                       .setExtent(extent)
                                       .setMipLevels(1)
                                       .setArrayLayers(1)
                                       .setSamples(vk::SampleCountFlagBits::e1)
                                       .setTiling(vk::ImageTiling::eOptimal)
                                       .setUsage(usageFlags);

    return info;
}

vk::ImageViewCreateInfo vkinit::imageview_create_info(vk::Format format, vk::Image image,
                                                      vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo info = vk::ImageViewCreateInfo()
                                           .setPNext(nullptr)
                                           .setViewType(vk::ImageViewType::e2D)
                                           .setImage(image)
                                           .setFormat(format)
                                           .setSubresourceRange(vk::ImageSubresourceRange()
                                                                        .setBaseMipLevel(0)
                                                                        .setLevelCount(1)
                                                                        .setBaseArrayLayer(0)
                                                                        .setLayerCount(1)
                                                                        .setAspectMask(aspectFlags));

    return info;
}

vk::PipelineLayoutCreateInfo vkinit::pipeline_layout_create_info() {
    vk::PipelineLayoutCreateInfo info = vk::PipelineLayoutCreateInfo()
                                                .setPNext(nullptr)
                                                .setFlags(vk::PipelineLayoutCreateFlags())
                                                .setSetLayoutCount(0)
                                                .setPSetLayouts(nullptr)
                                                .setPushConstantRangeCount(0)
                                                .setPPushConstantRanges(nullptr);

    return info;
}

vk::PipelineShaderStageCreateInfo vkinit::pipeline_shader_stage_create_info(vk::ShaderStageFlagBits stage,
                                                                            vk::ShaderModule shaderModule,
                                                                            const char *entry) {
    vk::PipelineShaderStageCreateInfo info = vk::PipelineShaderStageCreateInfo()
                                                     .setPNext(nullptr)
                                                     .setStage(stage)
                                                     .setModule(shaderModule)
                                                     .setPName(entry);

    return info;
}
