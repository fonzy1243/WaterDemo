#pragma once

#include "vk_types.h"

#ifndef VK_INITIALIZERS_H
#define VK_INITIALIZERS_H

namespace vkinit {
    vk::CommandPoolCreateInfo
    command_pool_create_info(uint32_t queueFamilyIndex,
                             vk::CommandPoolCreateFlags flags = vk::CommandPoolCreateFlagBits());
    vk::CommandBufferAllocateInfo
    command_buffer_allocate_info(vk::CommandPool pool, uint32_t count = 1,
                                 vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);

    vk::CommandBufferBeginInfo
    command_buffer_begin_info(vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits());
    vk::CommandBufferSubmitInfo command_buffer_submit_info(vk::CommandBuffer cmd);

    vk::FenceCreateInfo fence_create_info(vk::FenceCreateFlags flags = vk::FenceCreateFlagBits());

    vk::SemaphoreCreateInfo semaphore_create_info(vk::SemaphoreCreateFlags flags = vk::SemaphoreCreateFlagBits());


    vk::SubmitInfo2 submit_info(vk::CommandBufferSubmitInfo *cmd, vk::SemaphoreSubmitInfo *signalSemaphoreInfo,
                                vk::SemaphoreSubmitInfo *waitSemaphoreInfo);
    vk::PresentInfoKHR present_info();

    vk::RenderingAttachmentInfo attachment_info(vk::ImageView view, vk::ClearValue *clear, vk::ImageLayout layout);

    vk::RenderingAttachmentInfo depth_attachment_info(vk::ImageView view, vk::ImageLayout layout);

    vk::RenderingInfo rendering_info(vk::Extent2D renderExtent, vk::RenderingAttachmentInfo *colorAttachment,
                                     vk::RenderingAttachmentInfo *depthAttachment);

    vk::ImageSubresourceRange image_subresource_range(vk::ImageAspectFlags aspectMask);

    vk::SemaphoreSubmitInfo semaphore_submit_info(vk::PipelineStageFlags2 stageMask, vk::Semaphore semaphore);
    vk::DescriptorSetLayoutBinding descriptorset_layout_binding(vk::DescriptorType type,
                                                                vk::ShaderStageFlags stageFlags, uint32_t binding);
    vk::DescriptorSetLayoutCreateInfo descriptorset_layout_create_info(vk::DescriptorSetLayoutBinding *bindings,
                                                                       uint32_t bindingCount);
    vk::WriteDescriptorSet write_descriptor_image(vk::DescriptorType type, vk::DescriptorSet dstSet,
                                                  vk::DescriptorImageInfo *imageInfo, uint32_t binding);
    vk::WriteDescriptorSet write_descriptor_buffer(vk::DescriptorType type, vk::DescriptorSet dstSet,
                                                   uint32_t bindingCount);
    vk::DescriptorBufferInfo buffer_info(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize range);

    vk::ImageCreateInfo image_create_info(vk::Format format, vk::ImageUsageFlags usageFlags, vk::Extent3D extent);
    vk::ImageViewCreateInfo imageview_create_info(vk::Format, vk::Image image, vk::ImageAspectFlags aspectFlags);
} // namespace vkinit

#endif // VK_INITIALIZERS_H
