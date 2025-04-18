#include "vk_images.h"
#include "vk_initializers.h"

void vkutil::transition_image(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout currentLayout,
                              vk::ImageLayout newLayout) {
    vk::ImageAspectFlags aspectMask = (newLayout == vk::ImageLayout::eDepthAttachmentOptimal)
                                              ? vk::ImageAspectFlagBits::eDepth
                                              : vk::ImageAspectFlagBits::eColor;
    vk::ImageMemoryBarrier2 imageBarrier =
            vk::ImageMemoryBarrier2()
                    .setPNext(nullptr)
                    .setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                    .setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite)
                    .setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands)
                    .setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)

                    .setOldLayout(currentLayout)
                    .setNewLayout(newLayout)

                    .setSubresourceRange(vkinit::image_subresource_range(aspectMask))
                    .setImage(image);

    vk::DependencyInfo depInfo =
            vk::DependencyInfo().setPNext(nullptr).setImageMemoryBarrierCount(1).setPImageMemoryBarriers(&imageBarrier);

    cmd.pipelineBarrier2(depInfo);
}

void vkutil::copy_image_to_image(vk::CommandBuffer cmd, vk::Image source, vk::Image destination, vk::Extent2D srcSize,
                                 vk::Extent2D dstSize) {
    vk::ImageBlit2 blitRegion = {};
    blitRegion.srcOffsets[1].x = srcSize.width;
    blitRegion.srcOffsets[1].y = srcSize.height;
    blitRegion.srcOffsets[1].z = 1;

    blitRegion.dstOffsets[1].x = dstSize.width;
    blitRegion.dstOffsets[1].y = dstSize.height;
    blitRegion.dstOffsets[1].z = 1;

    blitRegion.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blitRegion.srcSubresource.baseArrayLayer = 0;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcSubresource.mipLevel = 0;

    blitRegion.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blitRegion.dstSubresource.baseArrayLayer = 0;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstSubresource.mipLevel = 0;

    vk::BlitImageInfo2 blitInfo = vk::BlitImageInfo2()
                                          .setDstImage(destination)
                                          .setDstImageLayout(vk::ImageLayout::eTransferDstOptimal)
                                          .setSrcImage(source)
                                          .setSrcImageLayout(vk::ImageLayout::eTransferSrcOptimal)
                                          .setFilter(vk::Filter::eLinear)
                                          .setRegionCount(1)
                                          .setPRegions(&blitRegion);

    cmd.blitImage2(&blitInfo);
}
