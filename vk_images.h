#pragma once

#include "vk_types.h"

#ifndef VK_IMAGES_H
#define VK_IMAGES_H

namespace vkutil {
    void transition_image(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout currentLayout,
                          vk::ImageLayout newLayout);

    void copy_image_to_image(vk::CommandBuffer cmd, vk::Image source, vk::Image destination, vk::Extent2D srcSize,
                             vk::Extent2D dstSize);

    void generate_mipmaps(vk::CommandBuffer cmd, vk::Image image, vk::Extent2D imageSize);
} // namespace vkutil

#endif // VK_IMAGES_H
