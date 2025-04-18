#pragma once

#include <vk_include.h>
#include <vma/vk_mem_alloc.h>
#include <vma/vk_mem_alloc.hpp>

#include <fmt/core.h>

#ifndef VK_TYPES_H
#define VK_TYPES_H

struct AllocatedImage {
    vk::Image image;
    vk::ImageView imageView;
    vma::Allocation allocation;
    vk::Extent3D imageExtent;
    vk::Format imageFormat;
};

#endif // VK_TYPES_H
