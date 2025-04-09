cmake_minimum_required(VERSION 3.30)
project(WaterDemo)

set(CMAKE_CXX_STANDARD 23)

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)

if (MSVC)
    add_compile_options(/Zc:__cplusplus)
endif ()

find_package(glfw3)
find_package(glm)
find_package(imgui)
find_package(Eigen3)
find_package(Vulkan REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${Vulkan_INCLUDE_DIRS})

# ----------------------
# Shader Compilation
# ----------------------

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

file(GLOB_RECURSE SHADER_FILES
        ${SHADER_SOURCE_DIR}/*.vert
        ${SHADER_SOURCE_DIR}/*.frag
)

set(COMPILED_SHADERS)

foreach (SHADER ${SHADER_FILES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(OUTPUT_FILE ${SHADER_BINARY_DIR}/${FILE_NAME}.spv)

    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND glslangValidator -V ${SHADER} -o ${OUTPUT_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader  ${FILE_NAME}"
            VERBATIM
    )

    list(APPEND COMPILED_SHADERS ${OUTPUT_FILE})
endforeach ()

add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})

# -----------------------------
# Model and texture build files
# -----------------------------

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/textures DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ---------------------------
# vk-bootstrap
# ---------------------------
include(FetchContent)
FetchContent_Declare(
        fetch_vk_bootstrap
        GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
        GIT_TAG main
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)

add_executable(WaterDemo main.cpp
        simulation/ShallowWaterSolver.cpp
        simulation/ShallowWaterSolver.h
        simulation/FFTOcean.cpp
        utils/VulkanUtils.cpp
        utils/VulkanUtils.h)

add_dependencies(WaterDemo compile_shaders)

target_link_libraries(WaterDemo glfw)
target_link_libraries(WaterDemo glm::glm)
target_link_libraries(WaterDemo imgui::imgui)
target_link_libraries(WaterDemo Eigen3::Eigen)
target_link_libraries(WaterDemo Vulkan::Vulkan)
target_link_libraries(WaterDemo vk-bootstrap::vk-bootstrap)