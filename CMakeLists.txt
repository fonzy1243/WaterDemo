cmake_minimum_required(VERSION 3.30)
project(WaterDemo)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
add_compile_definitions(GLM_FORCE_DEPTH_ZERO_TO_ONE)

if (MSVC)
    add_compile_options(/Zc:__cplusplus)
    add_compile_options(-Wno-nullability-completeness)
endif ()

# find_package(glfw3)
add_subdirectory(SDL EXCLUDE_FROM_ALL)
find_package(fmt)
find_package(glm)
# find_package(imgui)
find_package(Eigen3)
find_package(Vulkan REQUIRED)
find_package(fastgltf)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/vma)
include_directories(${Vulkan_INCLUDE_DIRS})

# ----------------------
# imgui
# ----------------------

set(IMGUI_DIR imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# ----------------------
# Shader Compilation
# ----------------------

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

file(GLOB_RECURSE SHADER_FILES
        ${SHADER_SOURCE_DIR}/*.vert
        ${SHADER_SOURCE_DIR}/*.frag
        ${SHADER_SOURCE_DIR}/*.comp
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

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

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
        simulation/FFTOcean.h
        include/tiny_obj_loader.cpp
        water_engine.cpp
        water_engine.h
        vk_initializers.cpp
        vk_initializers.h
        vk_types.h
        include/vk_disp_loader.cpp
        include/vk_include.h
        vk_images.cpp
        vk_images.h
        vk_descriptors.cpp
        vk_descriptors.h
        vk_pipelines.cpp
        vk_pipelines.h
        ${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
        ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        vk_loader.cpp
        vk_loader.h
        camera.cpp
        camera.h
)

add_dependencies(WaterDemo compile_shaders)

target_link_libraries(WaterDemo SDL3::SDL3)
target_link_libraries(WaterDemo glm::glm)
target_link_libraries(WaterDemo Eigen3::Eigen)
target_link_libraries(WaterDemo Vulkan::Vulkan)
target_link_libraries(WaterDemo vk-bootstrap::vk-bootstrap)
target_link_libraries(WaterDemo fmt::fmt)
target_link_libraries(WaterDemo fastgltf::fastgltf)