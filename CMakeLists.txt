cmake_minimum_required(VERSION 3.30)
project(WaterDemo)

set(CMAKE_CXX_STANDARD 23)

find_package(glfw3)
find_package(glm)
find_package(imgui)
find_package(Vulkan REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

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

add_executable(WaterDemo main.cpp)

add_dependencies(WaterDemo compile_shaders)

target_link_libraries(WaterDemo glfw)
target_link_libraries(WaterDemo glm::glm)
target_link_libraries(WaterDemo imgui::imgui)
target_link_libraries(WaterDemo Vulkan::Vulkan)