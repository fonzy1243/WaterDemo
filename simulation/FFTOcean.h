#pragma once

#define OCEAN_SIZE 512.0f
#define OCEAN_SIZE_INT int(OCEAN_SIZE)
#include <glm/glm.hpp>

struct OceanParameters {
    glm::vec2 resolution = glm::vec2(OCEAN_SIZE);
    glm::vec2 windDirection = glm::vec2(13.0f, 1.0f);
    float windSpeed = 30.0f;
    float g = 9.8f;
    float waveAmplitude = 1.0f;
    float t = 0.0f;
    float lengthScale = 0.01f;
};

struct IFFTParameters {
    uint32_t stage;
    uint32_t direction;
    uint32_t resolution;
    uint32_t pingPong;
};

struct ButterflyParameters {
    uint32_t resolution;
};
