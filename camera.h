//
// Created by Alfon on 5/4/2025.
//
#pragma once

#include <SDL3/SDL_events.h>
#include "vk_types.h"

#ifndef CAMERA_H
#define CAMERA_H

class Camera {
public:
    glm::vec3 velocity;
    glm::vec3 position;
    // vertical rotation
    float pitch{0.f};
    // horizontal rotation
    float yaw{0.f};

    glm::mat4 getViewMatrix();
    glm::mat4 getRotationMatrix();

    void processSDLEvent(SDL_Event &e);

    void update();
};

#endif // CAMERA_H
