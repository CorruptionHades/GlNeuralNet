//
// Created by CorruptionHades on 19/09/2025.
//

#include "SetupUtil.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>


GLFWwindow *window;

void error_callback(int error, const char *description) {
    std::cerr << "Error: " << description << std::endl;
}

int setupOpenGLWindow() {
    // --- 1. Initialize GLFW and GLEW ---
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    glfwSetErrorCallback(error_callback);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // hide window
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(640, 480, "Compute", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // create context for shaders
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    return 0;
}

void cleanupOpenGLWindow() {
    if (!window) return;

    glfwDestroyWindow(window);
    glfwTerminate();
}
