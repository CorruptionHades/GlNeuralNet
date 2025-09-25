//
// Created by CorruptionHades on 19/09/2025.
//
/*
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <iomanip>

#include "gl/SetupUtil.h"
#include "gl/Shader.h"
#include "nn/Layer.h"

// Helper function to print a vector as a matrixx
void print_vector_as_matrixx(const std::vector<float>& vec, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << vec[i * cols + j] << "  ";
        }
        std::cout << std::endl;
    }
}

int maina() {
    if(setupOpenGLWindow() != 0) {
        std::cerr << "Failed to set up OpenGL window." << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // --- 2. Load All Shaders ---
    Shader matmulShader, elementwiseShader, activationShader;
    matmulShader.loadComputeShader("shaders/matmul.comp");
    elementwiseShader.loadComputeShader("shaders/elementwise.comp");
    activationShader.loadComputeShader("shaders/activation.comp");
    std::cout << "Loaded all shaders." << std::endl;

    // --- 3. Test a Single Layer ---
    const int INPUT_SIZE = 4;
    const int NEURON_COUNT = 2;
    Layer testLayer(INPUT_SIZE, NEURON_COUNT, &matmulShader, &elementwiseShader, &activationShader);

    // --- 4. Create CPU-side input data and GPU buffers ---
    std::vector<float> inputData = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> outputData(NEURON_COUNT);

    GLuint inputBuffer, outputBuffer;
    glGenBuffers(1, &inputBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, inputData.size() * sizeof(float), inputData.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &outputBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, outputData.size() * sizeof(float), nullptr, GL_STATIC_READ);

    std::cout << "\n--- Testing Layer Forward Pass ---" << std::endl;
    std::cout << "Input Vector:" << std::endl;
    print_vector_as_matrixx(inputData, INPUT_SIZE, 1);

    // --- 5. Run the forward pass ---
    testLayer.forward(inputBuffer, outputBuffer);
    std::cout << "\nForward pass executed on GPU." << std::endl;

    // --- 6. Retrieve and print the result ---
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, outputData.size() * sizeof(float), outputData.data());

    std::cout << "\nOutput Vector (from GPU):" << std::endl;
    print_vector_as_matrixx(outputData, NEURON_COUNT, 1);

    // --- 7. Cleanup ---
    glDeleteBuffers(1, &inputBuffer);
    glDeleteBuffers(1, &outputBuffer);

    cleanupOpenGLWindow();

    return 0;
}
*/