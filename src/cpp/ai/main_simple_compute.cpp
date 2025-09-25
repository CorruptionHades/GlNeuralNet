//
// Created by CorruptionHades on 19/09/2025.
//


#include <GL/glew.h>
#include <iostream>

#include "utils/SetupUtil.h"
#include "gl/Shader.h"
#include "nn/Matrix.h"

int mainc() {
    if (setupOpenGLWindow() != 0) {
        std::cerr << "Failed to set up OpenGL window." << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // --- 2. Load and Compile Shaders ---
    Shader matmulShader{};
    matmulShader.loadComputeShader("H:/C++/GlNeuralNet/src/shaders/matmul.comp");
    std::cout << "Loaded Matrix Multiplication Shader." << std::endl;

    // --- 3. Example: Matrix Multiplication on the GPU ---
    std::cout << "\n--- GPU Matrix Multiplication Example ---" << std::endl;

    // Create CPU-side matrices
    const Matrix matA = Matrix::random(4, 2);
    const Matrix matB = Matrix::random(2, 4);
    Matrix matC(matA.rows, matB.cols); // To hold the result

    std::cout << "Matrix A:" << std::endl;
    matA.print();
    std::cout << "\nMatrix B:" << std::endl;
    matB.print();

    // Create GPU buffers (Shader Storage Buffer Objects)
    GLuint ssbo_A, ssbo_B, ssbo_C;
    glGenBuffers(1, &ssbo_A);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_A);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matA.data.size() * sizeof(float), matA.data.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ssbo_B);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_B);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matB.data.size() * sizeof(float), matB.data.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &ssbo_C);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_C);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matC.data.size() * sizeof(float), nullptr, GL_STATIC_READ);

    // --- 4. Run the Compute Shader ---
    matmulShader.use();

    // Set uniforms
    matmulShader.setInt("u_A_rows", matA.rows);
    matmulShader.setInt("u_A_cols", matA.cols);
    matmulShader.setInt("u_B_cols", matB.cols);

    // Bind buffers to binding points specified in the shader
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_A);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_B);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssbo_C);

    // Calculate workgroup counts
    // We want one thread per element in the output matrix C.
    // The local workgroup size is 16x16, so we divide the total size by the local size.
    const GLuint group_x = (matC.cols + 15) / 16; // Ceiling division
    const GLuint group_y = (matC.rows + 15) / 16;

    matmulShader.dispatch(group_x, group_y, 1);

    // --- 5. Retrieve the result ---
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_C);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, matC.data.size() * sizeof(float), matC.data.data());

    std::cout << "\nResult C (from GPU):" << std::endl;
    matC.print();

    // --- 6. Cleanup ---
    glDeleteBuffers(1, &ssbo_A);
    glDeleteBuffers(1, &ssbo_B);
    glDeleteBuffers(1, &ssbo_C);

    cleanupOpenGLWindow();

    return 0;
}