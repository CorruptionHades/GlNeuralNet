//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef SHADER_H
#define SHADER_H

#include <string>
#include <GL/glew.h>

class Shader {
public:
    GLuint ID;

    Shader() = default;

    void loadComputeShader(const std::string &shaderPath);

    void use() const;

    void setInt(const std::string &name, int value) const;

    /**
     * Dispatches a compute shader.
     * The global work group counts are the total number of invocations you want.
     * For example, for an 800x600 image, you'd call dispatch(800, 600, 1).
     * The local workgroup size is defined inside the shader file.
    */
    void dispatch(GLuint group_x, GLuint group_y, GLuint group_z) const;

private:
    void checkCompileErrors(GLuint shader, const std::string &type);
};

#endif
