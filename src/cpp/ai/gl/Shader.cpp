//
// Created by CorruptionHades on 19/09/2025.
//

#include "Shader.h"
#include <fstream>
#include <sstream>
#include <iostream>

void Shader::loadComputeShader(const std::string &shaderPath) {
    std::string shaderCode;
    std::ifstream shaderFile;

    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        shaderFile.open(shaderPath);
        std::stringstream shaderStream;
        shaderStream << shaderFile.rdbuf();
        shaderFile.close();
        shaderCode = shaderStream.str();
    } catch (std::ifstream::failure &e) {
        std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << shaderPath << std::endl;
    }

    const char *cShaderCode = shaderCode.c_str();
    const GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &cShaderCode, nullptr);
    glCompileShader(computeShader);
    checkCompileErrors(computeShader, "COMPUTE");

    ID = glCreateProgram();
    glAttachShader(ID, computeShader);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    glDeleteShader(computeShader);
}

void Shader::use() const {
    glUseProgram(ID);
}

void Shader::setInt(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::dispatch(const GLuint group_x, const GLuint group_y, const GLuint group_z) const {
    glDispatchCompute(group_x, group_y, group_z);
    // Block until all writes from this shader call are complete
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Shader::checkCompileErrors(const GLuint shader, const std::string &type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog <<
                    "\n -- --------------------------------------------------- -- " << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog <<
                    "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}
