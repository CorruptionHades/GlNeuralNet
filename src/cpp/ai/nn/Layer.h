//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef LAYER_H
#define LAYER_H

#include <GL/glew.h>
#include "../gl/Shader.h"
#include <nlohmann/json.hpp>

// Enum for activation function types, making the code more readable.
enum ActivationType {
    SIGMOID = 0,
    SIGMOID_DERIVATIVE = 1
};

class Layer {
public:
    int inputSize;
    int neuronCount;

    // --- GPU Buffer Handles ---
    GLuint weightsBuffer;
    GLuint biasesBuffer;
    GLuint lastInputBuffer;
    GLuint lastWeightedSumBuffer;
    GLuint gradWeightsBuffer;
    GLuint gradBiasesBuffer;
    GLuint deltaBuffer; // To store the error δ for this layer

    Layer(int inSize, int outSize, Shader *matmul, Shader *matmul_T, Shader *elementwise,
          Shader *activation, Shader *outer_prod, Shader *sgd_update);

    ~Layer();

    // Disallow copying to prevent issues with GPU resource management.
    Layer(const Layer &) = delete;

    Layer &operator=(const Layer &) = delete;

    void forward(GLuint inputBuffer, GLuint outputBuffer);

    /**
     * @brief Backward pass for the OUTPUT layer.
     * @param errorFromOutput The SSBO containing the initial error (prediction - target).
     */
    void backward(GLuint errorFromOutput);

    /**
     * @brief Backward pass for HIDDEN layers.
     * @param errorFromNextLayer The SSBO containing the error δ from the layer ahead.
     * @param weightsOfNextLayer The SSBO containing the weights W of the layer ahead.
     * @param errorForPrevLayer The SSBO where this function will store the calculated error for the previous layer.
     */
    void backward(GLuint errorFromNextLayer, GLuint weightsOfNextLayer, GLuint errorForPrevLayer);

    /**
     * @brief Updates the layer's weights and biases using the computed gradients and learning rate.
     */
    void update(float learningRate);

    // saving/loading
    [[nodiscard]] nlohmann::json toJson() const;

    void loadParameters(const nlohmann::json &j);

private:
    Shader *matmulShader;
    Shader *matmulTransposeAShader;
    Shader *elementwiseShader;
    Shader *activationShader;
    Shader *outerProductShader;
    Shader *sgdUpdateShader;
};

#endif
