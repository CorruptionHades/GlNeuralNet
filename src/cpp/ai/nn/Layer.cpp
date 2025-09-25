//
// Created by CorruptionHades on 19/09/2025.
//

#include "Layer.h"
#include "Matrix.h" // For initialization
#include <iostream>

Layer::Layer(int inSize, int outSize, Shader* matmul, Shader* matmul_T, Shader* elementwise,
             Shader* activation, Shader* outer_prod, Shader* sgd_update)
    : inputSize(inSize),
      neuronCount(outSize),
      matmulShader(matmul),
      matmulTransposeAShader(matmul_T),
      elementwiseShader(elementwise),
      activationShader(activation),
      outerProductShader(outer_prod),
      sgdUpdateShader(sgd_update)
{
    // 1. Initialize weights and biases on the CPU first for random values
    Matrix weights = Matrix::random(neuronCount, inputSize);
    Matrix biases = Matrix(neuronCount, 1); // Biases initialized to zero

    std::cout << "Initializing Layer (" << inputSize << " -> " << neuronCount << ")..." << std::endl;

    // 2. Generate all necessary GPU buffers
    glGenBuffers(1, &weightsBuffer);
    glGenBuffers(1, &biasesBuffer);
    glGenBuffers(1, &lastInputBuffer);
    glGenBuffers(1, &lastWeightedSumBuffer);
    glGenBuffers(1, &gradWeightsBuffer);
    glGenBuffers(1, &gradBiasesBuffer);
    glGenBuffers(1, &deltaBuffer);

    // 3. Allocate and upload initial data for weights and biases
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightsBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, weights.data.size() * sizeof(float), weights.data.data(), GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, biasesBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, biases.data.size() * sizeof(float), biases.data.data(), GL_DYNAMIC_COPY);

    // 4. Allocate empty buffers for intermediate and gradient values
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lastInputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lastWeightedSumBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, neuronCount * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradWeightsBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, weights.data.size() * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gradBiasesBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, biases.data.size() * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, deltaBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, neuronCount * sizeof(float), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); // Unbind
}

Layer::~Layer() {
    // Free all GPU resources when the layer is destroyed
    glDeleteBuffers(1, &weightsBuffer);
    glDeleteBuffers(1, &biasesBuffer);
    glDeleteBuffers(1, &lastInputBuffer);
    glDeleteBuffers(1, &lastWeightedSumBuffer);
    glDeleteBuffers(1, &gradWeightsBuffer);
    glDeleteBuffers(1, &gradBiasesBuffer);
    glDeleteBuffers(1, &deltaBuffer);
}

void Layer::forward(GLuint inputBuffer, GLuint outputBuffer) {
    // Step 0: Save the input for the backward pass
    glBindBuffer(GL_COPY_READ_BUFFER, inputBuffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, lastInputBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, inputSize * sizeof(float));

    // Step 1: Weighted Sum (z = W * a_prev)
    matmulShader->use();
    matmulShader->setInt("u_A_rows", neuronCount);
    matmulShader->setInt("u_A_cols", inputSize);
    matmulShader->setInt("u_B_cols", 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, inputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, lastWeightedSumBuffer);
    matmulShader->dispatch(1, (neuronCount + 15) / 16, 1);

    // Step 2: Add Biases (z = z + b)
    elementwiseShader->use();
    elementwiseShader->setInt("u_op_type", 0); // Addition
    elementwiseShader->setInt("u_element_count", neuronCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lastWeightedSumBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, biasesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, lastWeightedSumBuffer); // In-place
    elementwiseShader->dispatch((neuronCount + 255) / 256, 1, 1);

    // Step 3: Activation (a = g(z))
    activationShader->use();
    activationShader->setInt("u_func_type", SIGMOID);
    activationShader->setInt("u_element_count", neuronCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lastWeightedSumBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
    activationShader->dispatch((neuronCount + 255) / 256, 1, 1);
}

// For the OUTPUT layer
void Layer::backward(GLuint errorFromOutput) {
    // For the last layer, the error δ is simply (prediction - target), which is computed
    // in the train function and passed here. We just copy it to our internal deltaBuffer.
    glBindBuffer(GL_COPY_READ_BUFFER, errorFromOutput);
    glBindBuffer(GL_COPY_WRITE_BUFFER, deltaBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, neuronCount * sizeof(float));

    // --- Calculate Gradients ---
    // ∇W = δ * transpose(a_prev) -> outer product
    outerProductShader->use();
    outerProductShader->setInt("u_A_rows", neuronCount);
    outerProductShader->setInt("u_B_cols", inputSize);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, deltaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, lastInputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gradWeightsBuffer);
    outerProductShader->dispatch((inputSize + 15) / 16, (neuronCount + 15) / 16, 1);

    // ∇b = δ -> it's just a copy
    glBindBuffer(GL_COPY_READ_BUFFER, deltaBuffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, gradBiasesBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, neuronCount * sizeof(float));
}

// For HIDDEN layers
void Layer::backward(GLuint errorFromNextLayer, GLuint weightsOfNextLayer, GLuint errorForPrevLayer) {
    // --- Calculate δ_l = (transpose(W_{l+1}) * δ_{l+1}) .* g'(z_l) ---
    // Part A: Propagated error: (transpose(W_{l+1}) * δ_{l+1})
    GLint nextLayerNeuronCount;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, errorFromNextLayer);
    glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &nextLayerNeuronCount);
    nextLayerNeuronCount /= sizeof(float);

    matmulTransposeAShader->use();
    matmulTransposeAShader->setInt("u_A_rows", nextLayerNeuronCount);
    matmulTransposeAShader->setInt("u_A_cols", neuronCount);
    matmulTransposeAShader->setInt("u_B_cols", 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsOfNextLayer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, errorFromNextLayer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, errorForPrevLayer);
    matmulTransposeAShader->dispatch(1, (neuronCount + 15) / 16, 1);

    // Part B: Activation derivative: g'(z_l)
    activationShader->use();
    activationShader->setInt("u_func_type", SIGMOID_DERIVATIVE);
    activationShader->setInt("u_element_count", neuronCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lastWeightedSumBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, deltaBuffer); // Store derivative temporarily in deltaBuffer
    activationShader->dispatch((neuronCount + 255) / 256, 1, 1);

    // Part C: Element-wise product to get final δ_l
    elementwiseShader->use();
    elementwiseShader->setInt("u_op_type", 2); // Multiplication
    elementwiseShader->setInt("u_element_count", neuronCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, errorForPrevLayer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, deltaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, deltaBuffer); // Overwrite with final result
    elementwiseShader->dispatch((neuronCount + 255) / 256, 1, 1);

    // --- Calculate Gradients (same as for the output layer) ---
    outerProductShader->use();
    outerProductShader->setInt("u_A_rows", neuronCount);
    outerProductShader->setInt("u_B_cols", inputSize);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, deltaBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, lastInputBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gradWeightsBuffer);
    outerProductShader->dispatch((inputSize + 15) / 16, (neuronCount + 15) / 16, 1);

    glBindBuffer(GL_COPY_READ_BUFFER, deltaBuffer);
    glBindBuffer(GL_COPY_WRITE_BUFFER, gradBiasesBuffer);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, neuronCount * sizeof(float));
}

void Layer::update(float learningRate) {
    sgdUpdateShader->use();
    glUniform1f(glGetUniformLocation(sgdUpdateShader->ID, "u_learning_rate"), learningRate);

    // Update Weights: W = W - lr * ∇W
    sgdUpdateShader->setInt("u_element_count", neuronCount * inputSize);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, weightsBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradWeightsBuffer);
    sgdUpdateShader->dispatch(((neuronCount * inputSize) + 255) / 256, 1, 1);

    // Update Biases: b = b - lr * ∇b
    sgdUpdateShader->setInt("u_element_count", neuronCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, biasesBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gradBiasesBuffer);
    sgdUpdateShader->dispatch((neuronCount + 255) / 256, 1, 1);
}

nlohmann::json Layer::toJson() const {
    // 1. Create CPU-side vectors to hold the data
    std::vector<float> weights_data(neuronCount * inputSize);
    std::vector<float> biases_data(neuronCount);

    // 2. Download data from GPU buffers to CPU vectors
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightsBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights_data.size() * sizeof(float), weights_data.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, biasesBuffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, biases_data.size() * sizeof(float), biases_data.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // 3. Create JSON object and populate it
    nlohmann::json j;
    j["weights"] = weights_data;
    j["biases"] = biases_data;

    return j;
}

void Layer::loadParameters(const nlohmann::json& j) {
    // 1. Extract data from JSON into CPU-side vectors
    std::vector<float> weights_data = j.at("weights").get<std::vector<float>>();
    std::vector<float> biases_data = j.at("biases").get<std::vector<float>>();

    // Verify sizes to prevent buffer overflows
    if (weights_data.size() != neuronCount * inputSize || biases_data.size() != neuronCount) {
        throw std::runtime_error("Mismatched data size when loading layer parameters.");
    }

    // 2. Upload data from CPU vectors to existing GPU buffers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, weightsBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, weights_data.size() * sizeof(float), weights_data.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, biasesBuffer);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, biases_data.size() * sizeof(float), biases_data.data());

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}