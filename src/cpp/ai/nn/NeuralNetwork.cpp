//
// Created by CorruptionHades on 19/09/2025.
//

#include "NeuralNetwork.h"

#include <fstream>
#include <stdexcept>
#include <iostream>

NeuralNetwork::NeuralNetwork() : learningRate(0.1f) {
    // Load all the shaders once when the network is created
    matmulShader.loadComputeShader("shaders/matmul.comp");
    matmulTransposeAShader.loadComputeShader("shaders/matmul_transpose_A.comp");
    elementwiseShader.loadComputeShader("shaders/elementwise.comp");
    activationShader.loadComputeShader("shaders/activation.comp");
    outerProductShader.loadComputeShader("shaders/outer_product.comp");
    sgdUpdateShader.loadComputeShader("shaders/sgd_update.comp");
}

NeuralNetwork::~NeuralNetwork() {
    // Clean up all the network-managed GPU buffers
    glDeleteBuffers(activationBuffers.size(), activationBuffers.data());
    glDeleteBuffers(errorBuffers.size(), errorBuffers.data());
}

void NeuralNetwork::addLayer(int inputSize, int neuronCount) {
    if (!layers.empty()) {
        throw std::runtime_error("This method can only be used for the first layer.");
    }
    layerSizes.push_back(inputSize);

    // Create the very first activation buffer, which will hold the network's input
    GLuint inputActBuffer;
    glGenBuffers(1, &inputActBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, inputActBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    activationBuffers.push_back(inputActBuffer);

    // Now add the actual layer
    addLayer(neuronCount);
}

void NeuralNetwork::addLayer(int neuronCount) {
    if (layerSizes.empty()) {
        throw std::runtime_error("You must call the addLayer(inputSize, neuronCount) overload for the first layer.");
    }

    int inputSize = layerSizes.back();
    layers.emplace_back(std::make_unique<Layer>(inputSize, neuronCount, &matmulShader, &matmulTransposeAShader,
                                                &elementwiseShader, &activationShader, &outerProductShader, &sgdUpdateShader));
    layerSizes.push_back(neuronCount);

    // Create a new activation buffer and error buffer for the output of this new layer
    GLuint newActBuffer, newErrorBuffer;
    glGenBuffers(1, &newActBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, newActBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, neuronCount * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    activationBuffers.push_back(newActBuffer);

    glGenBuffers(1, &newErrorBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, newErrorBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, neuronCount * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    errorBuffers.push_back(newErrorBuffer);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

std::vector<float> NeuralNetwork::predict(const std::vector<float>& inputData) {
    if (layers.empty()) throw std::runtime_error("Cannot predict with an empty network.");
    if (inputData.size() != layerSizes.front()) throw std::invalid_argument("Input data size does not match network input size.");

    // Step 1: Upload input data to the first activation buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, activationBuffers[0]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, inputData.size() * sizeof(float), inputData.data());

    // Step 2: Propagate through all layers
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i]->forward(activationBuffers[i], activationBuffers[i + 1]);
    }

    // Step 3: Download the result from the last buffer
    int outputSize = layerSizes.back();
    std::vector<float> outputData(outputSize);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, activationBuffers.back());
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, outputData.size() * sizeof(float), outputData.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    return outputData;
}

void NeuralNetwork::train(const std::vector<float>& inputData, const std::vector<float>& targetData) {
    // 1. Forward pass (leaves activations in GPU buffers)
    predict(inputData);

    // 2. Calculate initial error at the output layer: δ_L = prediction - target
    int outputSize = layerSizes.back();
    GLuint targetBuffer;
    glGenBuffers(1, &targetBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, targetBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, targetData.size() * sizeof(float), targetData.data(), GL_STATIC_DRAW);

    elementwiseShader.use();
    elementwiseShader.setInt("u_op_type", 1); // Subtract
    elementwiseShader.setInt("u_element_count", outputSize);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, activationBuffers.back()); // prediction
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, targetBuffer);             // target
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, errorBuffers.back());      // result -> output error δ_L
    elementwiseShader.dispatch((outputSize + 255) / 256, 1, 1);
    glDeleteBuffers(1, &targetBuffer);

    // 3. Backward Pass
    // First, process the output layer (L) using its specialized backward method
    layers.back()->backward(errorBuffers.back());

    // Then, propagate the error backward through the hidden layers (L-1 to 1)
    for (int i = layers.size() - 2; i >= 0; --i) {
        GLuint errorFromNextLayer = errorBuffers[i + 1];
        GLuint weightsOfNextLayer = layers[i + 1]->weightsBuffer;
        GLuint errorForPrevLayer  = errorBuffers[i];
        layers[i]->backward(errorFromNextLayer, weightsOfNextLayer, errorForPrevLayer);
    }

    // 4. Update Parameters for all layers
    for (auto& layer : layers) {
        layer->update(learningRate);
    }
}

using json = nlohmann::json;

// Add these new methods at the end of the file
void NeuralNetwork::saveToFile(const std::string& path) const {
    json j;
    j["learning_rate"] = this->learningRate;
    j["architecture"] = this->layerSizes; // Save the full architecture [input, hidden1, ..., output]

    j["layers"] = json::array();
    for (const auto& layer : layers) {
        j["layers"].push_back(layer->toJson());
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }
    file << j.dump(4); // .dump(4) pretty-prints the JSON with an indent of 4 spaces
    file.close();
}

std::unique_ptr<NeuralNetwork> NeuralNetwork::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + path);
    }

    json j = json::parse(file);
    file.close();

    // 1. Reconstruct the network with the correct architecture
    auto nn = std::make_unique<NeuralNetwork>();
    nn->learningRate = j.at("learning_rate");

    std::vector<int> arch = j.at("architecture").get<std::vector<int>>();
    if (arch.size() < 2) {
        throw std::runtime_error("Invalid architecture in model file.");
    }

    // Create the network layer by layer
    nn->addLayer(arch[0], arch[1]); // First layer is special
    for (size_t i = 2; i < arch.size(); ++i) {
        nn->addLayer(arch[i]);
    }

    // 2. Load the parameters into each layer
    if (j.at("layers").size() != nn->layers.size()) {
        throw std::runtime_error("Mismatched layer count in model file.");
    }
    for (size_t i = 0; i < nn->layers.size(); ++i) {
        nn->layers[i]->loadParameters(j["layers"][i]);
    }

    return nn;
}

