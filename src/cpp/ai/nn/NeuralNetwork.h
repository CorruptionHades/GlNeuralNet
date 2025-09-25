//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <memory>
#include <string>
#include "Layer.h"
#include "../gl/Shader.h"

class NeuralNetwork {
public:
    float learningRate;

    /**
     * @brief Constructs the Neural Network, loading all necessary shaders.
     */
    NeuralNetwork();

    /**
     * @brief Destructor that cleans up GPU buffer resources.
     */
    ~NeuralNetwork();

    /**
     * @brief Adds the first layer, establishing the network's input size.
     */
    void addLayer(int inputSize, int neuronCount);

    /**
     * @brief Adds a subsequent hidden or output layer.
     */
    void addLayer(int neuronCount);

    /**
     * @brief Performs a full forward pass through all layers.
     */
    std::vector<float> predict(const std::vector<float>& inputData);

    /**
     * @brief Performs one full training step (forward pass, backpropagation, and parameter update).
     */
    void train(const std::vector<float>& inputData, const std::vector<float>& targetData);

    void saveToFile(const std::string& path) const;

    /**
     * @brief Loads a network from a file, completely reconstructing it.
     * @param path The file path to load the model from.
     * @return A unique_ptr to the newly created NeuralNetwork instance.
     */
    static std::unique_ptr<NeuralNetwork> loadFromFile(const std::string& path);

    // get input size
    [[nodiscard]] int getInputSize() const {
        if (layerSizes.empty()) {
            throw std::runtime_error("Network has no layers.");
        }
        return layerSizes.front();
    }
private:
    // The shaders are owned by the network and shared among layers via pointers
    Shader matmulShader;
    Shader matmulTransposeAShader;
    Shader elementwiseShader;
    Shader activationShader;
    Shader outerProductShader;
    Shader sgdUpdateShader;

    // A list of all layers in the network
    std::vector<std::unique_ptr<Layer>> layers;

    // Buffers to hold the activations and errors of each layer.
    std::vector<GLuint> activationBuffers;
    std::vector<GLuint> errorBuffers;

    // Stores the number of neurons in each layer, starting with the input size.
    std::vector<int> layerSizes;

    // Disallow copying.
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
};

#endif