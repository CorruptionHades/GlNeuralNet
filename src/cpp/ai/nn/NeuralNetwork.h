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

    NeuralNetwork();

    ~NeuralNetwork();

    /**
     * Adds input layer. Must be called first.
     * @param inputSize The size of the input vector.
     * @param neuronCount The number of neurons in this layer.
     */
    void addLayer(int inputSize, int neuronCount);

    void addLayer(int neuronCount);

    /**
     * @brief Performs a full forward pass through all layers.
     */
    std::vector<float> predict(const std::vector<float> &inputData);

    /**
     * @brief Performs one full training step (forward pass, backpropagation, and parameter update).
     */
    void train(const std::vector<float> &inputData, const std::vector<float> &targetData);

    void saveToFile(const std::string &path) const;

    /**
     * @brief Loads a network from a file
     * @param path The file path to load the model from.
     * @return A unique_ptr to the newly created NeuralNetwork instance.
     */
    static std::unique_ptr<NeuralNetwork> loadFromFile(const std::string &path);

    // get input size
    [[nodiscard]] int getInputSize() const {
        if (layerSizes.empty()) {
            throw std::runtime_error("Network has no layers.");
        }
        return layerSizes.front();
    }

private:
    Shader matmulShader;
    Shader matmulTransposeAShader;
    Shader elementwiseShader;
    Shader activationShader;
    Shader outerProductShader;
    Shader sgdUpdateShader;

    // A list of all layers in the network
    std::vector<std::unique_ptr<Layer> > layers;

    // Buffers to hold the activations and errors of each layer.
    std::vector<GLuint> activationBuffers;
    std::vector<GLuint> errorBuffers;

    // Stores the number of neurons in each layer, starting with the input size.
    std::vector<int> layerSizes;

    // Disallow copying.
    NeuralNetwork(const NeuralNetwork &) = delete;

    NeuralNetwork &operator=(const NeuralNetwork &) = delete;
};

#endif
