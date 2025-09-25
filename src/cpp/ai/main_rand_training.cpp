//
// Created by CorruptionHades on 19/09/2025.
//

#include <iostream>
#include <vector>
#include <iomanip>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "nn/NeuralNetwork.h"
#include "utils/SetupUtil.h"

void print_vector_as_matrix(const std::vector<float> &vec, const int rows, const int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << vec[i * cols + j] << "  ";
        }
        std::cout << std::endl;
    }
}

int mainr() {
    setupOpenGLWindow();

    NeuralNetwork nn;
    nn.addLayer(4, 8);
    nn.addLayer(2); // Output layer has 2 neurons
    std::cout << "Created a 4 -> 8 -> 2 network." << std::endl;

    const std::vector inputData = {0.8f, 0.2f, 0.5f, 0.1f};
    const std::vector targetData = {1.0f, 0.0f}; // Target output

    std::cout << "\n--- Initial Prediction ---" << std::endl;
    const std::vector<float> initialPrediction = nn.predict(inputData);
    print_vector_as_matrix(initialPrediction, initialPrediction.size(), 1);

    std::cout << "\n--- Performing One Training Step ---" << std::endl;
    nn.train(inputData, targetData);
    std::cout << "Train step executed." << std::endl;

    std::cout << "\n--- Prediction After Training ---" << std::endl;
    const std::vector<float> predictionAfterTrain = nn.predict(inputData);
    print_vector_as_matrix(predictionAfterTrain, predictionAfterTrain.size(), 1);

    cleanupOpenGLWindow();
    return 0;
}
