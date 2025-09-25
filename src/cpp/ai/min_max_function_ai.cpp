//
// Created by CorruptionHades on 25/09/2025.
//

#include <iostream>

#include "nn/NeuralNetwork.h"
#include "utils/SetupUtil.h"

std::pair<std::vector<std::vector<float> >, std::vector<std::vector<float> > >
generateMinMaxData(const int sampleCount) {
    std::vector<std::vector<float> > inputs;
    std::vector<std::vector<float> > targets;

    for (int i = 0; i < sampleCount; ++i) {
        float a = static_cast<float>(rand() % 100);
        float b = static_cast<float>(rand() % 100);
        inputs.push_back({a, b});
        if (a > b) {
            targets.push_back({1.0f, 0.0f}); // a is max
        } else if (a < b) {
            targets.push_back({0.0f, 1.0f}); // b is max
        } else {
            targets.push_back({1.0f, 1.0f}); // equal
        }
    }

    return {inputs, targets};
}

int main() {
    if (setupOpenGLWindow() != 0) {
        std::cerr << "Failed to set up OpenGL window." << std::endl;
        return -1;
    }

    // A simple neural network to output the minimum and maximum of two numbers.
    /*
     * a min or max function
     * it has 2 inputs and 2 outputs
     * The first output is for the first input, the second output is for the second input
     * the network outputs which input is the maximum or minimum (1 for max, 0 for min)
    */

    //region Example training data
    // Training data: [input1, input2] -> [is_input1_bigger, is_input2_bigger]
    const std::vector<std::vector<float> > inputs1 = {
        {5, 2}, // input1 > input2
        {1, 3}, // input2 > input1
        {7, 7}, // equal
        {0, -1}, // input1 > input2
        {-2, 4} // input2 > input1
    };
    const std::vector<std::vector<float> > targets1 = {
        {1, 0}, // input1 > input2
        {0, 1}, // input2 > input1
        {1, 1}, // equal
        {1, 0}, // input1 > input2
        {0, 1} // input2 > input1
    };

    // Generate data
    const auto minMaxData = generateMinMaxData(500);
    const auto &inputs = minMaxData.first;
    const auto &targets = minMaxData.second;

    std::cout << "Training data prepared with " << inputs.size() << " samples." << std::endl;
    //endregion

    NeuralNetwork nn{};
    std::cout << "Neural network initialized." << std::endl;
    nn.addLayer(2, 2);
    std::cout << "Added input layer." << std::endl;

    // Train the network
    std::cout << "Starting training..." << std::endl;
    for (int epoch = 0; epoch < 300; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            nn.train(inputs[i], targets[i]);
        }
        std::cout << "Epoch " << epoch + 1 << "/300 completed." << std::endl;
    }
    std::cout << "Training completed." << std::endl;

    // Test the network
    std::vector<std::vector<float> > testInputs = {
        {4, 9}, // input2 > input1
        {10, 3}, // input1 > input2
        {5, 5} // equal
    };
    for (const auto &testInput: testInputs) {
        auto output = nn.predict(testInput);
        std::cout << "Input: " << testInput[0] << ", " << testInput[1]
                << " | Output: [" << output[0] << ", " << output[1] << "]" << std::endl;
    }

    cleanupOpenGLWindow();

    return 0;
}
