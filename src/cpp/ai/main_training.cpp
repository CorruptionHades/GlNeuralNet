#include <iostream>
#include <vector>
#include <iomanip>

#include <chrono>
#include <numeric>
#include <random>

#include "nn/NeuralNetwork.h"
#include "utils/DatasetLoader.h"
#include "utils/SetupUtil.h"

int mainTrain() {
    setupOpenGLWindow();

    // --- 1. Network Setup ---
    constexpr int INPUT_SIZE = 450 * 450;
    constexpr int HIDDEN_SIZE = 128;
    constexpr int OUTPUT_SIZE = 1;
    constexpr int epochs = 10;

    NeuralNetwork nn;
    nn.learningRate = 0.01;
    nn.addLayer(INPUT_SIZE, HIDDEN_SIZE);
    nn.addLayer(OUTPUT_SIZE);
    std::cout << "Created a " << INPUT_SIZE << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << " network." << std::endl;

    // --- 2. Load Dataset ---
    TrainingData data = DatasetLoader::load("H:/Dart/LearnAI/src/fromscratch/img_class/datasets/dataset_players.txt",
                                            "H:/Dart/LearnAI/src/fromscratch/img_class/datasets/dataset_not_players.txt",
                                            INPUT_SIZE);

    // --- 3. Training Loop ---
    std::cout << "\n--- Starting Training for " << epochs << " epochs on " << data.inputs.size() << " samples ---" << std::endl;

    // Create an index vector to shuffle data without copying it
    std::vector<size_t> indices(data.inputs.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::ranges::shuffle(indices, std::mt19937{std::random_device{}()});

        auto epoch_start = std::chrono::high_resolution_clock::now();
        int correctPredictions = 0;

        for (size_t i = 0; i < data.inputs.size(); ++i) {
            // Use the shuffled index to get the training sample
            const size_t sample_idx = indices[i];
            nn.train(data.inputs[sample_idx], data.targets[sample_idx]);
        }

        // --- Validation and Metrics after each epoch ---
        for (size_t i = 0; i < data.inputs.size(); ++i) {
            std::vector<float> prediction = nn.predict(data.inputs[i]);
            const int predictedLabel = (prediction[0] > 0.5f) ? 1 : 0;
            const int actualLabel = static_cast<int>(data.targets[i][0]);
            if (predictedLabel == actualLabel) {
                correctPredictions++;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

        const double accuracy = static_cast<double>(correctPredictions) / data.inputs.size() * 100.0;
        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << epochs
                  << " - Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%"
                  << " - Time: " << duration.count() << "s" << std::endl;
    }

    std::cout << "--- Training Complete ---" << std::endl;

    const long msSinceEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    const std::string modelPath = "model_" + std::to_string(msSinceEpoch) + ".json";
    nn.saveToFile(modelPath);
    std::cout << "Model saved to " << modelPath << std::endl;

    cleanupOpenGLWindow();

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}