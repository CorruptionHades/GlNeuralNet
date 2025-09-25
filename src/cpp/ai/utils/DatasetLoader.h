//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <vector>
#include <string>

// A simple struct to hold our dataset in an organized way.
// This makes returning the data from the load function clean and easy.
struct TrainingData {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;
};

// A namespace for our utility functions. This avoids polluting the global namespace.
namespace DatasetLoader {

    /**
     * @brief Loads the player and not-player datasets from text files.
     * @param playerPath The file path to the dataset of "player" images.
     * @param notPlayerPath The file path to the dataset of "not-player" images.
     * @param inputSize The required number of features (pixels) for the network's input layer.
     * @return A TrainingData struct containing all inputs and their corresponding targets.
     */
    TrainingData load(const std::string& playerPath,
                      const std::string& notPlayerPath,
                      size_t inputSize);

} // namespace DatasetLoader

#endif //DATASETLOADER_H
