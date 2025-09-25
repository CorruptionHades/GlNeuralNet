//
// Created by CorruptionHades on 19/09/2025.
//

#include "DatasetLoader.h"
#include <iostream>
#include <fstream>
#include <chrono>   // For timing the load operation
#include <stdexcept>

namespace {

std::vector<float> parseImageLine(const std::string& line, size_t inputSize) {
    // Find the first space to separate the "size" from the "pixels" string.
    size_t space_pos = line.find(' ');
    if (space_pos == std::string::npos) {
        std::cerr << "Warning: Malformed line detected, skipping." << std::endl;
        return {};
    }

    std::string pixel_str = line.substr(space_pos + 1);

    std::vector<float> pixels;
    pixels.reserve(pixel_str.length());

    for (char c : pixel_str) {
        pixels.push_back(static_cast<float>(c - '0'));
    }

    pixels.resize(inputSize, 0.0f);

    return pixels;
}

}


namespace DatasetLoader {

TrainingData load(const std::string& playerPath,
                  const std::string& notPlayerPath,
                  size_t inputSize)
{
    std::cout << "Loading dataset..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    TrainingData data;
    std::string line;

    // --- Load Player Images (Target = 1.0) ---
    std::ifstream playerFile(playerPath);
    if (!playerFile.is_open()) {
        throw std::runtime_error("Error: Could not open player dataset file: " + playerPath);
    }

    while (std::getline(playerFile, line)) {
        std::vector<float> pixels = parseImageLine(line, inputSize);
        if (!pixels.empty()) {
            data.inputs.push_back(pixels);
            data.targets.push_back({1.0f}); // Target for "player" is 1.0
        }
    }
    playerFile.close();
    std::cout << "Loaded " << data.inputs.size() << " 'player' samples." << std::endl;

    size_t initial_count = data.inputs.size();

    std::ifstream notPlayerFile(notPlayerPath);
    if (!notPlayerFile.is_open()) {
        throw std::runtime_error("Error: Could not open not-player dataset file: " + notPlayerPath);
    }

    while (std::getline(notPlayerFile, line)) {
        std::vector<float> pixels = parseImageLine(line, inputSize);
        if (!pixels.empty()) {
            data.inputs.push_back(pixels);
            data.targets.push_back({0.0f});
        }
    }
    notPlayerFile.close();
    std::cout << "Loaded " << (data.inputs.size() - initial_count) << " 'not-player' samples." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "Finished loading " << data.inputs.size() << " total samples in "
              << duration.count() << "s." << std::endl;

    return data;
}

}
