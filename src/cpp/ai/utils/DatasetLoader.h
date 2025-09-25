//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <vector>
#include <string>

struct TrainingData {
    std::vector<std::vector<float> > inputs;
    std::vector<std::vector<float> > targets;
};

namespace DatasetLoader {
    TrainingData load(const std::string &playerPath,
                      const std::string &notPlayerPath,
                      size_t inputSize);
}

#endif //DATASETLOADER_H
