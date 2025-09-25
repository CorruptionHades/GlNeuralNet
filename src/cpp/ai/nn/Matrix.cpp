//
// Created by CorruptionHades on 19/09/2025.
//

#include "Matrix.h"
#include <random>
#include <iomanip>

Matrix::Matrix(const int rows, const int cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, 0.0f);
}

Matrix Matrix::random(const int rows, const int cols) {
    Matrix m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(-1.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        m.data[i] = dis(gen);
    }
    return m;
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << data[i * cols + j] << "  ";
        }
        std::cout << std::endl;
    }
}
