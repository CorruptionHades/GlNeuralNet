//
// Created by CorruptionHades on 19/09/2025.
//

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
public:
    int rows;
    int cols;
    // Stored in a flat vector for easy GPU transfer
    std::vector<float> data;

    Matrix(int rows, int cols);

    static Matrix random(int rows, int cols);
    void print() const;
};

#endif
