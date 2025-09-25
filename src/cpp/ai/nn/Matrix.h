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

    Matrix(int r, int c);

    static Matrix random(int r, int c);
    void print() const;
};

#endif
