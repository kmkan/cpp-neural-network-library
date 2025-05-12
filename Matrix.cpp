#include "Matrix.h"
#include <stdexcept>
#include <iostream>
#include <cstdio>

Matrix::Matrix(int _row, int _col) {
    if (_row <= 0 || _col <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.\n");
    }
    row = _row;
    col = _col;

    matrix = new double[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i * col + j] = 0;
        }
    }
}

Matrix::Matrix(int _row, int _col, double val) {
    if (_row <= 0 || _col <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.\n");
    }
    row = _row;
    col = _col;

    matrix = new double[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            matrix[i * col + j] = val;
        }
    }
}

Matrix::~Matrix() {
    delete[] matrix;
}

int Matrix::getRow() const {
    return row;
}

int Matrix::getCol() const {
    return col;
}

double Matrix::getEntry(int _row, int _col) const {
    if (!isValidIndex(_row, _col)) {
        throw std::invalid_argument("Matrix dimensions must be positive.\n");
    }
    return matrix[_row * col + _col];
}

void Matrix::setEntry(int _row, int _col, double entry) {
    if (!isValidIndex(_row, _col)) {
        throw std::invalid_argument("Matrix dimensions must be positive.\n");
    }
    matrix[_row * col + _col] = entry;
}

void Matrix::display() const {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", matrix[i * col + j]);
        }
        printf("\n");
    }
}

Matrix Matrix::applyFunction(double (*f)(double)) {
    Matrix result(row, col);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result.setEntry(i, j, f(getEntry(i, j)));
        }
    }
    return result;
}

Matrix Matrix::add(Matrix & m) const {
    if (col != m.getCol() || row != m.getRow()) {
        throw std::invalid_argument("Dimensions of matrices not compatible.\n");
    }

    Matrix sum(row, col);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sum.matrix[i * col + j] = matrix[i * col + j] + m.matrix[i * col + j];
        }
    }

    return sum;
}

Matrix Matrix::multiply(Matrix& m) const {
    if (col != m.getRow()) {
        throw std::invalid_argument("Dimensions of matrices not compatible.\n");
    }

    Matrix product(row, m.getCol());
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < m.getCol(); j++) { 
            double vectorProd = 0;
            for (int k = 0; k < col; k++) {
                vectorProd += matrix[i * col + k] * m.matrix[k * m.getCol() + j];
            }
            product.matrix[i * m.getCol() + j] = vectorProd;
        }
    }

    return product;
}

Matrix Matrix::operator+(Matrix& m) const {
    return this->add(m);  
}

Matrix Matrix::operator*(Matrix& m) const {
    return this->multiply(m);  
}

bool Matrix::isValidIndex(int _row, int _col) const {
    return _row >= 0 && _row < row && _col >= 0 && _col < col;
}