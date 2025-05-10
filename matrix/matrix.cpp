#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

class Matrix {
public:
    Matrix(int _row, int _col) {
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
    Matrix(int _row, int _col, double val) {
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
    ~Matrix() {
        delete[] matrix;
    }
    int getRow() {
        return row;
    }
    int getCol() {
        return col;
    }
    double getEntry(int _row, int _col) {
        if (!isValidIndex(_row, _col)) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        return matrix[_row * col + _col];
    }
    void setEntry(int _row, int _col, double entry) {
        if (!isValidIndex(_row, _col)) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        matrix[_row * col + _col] = entry;
    }
    void display() const {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%.2f ", matrix[i * col + j]);
            }
            printf("\n");
        }
    }
    Matrix multiply(Matrix& m) const {
        if (col != m.getRow()) {
            throw std::invalid_argument("Dimensions of matrices not compatible.\n");
        }

        Matrix product = Matrix(row, m.getCol());
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
    Matrix operator*(Matrix& m) const {
    return this->multiply(m);  
    }
private: 
    int row;
    int col;
    double* matrix;
    bool isValidIndex(int _row, int _col) const {
        return _row >= 0 && _row < row && _col >= 0 && _col < col;
    }
};

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double relu(double x) {
    return x > 0 ? x : 0;
}

int main() {
    return 0;
}