#include "Matrix.h"
#include <stdexcept>
#include <vector>
#include <iostream> 
#include <iomanip>  

Matrix::Matrix(int _row, int _col)
    : row(_row), col(_col) {
    if (_row <= 0 || _col <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    matrix_data.assign(static_cast<size_t>(_row) * _col, 0.0);
}

Matrix::Matrix(int _row, int _col, double val)
    : row(_row), col(_col) {
    if (_row <= 0 || _col <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive.");
    }
    matrix_data.assign(static_cast<size_t>(_row) * _col, val);
}

int Matrix::getRow() const {
    return row;
}

int Matrix::getCol() const {
    return col;
}

bool Matrix::isValidIndex(int _r, int _c) const {
    return _r >= 0 && _r < row && _c >= 0 && _c < col;
}

double Matrix::getEntry(int _r, int _c) const {
    if (!isValidIndex(_r, _c)) {
        throw std::out_of_range("Matrix::getEntry: Index (" + std::to_string(_r) + "," + std::to_string(_c) + ") out of bounds for " + std::to_string(row) + "x" + std::to_string(col) + " matrix.");
    }
    return matrix_data[static_cast<size_t>(_r) * col + _c];
}

void Matrix::setEntry(int _r, int _c, double entry) {
    if (!isValidIndex(_r, _c)) {
        throw std::out_of_range("Matrix::setEntry: Index (" + std::to_string(_r) + "," + std::to_string(_c) + ") out of bounds for " + std::to_string(row) + "x" + std::to_string(col) + " matrix.");
    }
    matrix_data[static_cast<size_t>(_r) * col + _c] = entry;
}

void Matrix::display() const {
    if (matrix_data.empty() && (row > 0 || col > 0) ) {
        std::cout << "Matrix is uninitialized or has zero dimensions but non-zero row/col members." << std::endl;
        return;
    }
    if (row == 0 || col == 0) {
        std::cout << "Matrix has zero dimension." << std::endl;
        return;
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << std::fixed << std::setprecision(4) << matrix_data[static_cast<size_t>(i) * col + j] << " "; // Increased precision for debugging
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::applyFunction(double (*f)(double x)) {
    Matrix result(row, col);
    for (size_t i = 0; i < matrix_data.size(); ++i) {
        result.matrix_data[i] = f(matrix_data[i]);
    }
    return result;
}

Matrix Matrix::add(const Matrix& m) const {
    if (col != m.getCol() || row != m.getRow()) {
        throw std::invalid_argument("Matrix::add: Dimensions of matrices not compatible. LHS: " +
                                    std::to_string(row) + "x" + std::to_string(col) + ", RHS: " +
                                    std::to_string(m.getRow()) + "x" + std::to_string(m.getCol()));
    }

    Matrix sum(row, col);
    for (size_t i = 0; i < matrix_data.size(); ++i) {
        sum.matrix_data[i] = matrix_data[i] + m.matrix_data[i];
    }
    return sum;
}

Matrix Matrix::subtract(const Matrix& m) const {
    if (row != m.getRow() || col != m.getCol()) {
        throw std::invalid_argument("Matrix::subtract: Dimensions of matrices not compatible. LHS: " +
                                    std::to_string(row) + "x" + std::to_string(col) + ", RHS: " +
                                    std::to_string(m.getRow()) + "x" + std::to_string(m.getCol()));
    }
    Matrix result(row, col);
    for (size_t i = 0; i < matrix_data.size(); ++i) {
        result.matrix_data[i] = matrix_data[i] - m.matrix_data[i];
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& m) const {
    if (col != m.getRow()) {
       throw std::invalid_argument("Matrix::multiply: Dimensions of matrices not compatible. LHS: " +
                                    std::to_string(row) + "x" + std::to_string(col) + ", RHS: " +
                                    std::to_string(m.getRow()) + "x" + std::to_string(m.getCol()));
    }

    Matrix product(row, m.getCol());
    for (int i = 0; i < product.row; i++) {
        for (int j = 0; j < product.col; j++) {
            double vectorProd = 0;
            for (int k = 0; k < col; k++) {
                vectorProd += matrix_data[static_cast<size_t>(i) * col + k] * m.matrix_data[static_cast<size_t>(k) * m.col + j];
            }
            product.matrix_data[static_cast<size_t>(i) * product.col + j] = vectorProd;
        }
    }
    return product;
}

Matrix Matrix::multiplyElements(const Matrix& m) const { // Hadamard Product
    if (row != m.getRow() || col != m.getCol()) {
        throw std::invalid_argument("Matrix::multiplyElements: Dimensions of matrices not compatible. LHS: " +
                                    std::to_string(row) + "x" + std::to_string(col) + ", RHS: " +
                                    std::to_string(m.getRow()) + "x" + std::to_string(m.getCol()));
    }
    Matrix result(row, col);
    for (size_t i = 0; i < matrix_data.size(); ++i) {
        result.matrix_data[i] = matrix_data[i] * m.matrix_data[i];
    }
    return result;
}

Matrix Matrix::multiplyScalar(double scalar) const {
    Matrix result(row, col);
    for (size_t i = 0; i < matrix_data.size(); ++i) {
        result.matrix_data[i] = matrix_data[i] * scalar;
    }
    return result;
}


Matrix Matrix::transpose() const {
    Matrix result(col, row);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result.setEntry(j, i, getEntry(i, j));
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& m) const {
    return this->add(m);
}

Matrix Matrix::operator-(const Matrix& m) const {
    return this->subtract(m);
}

Matrix Matrix::operator*(const Matrix& m) const {
    return this->multiply(m);
}