#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <string>

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
    int getRow() const {
        return row;
    }
    int getCol() const {
        return col;
    }
    double getEntry(int _row, int _col) const {
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
    Matrix applyFunction(double (*f)(double)) {
        Matrix result(row, col);
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                result.setEntry(i, j, f(getEntry(i, j)));
            }
        }
        return result;
    }
    Matrix add (Matrix & m) const {
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
    Matrix multiply(Matrix& m) const {
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
    Matrix operator+(Matrix& m) const {
    return this->add(m);  
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

class Layer {
public:
    Layer(int inputSize, int outputSize, std::string _activationName)
    : activationName(_activationName),
      weights(inputSize, outputSize),
      biases(outputSize, 1) 
    {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.setEntry(i, j, ((double)rand() / RAND_MAX) * 2.0 - 1.0);
            }
        }

        for (int i = 0; i < outputSize; i++) {
            biases.setEntry(i, 0, 0);
        }
    }

    Matrix forward(Matrix& input) {
        Matrix z = weights.multiply(input);   
        z = z.add(biases);                    
        return activate(z);        
    }

private:
    Matrix weights;
    Matrix biases;
    std::string activationName;

    Matrix Layer::activate(Matrix& z) const {
        if (activationName == "relu") return z.applyFunction(relu);
        if (activationName == "sigmoid") return z.applyFunction(sigmoid);
        throw std::invalid_argument("Unsupported activation function.");
    }
};

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double sigmoidPrime(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double reluPrime(double x) {
    return x <= 0 ? 0 : 1;
}

int main() {
    return 0;
}