#include "Layer.h"
#include <cmath>
#include <cstdlib>    
#include <iostream>
#include <stdexcept>
#include <utility>    

double randomDouble(double min, double max) {
    if (min > max) {
    }
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}

Layer::Layer(int inputSize, int outputSize, std::string _activationName)
    : weights(outputSize, inputSize),
      biases(outputSize, 1),
      activationName(std::move(_activationName)),
      last_input(inputSize, 1), 
      last_z(outputSize, 1),    
      grad_weights(outputSize, inputSize),
      grad_biases(outputSize, 1)
{
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Layer input and output sizes must be positive.");
    }

    double limit = std::sqrt(6.0 / (static_cast<double>(inputSize) + static_cast<double>(outputSize)));

    for (int i = 0; i < weights.getRow(); ++i) {
        for (int j = 0; j < weights.getCol(); ++j) {
            weights.setEntry(i, j, randomDouble(-limit, limit));
        }
    }

    for (int i = 0; i < biases.getRow(); ++i) {
        for (int j = 0; j < biases.getCol(); ++j) {
            biases.setEntry(i, j, randomDouble(-0.1, 0.1)); 
        }
    }
}

Matrix Layer::forward(Matrix& input) {
    if (input.getRow() != last_input.getRow() || input.getCol() != last_input.getCol()) {
        this->last_input = Matrix(input.getRow(), input.getCol()); 
    }
    for(int r=0; r < input.getRow(); ++r) { 
        for(int c=0; c < input.getCol(); ++c) {
            this->last_input.setEntry(r,c, input.getEntry(r,c));
        }
    }

    Matrix weighted = weights.multiply(input);
    Matrix z = weighted.add(biases); 

    if (z.getRow() != last_z.getRow() || z.getCol() != last_z.getCol()) {
        this->last_z = Matrix(z.getRow(), z.getCol()); 
    }
     for(int r=0; r < z.getRow(); ++r) { 
        for(int c=0; c < z.getCol(); ++c) {
            this->last_z.setEntry(r,c, z.getEntry(r,c));
        }
    }

    Matrix activated = activate(z); 
    return activated;
}

Matrix Layer::activate(Matrix& z) const {
    if (activationName == "relu") {
        return z.applyFunction(Layer::relu);
    }
    if (activationName == "sigmoid") {
        return z.applyFunction(Layer::sigmoid);
    }
    throw std::invalid_argument("Unsupported activation function: " + activationName);
}

Matrix Layer::activatePrime(Matrix& z_values) const {
    if (activationName == "relu") {
        return z_values.applyFunction(Layer::reluPrime);
    }
    if (activationName == "sigmoid") {
        return z_values.applyFunction(Layer::sigmoidPrime);
    }
    throw std::invalid_argument("Unsupported activation function for derivative: " + activationName);
}

Matrix Layer::backward(const Matrix& d_cost_d_activation) {
    Matrix activation_grad = activatePrime(this->last_z);
    Matrix d_z = d_cost_d_activation.multiplyElements(activation_grad); 

    Matrix last_input_transposed = this->last_input.transpose();
    this->grad_weights = d_z.multiply(last_input_transposed);
    
    if (d_z.getCol() > 1) { 
        this->grad_biases = Matrix(d_z.getRow(), 1, 0.0); 
        for (int r = 0; r < d_z.getRow(); ++r) {
            double sum_for_row = 0.0;
            for (int c = 0; c < d_z.getCol(); ++c) {
                sum_for_row += d_z.getEntry(r, c);
            }
            this->grad_biases.setEntry(r, 0, sum_for_row / d_z.getCol()); 
        }
    } else { 
        this->grad_biases = d_z; 
    }


    Matrix weights_transposed = this->weights.transpose();
    Matrix d_activation_prev = weights_transposed.multiply(d_z);

    return d_activation_prev;
}

double Layer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Layer::sigmoidPrime(double x) {
    double s = Layer::sigmoid(x);
    return s * (1.0 - s);
}

double Layer::relu(double x) {
    return x > 0 ? x : 0.0;
}

double Layer::reluPrime(double x) {
    return x <= 0 ? 0.0 : 1.0;
}

void Layer::printWeights() const {
    std::cout << "Layer Weights (" << weights.getRow() << "x" << weights.getCol() << "):\n";
    weights.display();
    std::cout << "Layer Biases (" << biases.getRow() << "x" << biases.getCol() << "):\n";
    biases.display();
}