#include "Layer.h"
#include <cmath>      
#include <cstdlib>    
#include <iostream>   
#include <stdexcept>  

double randomDouble(double min, double max) {
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}

Layer::Layer(int inputSize, int outputSize, std::string _activationName)
    : weights(outputSize, inputSize),         
      biases(outputSize, 1),                  
      activationName(std::move(_activationName)) 
{
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Layer input and output sizes must be positive.");
    }

    for (int i = 0; i < weights.getRow(); ++i) {
        for (int j = 0; j < weights.getCol(); ++j) {
            weights.setEntry(i, j, randomDouble(-0.5, 0.5));
        }
    }

    for (int i = 0; i < biases.getRow(); ++i) {
        for (int j = 0; j < biases.getCol(); ++j) {
            biases.setEntry(i, j, randomDouble(-0.1, 0.1));
        }
    }
}

Matrix Layer::forward(Matrix& input) {
    Matrix weighted = weights.multiply(input);
    std::cout << "Weighted (W·x) before adding bias:\n";
    weighted.display();

    Matrix z = weighted.add(biases);
    std::cout << "Z (W·x + b) after adding bias:\n";
    z.display();

    Matrix activated = activate(z); 
    std::cout << "Activated output (A = g(Z)):\n";
    activated.display();
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