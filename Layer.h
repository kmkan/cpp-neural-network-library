#ifndef LAYER_H
#define LAYER_H

#include <string>
#include "Matrix.h" 

class Layer {
public:
    // Constructor
    Layer(int inputSize, int outputSize, std::string _activationName);

    // Forward pass through the layer
    Matrix forward(Matrix& input);

private:
    Matrix weights;         // Weights for the layer
    Matrix biases;          // Biases for the layer
    std::string activationName;  // Activation function name

    // Activation function and its derivative
    Matrix activate(Matrix& z) const;

    // Helper activation functions
    static double sigmoid(double x);
    static double sigmoidPrime(double x);
    static double relu(double x);
    static double reluPrime(double x);
};

#endif 