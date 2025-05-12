#include "Layer.h"
#include <cmath>
#include <stdexcept>
#include <cstdlib>

Layer::Layer(int inputSize, int outputSize, std::string _activationName)
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

Matrix Layer::forward(Matrix& input) {
    Matrix z = weights.multiply(input);   
    z = z.add(biases);                    
    return activate(z);                   
}

Matrix Layer::activate(Matrix& z) const {
    if (activationName == "relu") {
        return z.applyFunction(relu);  
    }
    if (activationName == "sigmoid") {
        return z.applyFunction(sigmoid);  
    }
    throw std::invalid_argument("Unsupported activation function.");
}

double Layer::sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double Layer::sigmoidPrime(double x) {
    double s = sigmoid(x);
    return s * (1 - s);  
}

double Layer::relu(double x) {
    return x > 0 ? x : 0;  
}

double Layer::reluPrime(double x) {
    return x <= 0 ? 0 : 1;  
}