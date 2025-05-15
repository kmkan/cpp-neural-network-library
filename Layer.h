#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include <string>
#include <vector>
#include <stdexcept>

class Layer {
public:
    Matrix weights;
    Matrix biases;
    std::string activationName;

    // For backpropagation
    Matrix last_input;      // Input to this layer (activations from previous layer A_prev)
    Matrix last_z;          // Weighted sum + bias (Z = W * A_prev + B) before activation
    Matrix grad_weights;    // dL/dW for this layer
    Matrix grad_biases;     // dL/dB for this layer

    Layer(int inputSize, int outputSize, std::string _activationName);

    Matrix forward(Matrix& input);
    Matrix activate(Matrix& z) const;
    Matrix activatePrime(Matrix& z_values) const; 

    // Returns dL/dA_prev 
    Matrix backward(const Matrix& d_output_error); 

    void printWeights() const;

    static double sigmoid(double x);
    static double sigmoidPrime(double x);
    static double relu(double x);
    static double reluPrime(double x);
};

#endif  