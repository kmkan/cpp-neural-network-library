#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include "Layer.h"
#include "Matrix.h"

class Network {
public:
    Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations);

    // Perform a forward pass and return the output
    Matrix predict(Matrix& input);

    // Loss function and its derivative
    double meanSquaredError(const Matrix& predicted, const Matrix& actual) const;
    Matrix meanSquaredErrorDerivative(const Matrix& predicted, const Matrix& actual) const;

    // Backpropagation
    void backpropagate(const Matrix& output_error_gradient); // Takes dL/dOutput_Network
    
    // Parameter update
    void updateParameters(double learningRate);

    // Train the network on a single input-target sample and return the loss for that sample
    double train_on_sample(Matrix& input, Matrix& target, double learningRate);

private:
    std::vector<Layer> layers; // Layers in the neural network
};

#endif  