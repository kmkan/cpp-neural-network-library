#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include "Layer.h"
#include "Matrix.h" 

class Network {
public:
    // Constructor: Initializes the network with given layer sizes and activation functions
    Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations);

    // Perform a forward pass and return the output
    Matrix predict(const Matrix& input);

private:
    std::vector<Layer> layers; // Layers in the neural network
};

#endif 