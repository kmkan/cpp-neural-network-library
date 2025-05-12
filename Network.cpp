#include "Network.h"
#include <stdexcept>

Network::Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations) {
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Mismatch in layer sizes and activations");
    }

    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i + 1], activations[i]);
    }
}

Matrix Network::predict(const Matrix& input) {
    Matrix output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}