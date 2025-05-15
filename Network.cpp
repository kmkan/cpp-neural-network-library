#include "Network.h"
#include <stdexcept>
#include <iostream> 
#include <vector>   
#include <string>   
#include <iomanip> 

Network::Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations) {
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Mismatch in layer sizes (" + std::to_string(layerSizes.size()) +
                                    ") and activations (" + std::to_string(activations.size()) +
                                    "). Number of activations should be one less than number of layer sizes.");
    }
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and an output size (2 layer sizes).");
    }

    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        layers.emplace_back(layerSizes[i], layerSizes[i + 1], activations[i]);
    }
}

Matrix Network::predict(Matrix& input) {
    Matrix current_output = input;
    for (auto& layer : layers) {
        current_output = layer.forward(current_output);
    }
    return current_output;
}

double Network::meanSquaredError(const Matrix& predicted, const Matrix& actual) const {
    if (predicted.getRow() != actual.getRow() || predicted.getCol() != actual.getCol()) {
        throw std::invalid_argument("MSE: Predicted and actual matrices dimensions mismatch. Predicted: " +
                                    std::to_string(predicted.getRow()) + "x" + std::to_string(predicted.getCol()) + ", Actual: " +
                                    std::to_string(actual.getRow()) + "x" + std::to_string(actual.getCol()));
    }
    Matrix diff = predicted.subtract(actual);
    double sum_sq_error = 0.0;
    int num_elements = 0;
    for (int i = 0; i < diff.getRow(); ++i) {
        for (int j = 0; j < diff.getCol(); ++j) {
            double val = diff.getEntry(i, j);
            sum_sq_error += val * val;
            num_elements++;
        }
    }
    if (num_elements == 0) return 0.0; 
    return sum_sq_error / static_cast<double>(num_elements); 
}

Matrix Network::meanSquaredErrorDerivative(const Matrix& predicted, const Matrix& actual) const {
    if (predicted.getRow() != actual.getRow() || predicted.getCol() != actual.getCol()) {
        throw std::invalid_argument("MSE Derivative: Predicted and actual matrices dimensions mismatch. Predicted: " +
                                    std::to_string(predicted.getRow()) + "x" + std::to_string(predicted.getCol()) + ", Actual: " +
                                    std::to_string(actual.getRow()) + "x" + std::to_string(actual.getCol()));
    }
    
    int num_elements = predicted.getRow() * predicted.getCol();
    if (num_elements == 0) {
         return Matrix(predicted.getRow(), predicted.getCol(), 0.0); 
    }
    double scale = 2.0 / static_cast<double>(num_elements);
    return predicted.subtract(actual).multiplyScalar(scale);
}

void Network::backpropagate(const Matrix& initial_error_gradient) {
    Matrix current_error_gradient = initial_error_gradient;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        current_error_gradient = layers[static_cast<size_t>(i)].backward(current_error_gradient);
    }
}

void Network::updateParameters(double learningRate) {
    for (size_t i = 0; i < layers.size(); ++i) {
        Matrix weight_gradient_scaled = layers[i].grad_weights.multiplyScalar(learningRate);
        layers[i].weights = layers[i].weights.subtract(weight_gradient_scaled);

        Matrix bias_gradient_scaled = layers[i].grad_biases.multiplyScalar(learningRate);
        layers[i].biases = layers[i].biases.subtract(bias_gradient_scaled);
    }
}

double Network::train_on_sample(Matrix& input, Matrix& target, double learningRate) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot train an empty network.");
    }
    if (layers.back().weights.getRow() != target.getRow() || target.getCol() != 1) {
         throw std::invalid_argument("Target matrix dimensions (" + std::to_string(target.getRow()) + "x" + std::to_string(target.getCol()) +
                                    ") incompatible with network output size (" + std::to_string(layers.back().weights.getRow()) + "x1).");
    }
     if (layers.front().weights.getCol() != input.getRow() || input.getCol() != 1) {
         throw std::invalid_argument("Input matrix dimensions (" + std::to_string(input.getRow()) + "x" + std::to_string(input.getCol()) +
                                    ") incompatible with network input size (" + std::to_string(layers.front().weights.getCol()) + "x1).");
    }

    Matrix predicted_output = this->predict(input);

    double loss = this->meanSquaredError(predicted_output, target);

    Matrix error_gradient = this->meanSquaredErrorDerivative(predicted_output, target);

    this->backpropagate(error_gradient);

    this->updateParameters(learningRate);

    return loss; 
}