#include "Network.h"
#include <stdexcept>
#include <iostream> 
#include <vector>   
#include <string>   
#include <iomanip> 

Network::Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations) {
    if (activations.size() != layerSizes.size() - 1) {
        throw std::invalid_argument("Mismatch in layer sizes and activations.");
    }
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Network must have at least an input and an output size.");
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
        throw std::invalid_argument("MSE: Predicted and actual matrices dimensions mismatch.");
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
    int num_elements = predicted.getRow() * predicted.getCol();
    if (num_elements == 0) {
         return Matrix(predicted.getRow(), predicted.getCol(), 0.0, false); 
    }
    double scale = 2.0 / static_cast<double>(num_elements); 
    return predicted.subtract(actual).multiplyScalar(scale); 
}

void Network::backpropagate_sample(const Matrix& initial_error_gradient) {
    Matrix current_error_gradient = initial_error_gradient; 

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        current_error_gradient = layers[static_cast<size_t>(i)].backward(current_error_gradient);
    }
}

void Network::zero_all_layer_deltas() {
    for (auto& layer : layers) {
        layer.zero_deltas(); 
    }
}

void Network::accumulate_all_layer_gradients() {
    for (auto& layer : layers) {
        layer.accumulate_gradients(); 
    }
}

void Network::update_all_layer_parameters(double learning_rate, int batch_size) {
    for (auto& layer : layers) {
        layer.update_parameters_from_deltas(learning_rate, batch_size);
    }
}


double Network::train_on_batch(const std::vector<Matrix>& batch_inputs, 
                               const std::vector<Matrix>& batch_targets, 
                               double learning_rate) {
    if (batch_inputs.empty() || batch_targets.empty()) {
        throw std::invalid_argument("Batch inputs or targets cannot be empty.");
    }
    if (batch_inputs.size() != batch_targets.size()) {
        throw std::invalid_argument("Batch inputs and targets size mismatch.");
    }

    int batch_size_val = static_cast<int>(batch_inputs.size());
    double total_batch_loss = 0.0;

    zero_all_layer_deltas();

    for (int i = 0; i < batch_size_val; ++i) {
        Matrix current_input = batch_inputs[i]; 
        Matrix current_target = batch_targets[i]; 
        
        Matrix predicted_output = this->predict(current_input); 
        
        total_batch_loss += this->meanSquaredError(predicted_output, current_target); 

        Matrix error_gradient = this->meanSquaredErrorDerivative(predicted_output, current_target); 

        this->backpropagate_sample(error_gradient); 

        this->accumulate_all_layer_gradients(); 
    }

    this->update_all_layer_parameters(learning_rate, batch_size_val); 

    return total_batch_loss / static_cast<double>(batch_size_val); 
}