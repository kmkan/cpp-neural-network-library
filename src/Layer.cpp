#include "Layer.h"
#include <cmath>      
#include <cstdlib>    
#include <iostream>   
#include <stdexcept>
#include <utility>    

double randomDouble_for_layer_reverted(double min, double max) { 
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}

Layer::Layer(int inputSize, int outputSize, std::string _activationName)
    : weights(outputSize, inputSize), 
      biases(outputSize, 1),          
      activationName(std::move(_activationName)),
      last_input(), 
      last_z(),    
      grad_weights(outputSize, inputSize, 0.0, false), 
      grad_biases(outputSize, 1, 0.0, false),          
      delta_weights(outputSize, inputSize, 0.0, false), 
      delta_biases(outputSize, 1, 0.0, false)           
{
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Layer input and output sizes must be positive.");
    }

    double limit = std::sqrt(6.0 / (static_cast<double>(inputSize) + static_cast<double>(outputSize)));
    for (int i = 0; i < weights.getRow(); ++i) {
        for (int j = 0; j < weights.getCol(); ++j) {
            weights.setEntry(i, j, randomDouble_for_layer_reverted(-limit, limit));
        }
    }

    double bias_init_val = 0.0;
    if (this->activationName == "relu") {
        bias_init_val = 0.01;
    }
    for (int i = 0; i < biases.getRow(); ++i) {
        for (int j = 0; j < biases.getCol(); ++j) {
            biases.setEntry(i, j, bias_init_val); 
        }
    }
}

void Layer::zero_deltas() {
    this->delta_weights = Matrix(weights.getRow(), weights.getCol(), 0.0, false);
    this->delta_biases = Matrix(biases.getRow(), biases.getCol(), 0.0, false);
}

void Layer::accumulate_gradients() {
    this->delta_weights = this->delta_weights.add(this->grad_weights);
    this->delta_biases = this->delta_biases.add(this->grad_biases);
}

void Layer::update_parameters_from_deltas(double learning_rate, int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive for updating parameters.");
    }
    double scale = learning_rate / static_cast<double>(batch_size);

    Matrix scaled_delta_weights = this->delta_weights.multiplyScalar(scale); 
    this->weights = this->weights.subtract(scaled_delta_weights); 

    Matrix scaled_delta_biases = this->delta_biases.multiplyScalar(scale);
    this->biases = this->biases.subtract(scaled_delta_biases); 
}


Matrix Layer::forward(Matrix& input) {
    this->last_input = input; 

    Matrix weighted = weights.multiply(input); 
    Matrix z = weighted.add(biases);       
    
    this->last_z = z; 

    Matrix activated = activate(z); 
    
    return activated; 
}

Matrix Layer::activate(Matrix& z_host) const {
    if (activationName == "relu") return z_host.applyFunction(Layer::relu);
    if (activationName == "sigmoid") return z_host.applyFunction(Layer::sigmoid);
    throw std::invalid_argument("Unsupported activation function: " + activationName);
}

Matrix Layer::activatePrime(Matrix& z_values_host) const {
    if (activationName == "relu") return z_values_host.applyFunction(Layer::reluPrime);
    if (activationName == "sigmoid") return z_values_host.applyFunction(Layer::sigmoidPrime);
    throw std::invalid_argument("Unsupported activation function for derivative: " + activationName);
}

Matrix Layer::backward(const Matrix& d_cost_d_activation_from_next_layer) {
    Matrix activation_grad = activatePrime(this->last_z); 
    Matrix d_z = d_cost_d_activation_from_next_layer.multiplyElements(activation_grad); 

    Matrix last_input_T = this->last_input.transpose(); 
    this->grad_weights = d_z.multiply(last_input_T); 

    this->grad_biases = d_z; 

    Matrix weights_T = this->weights.transpose(); 
    Matrix d_activation_prev = weights_T.multiply(d_z); 
    
    return d_activation_prev; 
}

double Layer::sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double Layer::sigmoidPrime(double x) { double s = Layer::sigmoid(x); return s * (1.0 - s); }
double Layer::relu(double x) { return x > 0 ? x : 0.0; }
double Layer::reluPrime(double x) { return x <= 0 ? 0.0 : 1.0; }
void Layer::printWeights() const {
    std::cout << "Layer Weights (" << weights.getRow() << "x" << weights.getCol() << "):" << std::endl;
    weights.display();
    std::cout << "Layer Biases (" << biases.getRow() << "x" << biases.getCol() << "):" << std::endl;
    biases.display();
}