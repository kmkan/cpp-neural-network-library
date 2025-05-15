#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include "Layer.h"
#include "Matrix.h"

class Network {
public:
    Network(const std::vector<int>& layerSizes, const std::vector<std::string>& activations);

    Matrix predict(Matrix& input);

    double meanSquaredError(const Matrix& predicted, const Matrix& actual) const;
    Matrix meanSquaredErrorDerivative(const Matrix& predicted, const Matrix& actual) const;

    void backpropagate_sample(const Matrix& output_error_gradient); 
    
    double train_on_batch(const std::vector<Matrix>& batch_inputs, 
                          const std::vector<Matrix>& batch_targets, 
                          double learningRate);
private:
    void zero_all_layer_deltas();
    void accumulate_all_layer_gradients();
    void update_all_layer_parameters(double learning_rate, int batch_size);

    std::vector<Layer> layers; 
};

#endif
