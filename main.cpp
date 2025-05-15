#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> // For std::exception
#include <cstdlib>   // For srand, rand
#include <ctime>     // For time
#include <iomanip>   // For std::fixed, std::setprecision
#include <numeric>   // For std::iota for shuffling
#include <algorithm> // For std::shuffle
#include <random>    // For std::default_random_engine for shuffling

#include "Matrix.h"
#include "Layer.h"
#include "Network.h"

// Helper function to print predictions
void print_predictions(Network& net, const std::vector<Matrix>& inputs, const std::vector<Matrix>& targets) {
    std::cout << std::fixed << std::setprecision(5);
    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix input_sample = inputs[i]; 
        Matrix predicted = net.predict(input_sample);
        std::cout << "Input: [";
        for (int r = 0; r < inputs[i].getRow(); ++r) {
            std::cout << inputs[i].getEntry(r, 0) << (r == inputs[i].getRow() - 1 ? "" : ", ");
        }
        std::cout << "] -> Predicted: [";
        for (int r = 0; r < predicted.getRow(); ++r) {
            std::cout << predicted.getEntry(r, 0) << (r == predicted.getRow() - 1 ? "" : ", ");
        }
        std::cout << "], Target: [";
        for (int r = 0; r < targets[i].getRow(); ++r) {
            std::cout << targets[i].getEntry(r, 0) << (r == targets[i].getRow() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
}


int main() {
    bool use_fixed_seed = false; 
    unsigned int seed_value = 42; 

    if (use_fixed_seed) {
        srand(seed_value);
        std::cout << "Using fixed random seed: " << seed_value << std::endl;
    } else {
        unsigned int dynamic_seed = static_cast<unsigned int>(time(0));
        srand(dynamic_seed);
        std::cout << "Using dynamic random seed: " << dynamic_seed << std::endl;
    }

    // Test Case: XOR Problem 
    std::vector<Matrix> train_inputs;
    std::vector<Matrix> train_targets;

    Matrix x1(2, 1); x1.setEntry(0, 0, 0.0); x1.setEntry(1, 0, 0.0);
    Matrix y1(1, 1); y1.setEntry(0, 0, 0.0);
    train_inputs.push_back(x1); train_targets.push_back(y1);

    Matrix x2(2, 1); x2.setEntry(0, 0, 0.0); x2.setEntry(1, 0, 1.0);
    Matrix y2(1, 1); y2.setEntry(0, 0, 1.0);
    train_inputs.push_back(x2); train_targets.push_back(y2);

    Matrix x3(2, 1); x3.setEntry(0, 0, 1.0); x3.setEntry(1, 0, 0.0);
    Matrix y3(1, 1); y3.setEntry(0, 0, 1.0);
    train_inputs.push_back(x3); train_targets.push_back(y3);

    Matrix x4(2, 1); x4.setEntry(0, 0, 1.0); x4.setEntry(1, 0, 1.0);
    Matrix y4(1, 1); y4.setEntry(0, 0, 0.0);
    train_inputs.push_back(x4); train_targets.push_back(y4);

    std::vector<int> layer_sizes = {2, 3, 1}; 
    std::vector<std::string> activations = {"relu", "sigmoid"}; 
    
    double learning_rate = 0.1;
    int epochs = 10000; 

    std::vector<size_t> indices(train_inputs.size());
    std::iota(indices.begin(), indices.end(), 0); 
    std::default_random_engine rng(use_fixed_seed ? seed_value : static_cast<unsigned int>(time(0)));


    try {
        Network xor_net(layer_sizes, activations);

        std::cout << "--- Predictions Before Training ---" << std::endl;
        print_predictions(xor_net, train_inputs, train_targets);
        std::cout << std::endl;

        std::cout << "--- Training Started ---" << std::endl;
        std::cout << "Network: Input(" << layer_sizes[0] << ")";
        for(size_t i=0; i < activations.size(); ++i) {
            std::cout << " -> " << activations[i] << "(" << layer_sizes[i+1] << ")";
        }
        std::cout << std::endl;
        std::cout << "Learning Rate: " << learning_rate << ", Epochs: " << epochs << std::endl;


        for (int epoch = 0; epoch < epochs; ++epoch) {
            double current_epoch_total_loss = 0.0;

            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t i = 0; i < train_inputs.size(); ++i) {
                size_t current_idx = indices[i];
                double sample_loss = xor_net.train_on_sample(train_inputs[current_idx], train_targets[current_idx], learning_rate);
                current_epoch_total_loss += sample_loss;
            }

            if ((epoch % (epochs / 20 == 0 ? 1 : epochs / 20) == 0) || epoch == epochs - 1) {
                std::cout << "Epoch " << std::setw(5) << epoch << "/" << epochs - 1
                          << ", Average Loss: " << std::fixed << std::setprecision(8)
                          << (current_epoch_total_loss / train_inputs.size()) << std::endl;
            }
        }
        std::cout << "--- Training Finished ---" << std::endl << std::endl;

        std::cout << "--- Predictions After Training ---" << std::endl;
        print_predictions(xor_net, train_inputs, train_targets);

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}