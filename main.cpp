#include <iostream>
#include <vector>
#include <ctime>
#include "Matrix.h"
#include "Layer.h"
#include "Network.h"

int main() {
    try {
        // Seed random number generator
        srand(42);  // Fixed seed for deterministic results

        // Define the network architecture: 2 input neurons, 2 hidden, 1 output
        std::vector<int> layerSizes = {2, 2, 1};  // 2 inputs, 2 hidden, 1 output
        std::vector<std::string> activations = {"relu", "sigmoid"};  // ReLU for hidden, Sigmoid for output

        // Create the network
        Network net(layerSizes, activations);

        // Manually set the weights and biases for each layer (for deterministic results)
        // For testing purposes, this is already done in the Layer class constructor

        // Create an input column vector (2x1)
        Matrix input(2, 1);
        input.setEntry(0, 0, 1.0);  // First input
        input.setEntry(1, 0, -1.0); // Second input

        std::cout << "Input:\n";
        input.display();

        // Run the input through the network
        Matrix output = net.predict(input);

        // Display the output
        std::cout << "\nOutput:\n";
        output.display();

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << '\n';
    }

    return 0;
}