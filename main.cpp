#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> 
#include <cstdlib>   
#include <ctime>     
#include <iomanip>   
#include <numeric>   
#include <algorithm> 
#include <random>    

#include "Matrix.h"
#include "Network.h"
#include "MNISTLoader.h" // Include the MNIST loader

// Helper to get a prediction for a single image (returns predicted digit)
int get_prediction_digit(Network& net, Matrix& image_input) {
    Matrix prediction_vector = net.predict(image_input);
    int max_idx = 0;
    double max_val = -1.0;
    for (int i = 0; i < prediction_vector.getRow(); ++i) {
        if (prediction_vector.getEntry(i, 0) > max_val) {
            max_val = prediction_vector.getEntry(i, 0);
            max_idx = i;
        }
    }
    return max_idx;
}


int main() {
    // --- Configuration ---
    bool use_fixed_seed = true; 
    unsigned int seed_value = 42; 

    if (use_fixed_seed) {
        srand(seed_value);
        std::cout << "Using fixed random seed: " << seed_value << std::endl;
    } else {
        unsigned int dynamic_seed = static_cast<unsigned int>(time(0));
        srand(dynamic_seed);
        std::cout << "Using dynamic random seed: " << dynamic_seed << std::endl;
    }
    std::default_random_engine rng(use_fixed_seed ? seed_value : static_cast<unsigned int>(time(0)));

    // --- Load MNIST Data ---
    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";

    // Load a small subset for initial testing to speed things up
    // For full training, set max_items to 0 or a larger number (e.g., 60000 for train, 10000 for test)
    // int items_to_load_train = 1000; // Small subset for quick test
    // int items_to_load_test = 200;   // Small subset for quick test
    int items_to_load_train = 60000; // Full training set
    int items_to_load_test = 10000;  // Full test set


    MNISTDataset training_data;
    MNISTDataset test_data;

    try {
        std::cout << "Loading Training Data..." << std::endl;
        training_data = MNISTLoader::load(train_images_path, train_labels_path, items_to_load_train);
        std::cout << "Loading Test Data..." << std::endl;
        test_data = MNISTLoader::load(test_images_path, test_labels_path, items_to_load_test);
    } catch (const std::exception& e) {
        std::cerr << "Error loading MNIST data: " << e.what() << std::endl;
        return 1;
    }

    // --- Network Configuration for MNIST ---
    // Input: 28x28 = 784 pixels
    // Output: 10 digits (0-9), one-hot encoded
    // Hidden layer(s): e.g., one hidden layer with 100 neurons
    std::vector<int> layer_sizes = {784, 100, 10}; // Input -> Hidden -> Output
    std::vector<std::string> activations = {"relu", "sigmoid"}; // ReLU for hidden, Sigmoid for output

    double learning_rate = 0.1;
    int epochs = 10; // Start with a few epochs
    int batch_size = 32; 

    try {
        Network mnist_net(layer_sizes, activations);

        std::cout << "\n--- Training Started (MNIST) ---" << std::endl;
        std::cout << "Network: Input(" << layer_sizes[0] << ")";
        for(size_t i=0; i < activations.size(); ++i) {
            std::cout << " -> " << activations[i] << "(" << layer_sizes[i+1] << ")";
        }
        std::cout << std::endl;
        std::cout << "Learning Rate: " << learning_rate 
                  << ", Epochs: " << epochs 
                  << ", Batch Size: " << batch_size << std::endl;
        std::cout << "Training on " << training_data.number_of_items << " samples." << std::endl;


        std::vector<size_t> training_indices(training_data.number_of_items);
        std::iota(training_indices.begin(), training_indices.end(), 0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(training_indices.begin(), training_indices.end(), rng); // Shuffle data
            
            double epoch_total_loss = 0.0;
            int num_batches = 0;

            for (size_t i = 0; i < training_data.number_of_items; i += batch_size) {
                std::vector<Matrix> batch_inputs;
                std::vector<Matrix> batch_targets;
                
                size_t current_batch_end = std::min(i + batch_size, static_cast<size_t>(training_data.number_of_items));
                for (size_t j = i; j < current_batch_end; ++j) {
                    batch_inputs.push_back(training_data.images[training_indices[j]]);
                    batch_targets.push_back(training_data.labels[training_indices[j]]);
                }

                if (batch_inputs.empty()) continue;

                double batch_loss = mnist_net.train_on_batch(batch_inputs, batch_targets, learning_rate);
                epoch_total_loss += batch_loss * batch_inputs.size(); // Weighted by actual batch size
                num_batches++;

                // Optional: Print progress within an epoch
                // if (num_batches % 10 == 0) {
                //     std::cout << "Epoch " << epoch << ", Batch " << num_batches << ", Avg Batch Loss: " << batch_loss << std::endl;
                // }
            }

            double average_epoch_loss = epoch_total_loss / training_data.number_of_items;
            std::cout << "Epoch " << std::setw(3) << epoch << "/" << epochs - 1
                      << ", Average Training Loss: " << std::fixed << std::setprecision(8)
                      << average_epoch_loss << std::endl;

            // Optional: Evaluate on a subset of test data periodically
            if ((epoch % 2 == 0 || epoch == epochs -1) && !test_data.images.empty()) {
                int correct_predictions = 0;
                for(size_t k=0; k < test_data.images.size(); ++k) {
                    int predicted_digit = get_prediction_digit(mnist_net, test_data.images[k]);
                    int actual_digit = 0; // Find actual digit from one-hot vector
                    for(int l=0; l<test_data.labels[k].getRow(); ++l) {
                        if(test_data.labels[k].getEntry(l,0) == 1.0) {
                            actual_digit = l;
                            break;
                        }
                    }
                    if (predicted_digit == actual_digit) {
                        correct_predictions++;
                    }
                }
                double accuracy = static_cast<double>(correct_predictions) / test_data.images.size();
                std::cout << "  Test Accuracy after Epoch " << epoch << ": " 
                          << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" 
                          << " (" << correct_predictions << "/" << test_data.images.size() << ")" << std::endl;
            }
        }
        std::cout << "--- Training Finished ---" << std::endl << std::endl;

        // Final evaluation (optional, more detailed)
        std::cout << "--- Final Test Set Evaluation ---" << std::endl;
        if (!test_data.images.empty()) {
            int correct_predictions = 0;
            for(size_t k=0; k < test_data.images.size(); ++k) {
                int predicted_digit = get_prediction_digit(mnist_net, test_data.images[k]);
                int actual_digit = 0;
                for(int l=0; l<test_data.labels[k].getRow(); ++l) {
                    if(test_data.labels[k].getEntry(l,0) == 1.0) {
                        actual_digit = l;
                        break;
                    }
                }
                if (predicted_digit == actual_digit) {
                    correct_predictions++;
                }
                if (k < 10) { // Print first few test predictions
                    std::cout << "Test Sample " << k << ": Predicted=" << predicted_digit << ", Actual=" << actual_digit << std::endl;
                }
            }
            double accuracy = static_cast<double>(correct_predictions) / test_data.images.size();
            std::cout << "Final Test Accuracy: " << accuracy * 100.0 << "%" 
                      << " (" << correct_predictions << "/" << test_data.images.size() << ")" << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during network training/evaluation: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
