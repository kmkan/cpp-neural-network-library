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
#include "MNISTLoader.h" 

int get_prediction_digit(Network& net, Matrix& image_input_param) {
    Matrix image_input = image_input_param; 
    Matrix prediction_vector = net.predict(image_input);
    
    int max_idx = 0;
    double max_val = -1.0;
    if (prediction_vector.getRow() == 0 || prediction_vector.getCol() == 0) {
        std::cerr << "Warning: get_prediction_digit received an empty prediction vector (" 
                  << prediction_vector.getRow() << "x" << prediction_vector.getCol() 
                  << ")." << std::endl;
        return -1; 
    }

    for (int i = 0; i < prediction_vector.getRow(); ++i) {
        if (prediction_vector.getEntry(i, 0) > max_val) {
            max_val = prediction_vector.getEntry(i, 0);
            max_idx = i;
        }
    }
    return max_idx;
}


int main() {
    bool use_fixed_seed = true; 
    unsigned int seed_value = 123; 

    double learning_rate = 0.01; 
    int epochs = 20; 
    int batch_size = 32; 


    if (use_fixed_seed) {
        srand(seed_value);
        std::cout << "Using fixed random seed: " << seed_value << std::endl;
    } else {
        unsigned int dynamic_seed = static_cast<unsigned int>(time(0));
        srand(dynamic_seed);
        std::cout << "Using dynamic random seed: " << dynamic_seed << std::endl;
    }
    std::default_random_engine rng(use_fixed_seed ? seed_value : static_cast<unsigned int>(time(0)));

    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";

    int items_to_load_train = 60000; 
    int items_to_load_test = 10000;   

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

    std::vector<int> layer_sizes = {784, 100, 10}; 
    std::vector<std::string> activations = {"relu", "sigmoid"}; 

    try {
        Network mnist_net(layer_sizes, activations);

        std::cout << "\n--- Training Started (MNIST CPU-Centric - Full Dataset) ---" << std::endl;
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
            std::shuffle(training_indices.begin(), training_indices.end(), rng); 
            
            double epoch_total_loss = 0.0;
            int num_batches_processed = 0;

            for (size_t i = 0; i < static_cast<size_t>(training_data.number_of_items); i += batch_size) {
                std::vector<Matrix> batch_inputs;  
                std::vector<Matrix> batch_targets; 
                
                size_t current_batch_end = std::min(i + batch_size, static_cast<size_t>(training_data.number_of_items));
                for (size_t j = i; j < current_batch_end; ++j) {
                    batch_inputs.push_back(training_data.images[training_indices[j]]);
                    batch_targets.push_back(training_data.labels[training_indices[j]]);
                }

                if (batch_inputs.empty()) continue;

                double batch_loss = mnist_net.train_on_batch(batch_inputs, batch_targets, learning_rate);
                epoch_total_loss += batch_loss * batch_inputs.size(); 
                num_batches_processed++;
            }

            double average_epoch_loss = (training_data.number_of_items > 0) ? (epoch_total_loss / training_data.number_of_items) : 0.0;
            std::cout << "Epoch " << std::setw(3) << epoch << "/" << epochs - 1
                      << ", Average Training Loss: " << std::fixed << std::setprecision(8)
                      << average_epoch_loss << std::endl;

            if ((epoch % 1 == 0 || epoch == epochs -1) && !test_data.images.empty()) { 
                int correct_predictions = 0;
                for(size_t k=0; k < static_cast<size_t>(test_data.number_of_items); ++k) {
                    int predicted_digit = get_prediction_digit(mnist_net, test_data.images[k]);
                    
                    Matrix actual_label_host = test_data.labels[k]; 

                    int actual_digit = -1; 
                    for(int l=0; l<actual_label_host.getRow(); ++l) {
                        if(actual_label_host.getEntry(l,0) == 1.0) {
                            actual_digit = l;
                            break;
                        }
                    }
                    if (predicted_digit == actual_digit) {
                        correct_predictions++;
                    }
                }
                double accuracy = (test_data.images.size() > 0) ? (static_cast<double>(correct_predictions) / test_data.images.size()) : 0.0;
                std::cout << "  Test Accuracy after Epoch " << epoch << ": " 
                          << std::fixed << std::setprecision(4) << accuracy * 100.0 << "%" 
                          << " (" << correct_predictions << "/" << test_data.images.size() << ")" << std::endl;
            }
        }
        std::cout << "--- Training Finished ---" << std::endl << std::endl;

        std::cout << "--- Final Test Set Evaluation ---" << std::endl;
        if (!test_data.images.empty()) {
            int correct_predictions = 0;
            for(size_t k=0; k < static_cast<size_t>(test_data.number_of_items); ++k) {
                int predicted_digit = get_prediction_digit(mnist_net, test_data.images[k]);
                
                Matrix actual_label_host = test_data.labels[k]; 
                
                int actual_digit = -1;
                for(int l=0; l<actual_label_host.getRow(); ++l) {
                    if(actual_label_host.getEntry(l,0) == 1.0) {
                        actual_digit = l;
                        break;
                    }
                }
                if (predicted_digit == actual_digit) {
                    correct_predictions++;
                }
            }
            double accuracy = (test_data.images.size() > 0) ? (static_cast<double>(correct_predictions) / test_data.images.size()) : 0.0;
            std::cout << "Final Test Accuracy: " << accuracy * 100.0 << "%" 
                      << " (" << correct_predictions << "/" << test_data.images.size() << ")" << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during network training/evaluation: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}