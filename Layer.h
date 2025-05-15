#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include <string>
#include <vector>
#include <stdexcept>

class Layer {
public:
    Matrix weights;
    Matrix biases;
    std::string activationName;

    Matrix last_input;      
    Matrix last_z;          
    Matrix grad_weights;    
    Matrix grad_biases;     

    Matrix delta_weights;   
    Matrix delta_biases;    

    Layer(int inputSize, int outputSize, std::string _activationName);

    Matrix forward(Matrix& input);
    Matrix activate(Matrix& z) const;
    Matrix activatePrime(Matrix& z_values) const; 

    Matrix backward(const Matrix& d_output_error); 

    void zero_deltas(); 
    void accumulate_gradients();
    void update_parameters_from_deltas(double learning_rate, int batch_size); 

    void printWeights() const;

    static double sigmoid(double x);
    static double sigmoidPrime(double x); 
    static double relu(double x);
    static double reluPrime(double x); 
};

#endif  
