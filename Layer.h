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

    Layer(int inputSize, int outputSize, std::string _activationName);

    Matrix forward(Matrix& input); 
    Matrix activate(Matrix& z) const; 

    void printWeights() const;

    static double sigmoid(double x);
    static double sigmoidPrime(double x); 
    static double relu(double x);
    static double reluPrime(double x);    
};

#endif 