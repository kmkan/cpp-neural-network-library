# Simple C++ Neural Network Library

This project is a basic neural network library implemented in C++, demonstrating core concepts like feedforward and backpropagation. It includes an example implementation for training a network on the MNIST handwritten digit dataset.

The matrix operations within the library can be optionally accelerated using CUDA if a compatible NVIDIA GPU and the CUDA toolkit are available.

## Features

* **Core Neural Network Components:**
    * `Matrix` class for numerical operations.
    * `Layer` class supporting different activation functions.
    * `Network` class to build and train neural networks.
* **Training:**
    * Backpropagation algorithm for gradient calculation.
    * Stochastic Gradient Descent (via batch training) for parameter updates.
    * Mean Squared Error loss function.
* **MNIST Example:**
    * Code to load and preprocess the MNIST dataset.
    * A `main.cpp` example demonstrating how to configure, train, and test a network on MNIST.
* **Optional CUDA Acceleration:**
    * The `Matrix` class's multiplication operation (`*` or `multiply()`) is accelerated with cuBLAS if operands are on the GPU.

## Prerequisites

* A C++11 (or newer) compatible compiler (e.g., g++, clang++).
* (Optional for CUDA) NVIDIA GPU with CUDA Toolkit installed (includes `nvcc` compiler and cuBLAS library).
* The MNIST dataset files (download from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) and decompress them). It's recommended to place them in a `data/` subdirectory within the project.

## How to Use

1.  **Configure Makefile (if using CUDA):**
    * Open the `Makefile`.
    * Ensure `NVCC_PATH_OVERRIDE` points to your CUDA toolkit's `bin` directory (e.g., `/usr/local/cuda-12.9/bin/` on Linux/WSL).
    * Adjust `CUDA_ARCH` (e.g., `-arch=sm_75`) to match your GPU's compute capability. If `nvcc` is in your system PATH, you might be able to simplify the `NVCC` variable.

2.  **Compile:**
    * Navigate to the project's root directory in your terminal (e.g., your WSL Ubuntu terminal if using CUDA on WSL).
    * Run the make command:
        ```bash
        make clean && make
        ```

3.  **Run the MNIST Example:**
    * After successful compilation, an executable (e.g., `nn_cuda_test`) will be created in the project root.
    * Run it:
        ```bash
        ./nn_cuda_test
        ```
    * The `main.cpp` is currently configured to run a test of the CUDA matrix multiplication or the MNIST training example. You can modify `main.cpp` to switch between tests or adjust training parameters (dataset size, epochs, learning rate).
