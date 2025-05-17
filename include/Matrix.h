#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept> 

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err_)) + \
                                     " in file " + std::string(__FILE__) + " at line " + std::to_string(__LINE__)); \
        } \
    } while(0)

#define CUBLAS_CHECK(err) \
    do { \
        cublasStatus_t err_ = (err); \
        if (err_ != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS Error: status " + std::to_string(err_) + \
                                     " in file " + std::string(__FILE__) + " at line " + std::to_string(__LINE__)); \
        } \
    } while(0)

class Matrix {
private:
    int rows_val; 
    int cols_val; 
    std::vector<double> h_data; 
    double* d_data;             
    bool data_on_device;        

    bool isValidIndex(int r, int c) const;

    static cublasHandle_t cublas_handle;
    static bool cublas_initialized;

    void allocate_host_memory();
    void allocate_device_memory();
    void free_device_memory();
    void copy_from(const Matrix& other); 

public:
    static void initCublasGlobal();
    static void destroyCublasGlobal();

    Matrix(); 
    Matrix(int r, int c, bool on_gpu_default = false);
    Matrix(int r, int c, double val, bool on_gpu_default = false);

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    ~Matrix();

    int getRow() const { return rows_val; }
    int getCol() const { return cols_val; }
    double getEntry(int r, int c) const; 

    void setEntry(int r, int c, double entry); 

    void to_device();    
    void to_host();      
    bool is_on_device() const { return data_on_device; }
    double* get_device_ptr() { return d_data; } 
    const double* get_device_ptr() const { return d_data; } 

    void display() const; 
    Matrix applyFunction(double (*f)(double x)); 

    Matrix add(const Matrix& m) const;        
    Matrix subtract(const Matrix& m) const;   
    Matrix multiply(const Matrix& m) const;   
    Matrix multiplyElements(const Matrix& m) const; 
    Matrix multiplyScalar(double scalar) const;    
    Matrix transpose() const;                 

    Matrix operator+(const Matrix& m) const { return this->add(m); }
    Matrix operator-(const Matrix& m) const { return this->subtract(m); }
    Matrix operator*(const Matrix& m) const { return this->multiply(m); }
};

#endif 