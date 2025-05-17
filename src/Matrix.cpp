#include "Matrix.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm> 

cublasHandle_t Matrix::cublas_handle = nullptr;
bool Matrix::cublas_initialized = false;

void Matrix::initCublasGlobal() {
    if (!cublas_initialized) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        cublas_initialized = true;
        std::cout << "cuBLAS Initialized." << std::endl;
    }
} 

void Matrix::destroyCublasGlobal() {
    if (cublas_initialized) {
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        cublas_initialized = false;
        cublas_handle = nullptr;
        std::cout << "cuBLAS Destroyed." << std::endl;
    }
}

Matrix::Matrix()
    : rows_val(0), cols_val(0), d_data(nullptr), data_on_device(false) {}

Matrix::Matrix(int r, int c, bool on_gpu_default)
    : rows_val(r), cols_val(c), d_data(nullptr), data_on_device(false) {
    if (r < 0 || c < 0) { 
        throw std::invalid_argument("Matrix dimensions cannot be negative.");
    }
    if (r == 0 || c == 0) {
        if (on_gpu_default) {
            data_on_device = true;
        }
    } else {
        allocate_host_memory();
        if (on_gpu_default) {
            to_device();
        }
    }
}

Matrix::Matrix(int r, int c, double val, bool on_gpu_default)
    : rows_val(r), cols_val(c), d_data(nullptr), data_on_device(false) {
    if (r < 0 || c < 0) {
        throw std::invalid_argument("Matrix dimensions cannot be negative.");
    }
     if (r == 0 || c == 0) {
        if (on_gpu_default) {
            data_on_device = true;
        }
    } else {
        allocate_host_memory();
        std::fill(h_data.begin(), h_data.end(), val);
        if (on_gpu_default) {
            to_device();
        }
    }
}

Matrix::~Matrix() {
    free_device_memory();
}

void Matrix::copy_from(const Matrix& other) {
    rows_val = other.rows_val;
    cols_val = other.cols_val;
    
    d_data = nullptr; 
    data_on_device = false;

    if (other.data_on_device && other.d_data != nullptr && other.rows_val > 0 && other.cols_val > 0) {
        allocate_device_memory(); 
        size_t size_bytes = static_cast<size_t>(rows_val) * cols_val * sizeof(double);
        CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size_bytes, cudaMemcpyDeviceToDevice));
        data_on_device = true;
        
        if (h_data.size() != static_cast<size_t>(rows_val * cols_val)) {
            h_data.resize(static_cast<size_t>(rows_val * cols_val));
        }
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));

    } else if (other.data_on_device && (other.rows_val == 0 || other.cols_val == 0)) {
        h_data.clear(); 
        data_on_device = true; 
    }
     else {
        h_data = other.h_data; 
    }
}


Matrix::Matrix(const Matrix& other) {
    copy_from(other);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }
    free_device_memory(); 
    copy_from(other);
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : rows_val(other.rows_val), cols_val(other.cols_val),
      h_data(std::move(other.h_data)), 
      d_data(other.d_data),             
      data_on_device(other.data_on_device) {
    other.rows_val = 0;
    other.cols_val = 0;
    other.d_data = nullptr; 
    other.data_on_device = false;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    free_device_memory(); 
    h_data.clear();       

    rows_val = other.rows_val;
    cols_val = other.cols_val;
    h_data = std::move(other.h_data); 
    d_data = other.d_data;             
    data_on_device = other.data_on_device;

    other.rows_val = 0;
    other.cols_val = 0;
    other.d_data = nullptr; 
    other.data_on_device = false;
    return *this;
}

void Matrix::allocate_host_memory() {
    if (rows_val > 0 && cols_val > 0) {
        h_data.assign(static_cast<size_t>(rows_val) * cols_val, 0.0);
    } else {
        h_data.clear(); 
    }
}

void Matrix::allocate_device_memory() {
    if (d_data == nullptr && rows_val > 0 && cols_val > 0) { 
        size_t size_bytes = static_cast<size_t>(rows_val) * cols_val * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_data, size_bytes));
    }
}

void Matrix::free_device_memory() {
    if (d_data != nullptr) {
        cudaFree(d_data); 
        d_data = nullptr;
    }
    data_on_device = false;
}

void Matrix::to_device() {
    if (rows_val == 0 || cols_val == 0) { 
        data_on_device = true; 
        if (d_data) { 
             free_device_memory(); 
             data_on_device = true; 
        }
        return;
    }

    if (data_on_device && d_data != nullptr) return; 

    allocate_device_memory(); 
    size_t size_bytes = static_cast<size_t>(rows_val) * cols_val * sizeof(double);

    if (h_data.empty() && (rows_val > 0 && cols_val > 0)) {
    } else if (!h_data.empty()) {
         CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size_bytes, cudaMemcpyHostToDevice));
    }
    data_on_device = true;
}

void Matrix::to_host() {
    if (rows_val == 0 || cols_val == 0) { 
        if (!h_data.empty()) h_data.clear(); 
        return; 
    }

    if (!data_on_device || d_data == nullptr) {
        return;
    }
    
    if (h_data.size() != static_cast<size_t>(rows_val) * cols_val) {
        h_data.resize(static_cast<size_t>(rows_val) * cols_val);
    }
    size_t size_bytes = static_cast<size_t>(rows_val) * cols_val * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, size_bytes, cudaMemcpyDeviceToHost));
}

bool Matrix::isValidIndex(int r, int c) const {
    return r >= 0 && r < rows_val && c >= 0 && c < cols_val;
}

double Matrix::getEntry(int r, int c) const {
    if (rows_val == 0 || cols_val == 0) {
        throw std::out_of_range("Matrix::getEntry: Cannot get entry from zero-dimension matrix.");
    }
    if (h_data.empty() && (rows_val > 0 && cols_val > 0)) { 
         throw std::runtime_error("Matrix::getEntry: Host data not initialized or empty for non-empty matrix. Call to_host() if data is on device.");
    }
    if (!isValidIndex(r, c)) {
        throw std::out_of_range("Matrix::getEntry: Index (" + std::to_string(r) + "," + std::to_string(c) + ") out of bounds for " + std::to_string(rows_val) + "x" + std::to_string(cols_val) + " matrix.");
    }
    return h_data[static_cast<size_t>(r) * cols_val + c];
}

void Matrix::setEntry(int r, int c, double entry) {
    if (rows_val == 0 || cols_val == 0) {
        throw std::out_of_range("Matrix::setEntry: Cannot set entry in zero-dimension matrix.");
    }
    if (h_data.empty() || h_data.size() != static_cast<size_t>(rows_val) * cols_val) {
       allocate_host_memory();
    }
    if (!isValidIndex(r, c)) {
        throw std::out_of_range("Matrix::setEntry: Index (" + std::to_string(r) + "," + std::to_string(c) + ") out of bounds for " + std::to_string(rows_val) + "x" + std::to_string(cols_val) + " matrix.");
    }
    h_data[static_cast<size_t>(r) * cols_val + c] = entry;
    data_on_device = false; 
}

void Matrix::display() const {
    if (data_on_device && (rows_val > 0 && cols_val > 0)) {
        std::cout << "(Note: Displaying host data. Call to_host() to ensure it's up-to-date if recent ops were on GPU)" << std::endl;
    }
    if (rows_val == 0 || cols_val == 0) {
        std::cout << "Matrix is " << rows_val << "x" << cols_val << " (empty)." << std::endl;
        return;
    }
    if (h_data.empty()) { 
        std::cout << "Matrix host data is uninitialized for " << rows_val << "x" << cols_val << " matrix." << std::endl;
        std::cout << "  (data_on_device: " << (data_on_device ? "true" : "false") << ")" << std::endl;
        return;
    }
    for (int i = 0; i < rows_val; i++) {
        for (int j = 0; j < cols_val; j++) {
            size_t index = static_cast<size_t>(i) * cols_val + j;
            if (index < h_data.size()) {
                 std::cout << std::fixed << std::setprecision(4) << h_data[index] << " ";
            } else {
                std::cout << " OOB "; 
            }
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::multiply(const Matrix& m) const {
    if (cols_val != m.rows_val) {
        throw std::invalid_argument("Matrix::multiply: Dimensions not compatible. LHS: " +
                                    std::to_string(rows_val) + "x" + std::to_string(cols_val) + ", RHS: " +
                                    std::to_string(m.rows_val) + "x" + std::to_string(m.cols_val));
    }
    
    if (this->rows_val == 0 || m.cols_val == 0 || this->cols_val == 0) {
        Matrix zero_result(this->rows_val, m.cols_val, 0.0); 
        if (this->data_on_device && m.data_on_device && Matrix::cublas_initialized) {
             if (zero_result.rows_val > 0 || zero_result.cols_val > 0) { 
                zero_result.to_device();
             } else { 
                zero_result.data_on_device = true;
             }
        }
        return zero_result;
    }

    Matrix result(rows_val, m.cols_val); 

    if (this->data_on_device && this->d_data && m.data_on_device && m.d_data && Matrix::cublas_initialized) {
        result.to_device(); 

        const double alpha = 1.0;
        const double beta = 0.0;

        CUBLAS_CHECK(cublasDgemm(Matrix::cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N, m.cols_val, this->rows_val, this->cols_val,  
                                 &alpha, m.d_data, m.cols_val, this->d_data, this->cols_val,  
                                 &beta, result.d_data, result.cols_val));
        result.data_on_device = true; 

    } else { 
        Matrix temp_lhs; 
        Matrix temp_rhs; 
        const Matrix* lhs = this;
        const Matrix* rhs = &m;
        
        if (this->data_on_device && this->d_data) {
            temp_lhs = *this; 
            temp_lhs.to_host(); 
            lhs = &temp_lhs;
        }
        if (m.data_on_device && m.d_data) {
            temp_rhs = m; 
            temp_rhs.to_host(); 
            rhs = &temp_rhs;
        }
        
        if(result.rows_val > 0 && result.cols_val > 0 && result.h_data.empty()){
            result.allocate_host_memory(); 
        } else if (result.rows_val == 0 || result.cols_val == 0) {
            result.h_data.clear(); 
        }

        for (int i = 0; i < result.rows_val; i++) {
            for (int j = 0; j < result.cols_val; j++) {
                double vectorProd = 0.0;
                for (int k_inner = 0; k_inner < lhs->cols_val; k_inner++) { 
                    if (lhs->h_data.empty() && (lhs->rows_val > 0 && lhs->cols_val > 0)) throw std::runtime_error("LHS h_data empty in CPU multiply");
                    if (rhs->h_data.empty() && (rhs->rows_val > 0 && rhs->cols_val > 0)) throw std::runtime_error("RHS h_data empty in CPU multiply");
                    vectorProd += lhs->h_data[static_cast<size_t>(i) * lhs->cols_val + k_inner] * rhs->h_data[static_cast<size_t>(k_inner) * rhs->cols_val + j];
                }
                if (!result.h_data.empty()) { 
                    result.h_data[static_cast<size_t>(i) * result.cols_val + j] = vectorProd;
                }
            }
        }
        result.data_on_device = false; 
    }
    return result;
}

Matrix Matrix::applyFunction(double (*f)(double x)) {
    Matrix temp_this_storage; 
    const Matrix* current_this = this;

    if (this->data_on_device && this->d_data && this->rows_val > 0 && this->cols_val > 0) {
        temp_this_storage = *this; 
        temp_this_storage.to_host(); 
        current_this = &temp_this_storage;
    } else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) {
        throw std::runtime_error("Matrix::applyFunction: Host data is empty for non-empty matrix. Call to_host() if data is on device.");
    }
    
    Matrix result(rows_val, cols_val, false); 
    if (rows_val == 0 || cols_val == 0) return result; 

    for (size_t i = 0; i < current_this->h_data.size(); ++i) {
        result.h_data[i] = f(current_this->h_data[i]);
    }
    return result;
}

Matrix Matrix::add(const Matrix& m) const {
    if (cols_val != m.cols_val || rows_val != m.rows_val) {
        throw std::invalid_argument("Matrix::add: Dimensions not compatible. LHS:" +
            std::to_string(rows_val) + "x" + std::to_string(cols_val) + " RHS:" +
            std::to_string(m.rows_val) + "x" + std::to_string(m.cols_val));
    }
    Matrix result(rows_val, cols_val, false); 
    if (rows_val == 0 || cols_val == 0) return result;

    Matrix temp_lhs; 
    Matrix temp_rhs; 
    const Matrix* lhs = this;
    const Matrix* rhs = &m;

    if (this->data_on_device && this->d_data) { temp_lhs = *this; temp_lhs.to_host(); lhs = &temp_lhs; }
    else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) { throw std::runtime_error("Matrix::add (LHS): Host data empty.");}

    if (m.data_on_device && m.d_data) { temp_rhs = m; temp_rhs.to_host(); rhs = &temp_rhs; }
    else if (m.h_data.empty() && (m.rows_val > 0 && m.cols_val > 0)) { throw std::runtime_error("Matrix::add (RHS): Host data empty.");}


    for (size_t i = 0; i < lhs->h_data.size(); ++i) {
        result.h_data[i] = lhs->h_data[i] + rhs->h_data[i];
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& m) const {
     if (cols_val != m.cols_val || rows_val != m.rows_val) {
        throw std::invalid_argument("Matrix::subtract: Dimensions not compatible.");
    }
    Matrix result(rows_val, cols_val, false); 
    if (rows_val == 0 || cols_val == 0) return result;

    Matrix temp_lhs; 
    Matrix temp_rhs; 
    const Matrix* lhs = this;
    const Matrix* rhs = &m;

    if (this->data_on_device && this->d_data) { temp_lhs = *this; temp_lhs.to_host(); lhs = &temp_lhs; }
    else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) { throw std::runtime_error("Matrix::subtract (LHS): Host data empty.");}
    
    if (m.data_on_device && m.d_data) { temp_rhs = m; temp_rhs.to_host(); rhs = &temp_rhs; }
    else if (m.h_data.empty() && (m.rows_val > 0 && m.cols_val > 0)) { throw std::runtime_error("Matrix::subtract (RHS): Host data empty.");}

    for (size_t i = 0; i < lhs->h_data.size(); ++i) {
        result.h_data[i] = lhs->h_data[i] - rhs->h_data[i];
    }
    return result;
}

Matrix Matrix::multiplyElements(const Matrix& m) const {
     if (cols_val != m.cols_val || rows_val != m.rows_val) {
        throw std::invalid_argument("Matrix::multiplyElements: Dimensions not compatible.");
    }
    Matrix result(rows_val, cols_val, false); 
    if (rows_val == 0 || cols_val == 0) return result;

    Matrix temp_lhs; 
    Matrix temp_rhs; 
    const Matrix* lhs = this;
    const Matrix* rhs = &m;

    if (this->data_on_device && this->d_data) { temp_lhs = *this; temp_lhs.to_host(); lhs = &temp_lhs; }
    else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) { throw std::runtime_error("Matrix::multiplyElements (LHS): Host data empty.");}

    if (m.data_on_device && m.d_data) { temp_rhs = m; temp_rhs.to_host(); rhs = &temp_rhs; }
    else if (m.h_data.empty() && (m.rows_val > 0 && m.cols_val > 0)) { throw std::runtime_error("Matrix::multiplyElements (RHS): Host data empty.");}


    for (size_t i = 0; i < lhs->h_data.size(); ++i) {
        result.h_data[i] = lhs->h_data[i] * rhs->h_data[i];
    }
    return result;
}

Matrix Matrix::multiplyScalar(double scalar) const {
    Matrix result(rows_val, cols_val, false); 
    if (rows_val == 0 || cols_val == 0) return result;
    
    Matrix temp_this_storage; 
    const Matrix* current_this = this;
    if (this->data_on_device && this->d_data) { temp_this_storage = *this; temp_this_storage.to_host(); current_this = &temp_this_storage; }
    else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) { throw std::runtime_error("Matrix::multiplyScalar: Host data empty.");}


    for (size_t i = 0; i < current_this->h_data.size(); ++i) {
        result.h_data[i] = current_this->h_data[i] * scalar;
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_val, rows_val, false); 
    if (rows_val == 0 || cols_val == 0) return result;

    Matrix temp_this_storage; 
    const Matrix* current_this = this;
    if (this->data_on_device && this->d_data) { temp_this_storage = *this; temp_this_storage.to_host(); current_this = &temp_this_storage; }
    else if (this->h_data.empty() && (this->rows_val > 0 && this->cols_val > 0)) { throw std::runtime_error("Matrix::transpose: Host data empty.");}


    for (int i = 0; i < current_this->rows_val; ++i) {
        for (int j = 0; j < current_this->cols_val; ++j) {
            result.h_data[static_cast<size_t>(j) * result.cols_val + i] = current_this->h_data[static_cast<size_t>(i) * current_this->cols_val + j];
        }
    }
    return result;
}