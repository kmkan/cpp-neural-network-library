#include <iostream>
#include <stdexcept>

class Matrix {
public:
    Matrix(int _row, int _col) {
        if (_row <= 0 || _col <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        row = _row;
        col = _col;

        matrix = new double[row * col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i * col + j] = 0;
            }
        }
    }
    Matrix(int _row, int _col, double val) {
        if (_row <= 0 || _col <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        row = _row;
        col = _col;

        matrix = new double[row * col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i * col + j] = val;
            }
        }
    }
    ~Matrix() {
        delete[] matrix;
    }
    void display() const {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                printf("%.2f ", matrix[i * col + j]);
            }
            printf("\n");
        }
    }
    int getRow() {
        return row;
    }
    int getCol() {
        return col;
    }
    double getEntry(int _row, int _col) {
        if (!isValidIndex(_row, _col)) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        return matrix[_row * col + _col];
    }
    void setEntry(int _row, int _col, double entry) {
        if (!isValidIndex(_row, _col)) {
            throw std::invalid_argument("Matrix dimensions must be positive.\n");
        }
        matrix[_row * col + _col] = entry;
    }
private: 
    int row;
    int col;
    double* matrix;
    bool isValidIndex(int _row, int _col) const {
        return _row >= 0 && _row < row && _col >= 0 && _col < col;
    }
};



int main() {
    Matrix newMatrix = Matrix(3, 3);
    newMatrix.display();
    Matrix newMatrixTwo = Matrix(4, 5, 1);
    newMatrixTwo.display();
    return 0;
}