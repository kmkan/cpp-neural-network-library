#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <string> 
#include <iostream> 
#include <iomanip>  

class Matrix {
private:
    int row;
    int col;
    std::vector<double> matrix_data; 

    bool isValidIndex(int _row, int _col) const;

public:
    // Constructors
    Matrix(int _row, int _col);
    Matrix(int _row, int _col, double val);

    // Getters
    int getRow() const;
    int getCol() const;
    double getEntry(int _row, int _col) const;

    // Setter
    void setEntry(int _row, int _col, double entry);

    // Operations
    void display() const;
    Matrix applyFunction(double (*f)(double x));
    Matrix add(const Matrix& m) const;
    Matrix multiply(const Matrix& m) const;

    // Operator Overloads
    Matrix operator+(const Matrix& m) const;
    Matrix operator*(const Matrix& m) const;
};

#endif // MATRIX_H