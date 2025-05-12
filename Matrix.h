#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <stdexcept>
#include <vector>

class Matrix {
public:
    // Constructors
    Matrix(int _row, int _col);                     
    Matrix(int _row, int _col, double val);          
    
    // Destructor
    ~Matrix();                                      

    // Accessor functions
    int getRow() const;                             // Returns number of rows
    int getCol() const;                             // Returns number of columns
    double getEntry(int _row, int _col) const;       // Returns value at specified row and column
    void setEntry(int _row, int _col, double entry); // Sets value at specified row and column

    // Display the matrix
    void display() const;                           // Prints matrix to standard output

    // Matrix operations
    Matrix applyFunction(double (*f)(double));      // Applies a function to every element
    Matrix add(Matrix& m) const;                    // Adds another matrix
    Matrix multiply(Matrix& m) const;               // Multiplies with another matrix

    // Operator overloads
    Matrix operator+(Matrix& m) const;              // Overload the '+' operator for matrix addition
    Matrix operator*(Matrix& m) const;              // Overload the '*' operator for matrix multiplication

private:
    int row;         // Number of rows
    int col;         // Number of columns
    double* matrix;  // Pointer to the matrix data

    // Private helper function to validate indices
    bool isValidIndex(int _row, int _col) const;   
};

#endif 