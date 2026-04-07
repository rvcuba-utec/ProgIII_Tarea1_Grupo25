#ifndef TENSOR_H
#define TENSOR_H

#include "Transform.h"

#include <cstddef>
#include <vector>
#include <iostream>

class TensorTransform;

class Tensor {
    private:
    size_t* shape = nullptr;
    double* data = nullptr;
    size_t size;
    int nDims;
    bool ownsData;

    friend class ReLU;
    friend class Sigmoid;

    Tensor(const std::vector<size_t>& s, double* sharedData, size_t sz, bool own);

    public:
    // Default Constructor
    Tensor(const std::vector<size_t>& s, const std::vector<double>& d);
    // Copy Constructor
    Tensor(const Tensor& other);
    // Move Constructor
    Tensor(Tensor&& other) noexcept;
    // Destructor
    ~Tensor();

    // Copy Operator
    Tensor& operator= (const Tensor& other);
    // Move Operator
    Tensor& operator= (Tensor&& other) noexcept;

    // Static method zeros
    static Tensor zeros(const std::vector<size_t>& s);
    // Static method ones
    static Tensor ones(const std::vector<size_t> &s);
    // Static method random
    static Tensor random(const std::vector<size_t> &s, double min, double max);
    // Static method arange
    static Tensor arange(int min, int max);

    // Apply method
    Tensor apply(const TensorTransform& transform) const ;

    // Helper
    bool hasSameShape(const Tensor& other) const;

    // Operator overload +
    Tensor operator+ (const Tensor& other) const;
    // Operator overload -
    Tensor operator- (const Tensor& other) const;
    // Operator overload * number
    Tensor operator* (double number) const;
    // Operator overload *
    Tensor operator* (const Tensor& other) const;

    // View method
    Tensor view(const std::vector<size_t>& newShape);
    // Unsqueeze method
    Tensor unsqueeze(size_t axis);

    // Static concat function
    static Tensor concat(const std::vector<Tensor>& tensors, size_t axis);

    // Friend dot function
    friend double dot(const Tensor& a, const Tensor& b);
    // Friend matmul function
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    // Print overload
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
};

#endif
