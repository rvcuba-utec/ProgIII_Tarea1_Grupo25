#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <algorithm>
#include <vector>
#include <cstddef>
#include <cmath>

class Tensor;

class TensorTransform {
    public :
    TensorTransform() = default;
    virtual Tensor apply (const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU: public TensorTransform {
    Tensor apply(const Tensor& t) const override;
};

class Sigmoid: public TensorTransform {
    Tensor apply(const Tensor& t) const override;
};

#endif
