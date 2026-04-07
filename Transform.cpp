#include "Transform.h"
#include "Tensor.h"

Tensor ReLU::apply(const Tensor& t) const {
    std::vector<double> newValues(t.size);
    std::vector<size_t> shape;

    for(int i = 0; i < t.nDims; ++i) { shape.push_back(t.shape[i]); }

    for(size_t i = 0; i < t.size; ++i) {
        newValues[i] = std::max(0.0, t.data[i]);
    }

    return Tensor(shape, newValues);
}

Tensor Sigmoid::apply(const Tensor& t) const {
    std::vector<double> newValues(t.size);
    std::vector<size_t> shape;

    for(int i = 0; i < t.nDims; ++i) { shape.push_back(t.shape[i]); }

    for(size_t i = 0; i < t.size; ++i) {
        newValues[i] = 1.0 / (1.0 + std::exp(-t.data[i]));
    }

    return Tensor(shape, newValues);
}
