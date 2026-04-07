#include "Tensor.h"
#include "Transform.h"

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <random>
#include <algorithm>

static size_t computeSize(const std::vector<size_t>& s) {
    size_t total = 1;
    for (size_t i = 0; i < s.size(); ++i) {
        total *= s[i];
    }
    return total;
}

Tensor::Tensor(const std::vector<size_t>& s, double* sharedData, size_t sz, bool own): size(sz), nDims(static_cast<int>(s.size())), ownsData(own) {
    if (s.size() > 3) throw std::runtime_error("Máximo 3 dimensiones.");
    shape = new size_t[3]{0, 0, 0};
    for (size_t i = 0; i < s.size(); ++i) {
        shape[i] = s[i];
    }
    data = sharedData;
}

// Default Constructor
Tensor::Tensor(const std::vector<size_t>& s, const std::vector<double>& d): size(d.size()), nDims(s.size()), ownsData(true) {
    if (s.size() > 3) throw std::runtime_error("Máximo 3 dimensiones.");
    if (d.size() != computeSize(s)) throw std::runtime_error("Tamaño de values no coincide con shape.");

    shape = new size_t[3] {0, 0, 0};
    for (size_t i = 0; i < s.size(); ++i) shape[i] = s[i];

    data = new double[size];
    for (int i = 0; i < size; i++) data[i] = d[i];
}

// Copy Constructor
Tensor::Tensor(const Tensor& other): shape(new size_t[3]), size(other.size), data(new double[other.size]), nDims(other.nDims), ownsData(true) {
    std::copy(other.shape, other.shape + 3, shape);
    std::copy(other.data, other.data + other.size, data);
}

// Move Constructor
Tensor::Tensor(Tensor&& other) noexcept: shape(other.shape), size(other.size), data(other.data), nDims(other.nDims), ownsData(other.ownsData) {
    other.shape = nullptr;
    other.data = nullptr;
    other.size = 0;
    other.nDims = 0;
    other.ownsData = false;
}

// Copy Operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) { return *this; }

    if (ownsData) delete [] data;
    delete [] shape;

    size = other.size;
    nDims = other.nDims;
    ownsData = true;
    shape = new size_t[3] {0,0,0};
    std::copy(other.shape, other.shape + 3, shape);

    data = new double[size];
    std::copy(other.data, other.data + other.size, data);

    return *this;
}

// Move Operator
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {return *this;}

    if (ownsData) delete [] data;
    delete [] shape;

    size = other.size;
    nDims = other.nDims;
    shape = other.shape;
    data = other.data;
    ownsData = other.ownsData;

    other.size = 0;
    other.nDims = 0;
    other.data = nullptr;
    other.shape = nullptr;
    other.ownsData = false;

    return *this;
}

// Destructor
Tensor::~Tensor() {
    if (ownsData && data) delete [] data;
    delete [] shape;
}

// Static method zeros
Tensor Tensor::zeros(const std::vector<size_t> &s) {
    size_t total = computeSize(s);
    return Tensor(s, std::vector<double>(total,0.0));
}

// Static method ones
Tensor Tensor::ones(const std::vector<size_t> &s) {
    size_t total = computeSize(s);
    return Tensor(s, std::vector<double>(total,1.0));
}

// Static method random
Tensor Tensor::random(const std::vector<size_t> &s, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    size_t total = computeSize(s);
    std::vector<double> vec(total);
    for (auto& i : vec) i = dist(gen);

    return Tensor(s, vec);
}

// Static method arange - no he puesto corroboracion
Tensor Tensor::arange(int min, int max) {
    std::vector<double> vec;
    for (int i = min; i < max; i++) vec.push_back(i);
    return Tensor({vec.size()}, vec);
}

// Apply method
Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

// Helper
bool Tensor::hasSameShape(const Tensor& other) const {
    if (nDims != other.nDims) return false;
    for (int i = 0; i < nDims; ++i) {
        if (shape[i] != other.shape[i]) return false;
    }
    return true;
}

// Operator overload +
Tensor Tensor::operator+(const Tensor& other) const {
    if (hasSameShape(other)) {
        std::vector<double> result_vals(size);
        for (size_t i = 0; i < size; ++i) {
            result_vals[i] = data[i] + other.data[i];
        }
        std::vector<size_t> sh;
        for (int i = 0; i < nDims; ++i) sh.push_back(shape[i]);
        return Tensor(sh, result_vals);
    }

    if (nDims == 2 && other.nDims == 2 &&
        shape[0] > 1 && other.shape[0] == 1 &&
        shape[1] == other.shape[1]) {

        std::vector<double> result_vals(size);
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result_vals[i * shape[1] + j] = data[i * shape[1] + j] + other.data[j];
            }
        }
        std::vector<size_t> sh = {shape[0], shape[1]};
        return Tensor(sh, result_vals);
    }

    throw std::runtime_error("Dimensiones incompatibles para suma (broadcasting no soportado en este caso)");
}

// Operator overload -
Tensor Tensor::operator-(const Tensor& other) const {
    if (!hasSameShape(other)) throw std::runtime_error("Dimensiones incompatibles para resta.");

    std::vector<double> result_vals(size);
    for (size_t i = 0; i < size; ++i) result_vals[i] = data[i] - other.data[i];

    std::vector<size_t> current_shape;
    for(int i=0; i<nDims; ++i) current_shape.push_back(shape[i]);

    return Tensor(current_shape, result_vals);
}

// Operator overload * number
Tensor Tensor::operator*(double scalar) const {
    std::vector<double> result_vals(size);
    for (size_t i = 0; i < size; ++i) { result_vals[i] = data[i] * scalar; }

    std::vector<size_t> current_shape;
    for(int i=0; i<nDims; ++i) { current_shape.push_back(shape[i]); }

    return Tensor(current_shape, result_vals);
}

// Operator overload *
Tensor Tensor::operator*(const Tensor& other) const {
    if (!hasSameShape(other)) throw std::runtime_error("Dimensiones incompatibles para multiplicacion.");

    std::vector<double> result_vals(size);
    for (size_t i = 0; i < size; ++i) result_vals[i] = data[i] * other.data[i];

    std::vector<size_t> current_shape;
    for(int i=0; i<nDims; ++i) current_shape.push_back(shape[i]);

    return Tensor(current_shape, result_vals);
}

// View method
Tensor Tensor::view(const std::vector<size_t>& newShape) {
    size_t newSize = computeSize(newShape);
    if (newSize != size) throw std::runtime_error("El número total de elementos no coincide para view.");
    if (newShape.size() > 3) throw std::runtime_error("Máximo 3 dimensiones en view.");

    Tensor result(newShape, data, size, false);
    return result;
}

// Unsqueeze method
Tensor Tensor::unsqueeze(size_t axis) {
    if (nDims >= 3) throw std::runtime_error("No se puede hacer unsqueeze: ya tiene 3 dimensiones.");
    if (axis > (size_t)nDims) throw std::runtime_error("Eje invalido para unsqueeze.");

    std::vector<size_t> newShape;
    for (int i = 0; i < nDims; i++) newShape.push_back(shape[i]);

    newShape.insert(newShape.begin() + axis, 1);

    return this->view(newShape);
}

// Static concat function
Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t axis) {
    int nDims = tensors[0].nDims;
    size_t concatDimSize = 0;

    for (const auto& t : tensors) {
        if (t.nDims != nDims) throw std::runtime_error("Todas los tensores deben tener el mismo número de dimensiones");
        for (int i = 0; i < nDims; ++i) {
            if (i != static_cast<int>(axis) && t.shape[i] != tensors[0].shape[i]) {
                throw std::runtime_error("Dimensiones incompatibles");
            }
        }
        concatDimSize += t.shape[axis];
    }

    std::vector<size_t> newShape;
    for (int i = 0; i < nDims; ++i) {
        if (i == static_cast<int>(axis)) newShape.push_back(concatDimSize);
        else newShape.push_back(tensors[0].shape[i]);
    }

    size_t newSize = computeSize(newShape);
    std::vector<double> newData(newSize);

    size_t offset = 0;
    for (const auto& t : tensors) {
        size_t sliceSize = t.size / t.shape[axis];
        for (size_t i = 0; i < t.shape[axis]; ++i) {
            std::copy(t.data + i * sliceSize, t.data + (i + 1) * sliceSize,
                      newData.begin() + offset);
            offset += sliceSize;
        }
    }

    return Tensor(newShape, newData);
}

// Friend dot function
double dot(const Tensor& a, const Tensor& b) {
    if (a.nDims != 1 || b.nDims != 1) throw std::runtime_error("La función dot solo acepta vectores (1D). Usa matmul para matrices.");
    if (a.size != b.size) throw std::runtime_error("Los vectores deben tener el mismo tamaño para el producto punto.");

    double result = 0.0;
    for (size_t i = 0; i < a.size; ++i) result += a.data[i] * b.data[i];

    return result;
}

// Friend matmul function
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape[1] != b.shape[0]) throw std::runtime_error("Dimensiones incompatibles para matmul.");

    std::vector<size_t> newShape = { a.shape[0], b.shape[1] };
    size_t newSize = newShape[0] * newShape[1];
    std::vector<double> newData(newSize, 0.0);

    for (size_t i = 0; i < a.shape[0]; ++i) {
        for (size_t j = 0; j < b.shape[1]; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a.shape[1]; ++k) sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
            newData[i * b.shape[1] + j] = sum;
        }
    }
    return Tensor(newShape, newData);
}

// Print overload
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(";
    for (int i = 0; i < t.nDims; ++i) {
        os << t.shape[i];
        if (i < t.nDims - 1) os << ", ";
    }
    os << ") =\n";

    if (t.nDims == 1) {
        os << "  ";
        for (size_t i = 0; i < t.size; ++i) {
            os << t.data[i];
            if (i < t.size - 1) os << ", ";
        }
    } else if (t.nDims == 2) {
        for (size_t i = 0; i < t.shape[0]; ++i) {
            os << "  [";
            for (size_t j = 0; j < t.shape[1]; ++j) {
                os << t.data[i * t.shape[1] + j];
                if (j < t.shape[1] - 1) os << ", ";
            }
            os << "]";
            if (i < t.shape[0] - 1) os << ",\n";
        }
    } else {  // 3D
        os << " 3D\n";
        for (size_t i = 0; i < t.size; ++i) os << t.data[i] << " ";
    }
    os << "\n";
    return os;
}
