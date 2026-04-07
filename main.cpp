#include <iostream>
#include <vector>
#include "Tensor.h"
#include "Transform.h"

int main() {
    Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);

    Tensor x = input.view({1000, 400});

    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor out1 = matmul(x, W1);

    Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
    Tensor out2 = out1 + b1;

    ReLU relu;
    Tensor out3 = out2.apply(relu);

    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor out4 = matmul(out3, W2);

    Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
    Tensor out5 = out4 + b2;

    Sigmoid sig;
    Tensor output = out5.apply(sig);

    std::cout << output << "\n";
    return 0;
}
