#include <iostream>
#include <memory>

#define TIGRAD_IMPLEMENTATION
#include <Tigrad/Tigrad.hpp>

int main()
{
    tg::Tigrad engine;
    tg::Tigrad::setActive(&engine);

    const tg::Value w = engine.createVal("w");
    const tg::Value x = engine.createVal("x");
    const tg::Value b = engine.createVal("b");

    const tg::Value z = (w * x) + b;
    const tg::Value y = tg::sigmoid(z);

    const size_t count = engine.compile();

    engine.data[w] = 2;
    engine.data[x] = 3;
    engine.data[b] = 4;
    engine.label[z] = "z";
    engine.label[y] = "y";

    engine.calcData();

    for (size_t i = 0; i < count; i++)
    {
        std::cout << engine.label[i] << ": Data: " << engine.data[i] << " Grad: " << engine.grad[i] << std::endl;
    }

    std::cout << std::endl;

    engine.zeroGrad();

    engine.calcGrad(y);
    for (size_t i = 0; i < count; i++)
    {
        std::cout << engine.label[i] << ": Data: " << engine.data[i] << " Grad: " << engine.grad[i] << std::endl;
    }

    tg::Tigrad::visualize(&engine);

    return 0;
}
