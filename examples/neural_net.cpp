#include <iostream>
#include <ranges>
#include <vector>



#define TIGRAD_IMPLEMENTATION
#include <Tigrad/Tigrad.hpp>

class Neuron
{
public:
    explicit Neuron(const std::vector<float>& inputs) : b(tg::Value(-1)), y(tg::Value(-1))
    {
        tg::Tigrad::setActive(&engine);

        x.reserve(inputs.size());
        for (const float input : inputs)
            x.emplace_back(engine.createConstant(input));

        w.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            w.emplace_back(engine.createVal("w" + std::to_string(i)));

        b = engine.createVal("b");

        auto sum = engine.createVal("sum");
        for (const auto& [w_i, x_i] : std::views::zip(w, x))
        {
            const auto sum_i = w_i * x_i;
            sum = sum + sum_i;
        }

        const auto z = sum + b;
        engine.label[z] = "z";
        y = tg::sigmoid(z);
        engine.label[y] = "y";

        cnt = engine.compile();

        // set all the data
        // todo: need better initialization for the weights and biases
        for (const auto & w_i : w)
        {
            engine.data[w_i] = 0.0f; // need to change this
        }
        engine.data[b] = 0.0f; // need to change this
    }

    ~Neuron() = default;


    void forward()
    {
        engine.calcData();
    }

    void backward()
    {
        engine.zeroGrad();
        engine.calcGrad(y);
    }

    void print()
    {
        for (int i = 0; i < cnt; i++)
        {
            std::cout << engine.label[i] << " - " << "Data: " << engine.data[i] << " Grad: " << engine.grad[i] << "\n";
        }
        std::cout << std::endl;
    }

private:

    tg::Tigrad engine;
    std::vector<tg::Value> w;
    std::vector<tg::Value> x;
    tg::Value b;
    tg::Value y;
    size_t cnt = 0;
};


int main()
{
    const std::vector inp = { 1.0f, 2.0f, 3.0f };
    Neuron n(inp);

    std::cout << "Initially\n";
    n.print();

    n.forward();
    std::cout << "Forward Pass\n";
    n.print();

    n.backward();
    std::cout << "Backward Pass\n";
    n.print();

    return 0;
}