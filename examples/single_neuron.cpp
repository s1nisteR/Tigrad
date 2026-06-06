// Single neuron and Stochastic Gradient Descent (SGD)

#include <array>
#include <iostream>
#include <random>
#include <vector>

#define TIGRAD_IMPLEMENTATION
#include <Tigrad/Tigrad.hpp>


std::array<std::array<float, 2>, 6> testData =
{
    {
        { 1.0f, 3.0f },
        { 2.0f, 5.0f },
        { 3.0f, 7.0f },
        { 4.0f, 9.0f },
        { 5.0f, 11.0f },
        { 6.0f, 13.0f }
    }
};


int main()
{
    tg::Tigrad ctx;
    tg::Tigrad::setActive(&ctx);

    // collections of all parameters for a neuron
    std::vector<tg::Value> params;

    // a single neuron and its equations
    auto x = ctx.createVal("x");
    auto y_real = ctx.createVal("y_real");
    auto w = ctx.createVal("w");
    auto b = ctx.createVal("b");
    auto y = w*x + b;
    ctx.label[y] = "y";
    auto loss = (y_real - y) * (y_real - y);    // squared error loss
    ctx.label[loss] = "loss";
    ctx.compile();

    // visualize the computation graph
    tg::Tigrad::visualize(&ctx);

    // add in all the parameters that we want to optimize for this neural net
    params.push_back(w);
    params.push_back(b);

    // initialize the params with something small and random (better initial weights could have been used here btw)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.0f, 1.0f);
    for (const auto& param : params)
        ctx.data[param] = dis(gen);

    // settings
    constexpr size_t epochs = 100;
    constexpr float learningRate = 0.01f;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // run for each sample in the dataset
        float totalLoss = 0.0f;
        for (const auto& sample : testData)
        {
            // set the inputs from data
            ctx.data[x] = sample[0];
            ctx.data[y_real] = sample[1];

            // forward pass
            ctx.calcData();
            totalLoss += ctx.data[loss];

            // backward pass
            ctx.zeroGrad();
            ctx.calcGrad(loss);

            // update params immediately after each sample (SGD)
            // (for batch GD, only the grads for the params should be accumulated(rest should be zeroed) and then 1/N should be multiplied with learning rate)
            for (const auto& param : params)
                ctx.data[param] -= learningRate * ctx.grad[param];
        }

        // print the current epoch, values of params, and MSE cost
        std::cout << std::format("Epoch {}: w: {} b: {} MSE: {}\n", epoch, ctx.data[w], ctx.data[b], totalLoss / testData.size());
    }
    return 0;
}