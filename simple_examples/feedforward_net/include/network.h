#pragma once

#include<torch/torch.h>

struct NetImpl : torch::nn::Module {
    NetImpl(int input_dims, int fc1_dims, int fc2_dims, int output_dims) : fc1(input_dims, fc1_dims), fc2(fc1_dims, fc2_dims), out(fc2_dims, output_dims) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("out", out);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = out(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};

};

TORCH_MODULE(Net);