#include <iostream>
#include <torch/torch.h>
#include "network.h"

int main(){
    Net net = Net(5, 20, 10, 3);
    torch::Tensor x = torch::randn({3, 5});
    torch::Tensor out = net->forward(x);
    std::cout << x << std::endl;
    std::cout << net << std::endl;
    std::cout << out << std::endl;
    return 0;
}