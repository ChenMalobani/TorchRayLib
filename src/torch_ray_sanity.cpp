//#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <iostream>
#include <chrono>
#include <iostream>
#include <typeinfo>
#include <thread>
#include <future>

using namespace std;
using namespace std::chrono;

#include <torch/torch.h>
#include "../include/utils/vision_utils.hpp"


int main(int argc, char *argv[]) {

    VisionUtils VU = VisionUtils();
    torch::Device device = VU.getDevice();
    torch::Tensor tensor = torch::eye(1).to(device);
    std::cout << tensor << std::endl;

    std::stringstream sstm;
    sstm << tensor.toString();
    std::cout << tensor.data().detach().item().toFloat() << std::endl;

    const std::string modelName = "erfnet_fs.pt";
    auto module = torch::jit::load(modelName, device);
    if (!std::ifstream(modelName)) {
        std::cout << "ERROR: Could not open the required trained PyTorch module file from path";
    } else {
        std::cout << "Loaded required trained PyTorch module module file from path";
    }
    return 0;
}