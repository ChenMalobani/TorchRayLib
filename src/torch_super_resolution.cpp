
#include "../include/utils/vision_utils.hpp"
//#include "../include/png++/rgb_pixel.hpp"
#include <stdlib.h>             // Required for: free()
//#include "texture_types.h"
#include <torch/script.h>
#include <torch/torch.h>


int main(int argc, char *argv[]) {
    VisionUtils VU = VisionUtils();
    torch::DeviceType device_type = torch::kCPU;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Running on a GPU" << std::endl;
    } else {
        std::cout << "Running on a CPU" << std::endl;
    }
    torch::Device device(device_type);
    const std::string modelNameUdnie = "RRDB_ESRGAN_x4_000.pt";
    auto moduleUdnie = torch::jit::load(modelNameUdnie, device);
    torch::NoGradGuard no_grad_guard;
    at::init_num_threads();

    const char *fileName = "comic.png";
    const char *fileNameESRext = "_esr.png";
    string fileNameESR(string(fileName) + fileNameESRext);

    Image image = LoadImage(fileName);   // Loaded in CPU memory (RAM)
    Image imageGAN = VU.applyModelOnImage(device, moduleUdnie, image);
    ExportImage(imageGAN, fileNameESR.c_str());
    return 0;
}
