/*******************************************************************************************
*
*   raylib [core] example - Windows drop files
*
*   This example only works on platforms that support drag & drop (Windows, Linux, OSX, Html5?)
*
*   This example has been created using raylib 1.3 (www.raylib.com)
*   raylib is licensed under an unmodified zlib/libpng license (View raylib.h for details)
*
*   Copyright (c) 2015 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

//#include "raylib.h"

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
//    const std::string modelNameUdnie = "mosaic_cpp.pt";
    auto moduleUdnie = torch::jit::load(modelNameUdnie, device);
    torch::NoGradGuard no_grad_guard;
    at::init_num_threads();

    const char *fileName="img_003_SRF_2_LR.png";
    const char *fileNameESR="baboon_esr.png";
    int count = 0;

    Image image = LoadImage(fileName);   // Loaded in CPU memory (RAM)
//    ImageFormat(&image,UNCOMPRESSED_R8G8B8A8);
    Image imageGAN = VU.applyModelOnImage(device, moduleUdnie,  image);
    ExportImage(imageGAN,fileNameESR);
//    imageGAN=LoadImage(fileNameESR);
//    Texture2D textureGAN = LoadTextureFromImage(imageGAN);    // Image converted to texture, GPU memory (VRAM)
//    DrawTexture(textureGAN, screenWidth - textureGAN.width - 60, screenHeight / 2 - textureGAN.height / 2, WHITE);

//    Image image = LoadImage(fileName);   // Loaded in CPU memory (RAM)
//    ImageFormat(&image,UNCOMPRESSED_R8G8B8A8);
//    Texture2D texture = LoadTextureFromImage(image);    // Image converted to texture, GPU memory (VRAM)
//    UnloadImage(image);

    return 0;
}
