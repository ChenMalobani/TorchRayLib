
#include "raylib.h"
#include "../include/utils/vision_utils.hpp"
//#include "../include/png++/rgb_pixel.hpp"
#include <stdlib.h>             // Required for: free()
//#include "texture_types.h"
#include <torch/script.h>
#include <torch/torch.h>

#define PL_MPEG_IMPLEMENTATION
#include "../include/pl_mpeg/pl_mpeg.h"
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stbi/stb_image_write.h"


int main(int argc, char *argv[]) {
    VisionUtils VU = VisionUtils();
    torch::DeviceType device_type = torch::kCPU;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Running on a GPU" << std::endl;
    } else {
        std::cout << "Running on a CPU" << std::endl;
    }
//    torch::Device device(device_type);
//    const std::string modelNameUdnie = "mosaic_cpp.pt";
//    auto moduleUdnie = torch::jit::load(modelNameUdnie, device);
//    torch::NoGradGuard no_grad_guard;
//    at::init_num_threads();
//
//    const char *fileName = "comic.png";
//    const char *fileNameESRext = "_esr.png";
//    string fileNameESR(string(fileName) + fileNameESRext);
//
//    Image image = LoadImage(fileName);   // Loaded in CPU memory (RAM)
//    Image imageGAN = VU.applyModelOnImage(device, moduleUdnie, image);
////    ExportImage(imageGAN, fileNameESR.c_str());

    const std::string videoFileName = "bjork-all-is-full-of-love.mpg";
    plm_t *plm = plm_create_with_filename(videoFileName.c_str());
    if (!plm) {
        std::cout <<"Couldn't open file:"<< videoFileName << std::endl;
        return 1;
    }
    plm_set_audio_enabled(plm, FALSE);
    int w = plm_get_width(plm);
    int h = plm_get_height(plm);
    int num_pixels = w * h;
    char png_name[16];
    plm_frame_t *frame = NULL;
    Image im = {0};

    for (int i = 0; i<100; i++) {
        frame = plm_decode_video(plm);
        uint8_t *rgb_data = (uint8_t*)malloc(num_pixels * 3);
        plm_frame_to_rgb(frame, rgb_data, w * 3);
        sprintf(png_name, "x/%04d.png", i);
        printf("Writing %s\n", png_name);

        im.data = rgb_data;
        im.width = w;
        im.height = h;
        im.format = UNCOMPRESSED_R8G8B8;
        im.mipmaps = 1;
        ExportImage(im, png_name);
    }

    return 0;
}
