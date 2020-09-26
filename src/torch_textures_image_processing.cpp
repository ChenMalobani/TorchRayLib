/*******************************************************************************************
*
*   raylib [textures] example - Image processing
*
*   NOTE: Images are loaded in CPU memory (RAM); textures are loaded in GPU memory (VRAM)
*
*   This example has been created using raylib 1.4 (www.raylib.com)
*   raylib is licensed under an unmodified zlib/libpng license (View raylib.h for details)
*
*   Copyright (c) 2016 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

#include "raylib.h"
#include "../include/utils/vision_utils.hpp"
//#include "../include/png++/rgb_pixel.hpp"

#include <stdlib.h>             // Required for: free()

#include <torch/script.h>

#define NUM_PROCESSES    8

typedef enum {
    NONE = 0,
    COLOR_GRAYSCALE,
    COLOR_TINT,
    COLOR_INVERT,
    COLOR_CONTRAST,
    COLOR_BRIGHTNESS,
    FLIP_VERTICAL,
    FLIP_HORIZONTAL
} ImageProcess;

static const char *processText[] = {
        "NO PROCESSING",
        "COLOR GRAYSCALE",
        "COLOR TINT",
        "COLOR INVERT",
        "COLOR CONTRAST",
        "COLOR BRIGHTNESS",
        "FLIP VERTICAL",
        "FLIP HORIZONTAL"
};


//Image torchToPng(torch::Tensor &tensor_){
//    torch::Tensor tensor = tensor_.squeeze().detach().cpu().permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
//    tensor = tensor.clamp(0, 255);
//    tensor = tensor.to(torch::kU8);
//    size_t width = tensor.size(1);
//    size_t height = tensor.size(0);
//    auto pointer = tensor.data_ptr<unsigned char>();
//
//    Image image (width, height);
//    for (size_t j = 0; j < height; j++){
//        for (size_t i = 0; i < width; i++){
//            image[j][i].red = pointer[j * width * 3 + i * 3 + 0];
//            image[j][i].green = pointer[j * width * 3 + i * 3 + 1];
//            image[j][i].blue = pointer[j * width * 3 + i * 3 + 2];
//        }
//    }
//    return image;
//}

int main(int argc, char* argv[])
{
    VisionUtils VU = VisionUtils();
    torch::Device device(torch::kCUDA);
    const std::string modelName = "mosaic_cpp.pt";
    auto module = torch::jit::load(modelName, device);

    const int screenWidth = 800;
    const int screenHeight = 450;
    InitWindow(screenWidth, screenHeight, "raylib [textures] example - image processing");

    //From ray
    Image image = LoadImage("windmill.png");   // Loaded in CPU memory (RAM)
    // To torch

    auto tensor=VU.rayImageToTorch (image, device);
    VU.tensorDIMS(tensor);
    // For inference
    tensor = tensor.
            to(torch::kFloat). // For inference
            unsqueeze(-1). // Add batch
            permute({ 3, 0, 1, 2 }). // Fix order, now its {B,C,H,W}
            to(device);
    VU.tensorDIMS(tensor);
    // Apply the model
    torch::Tensor out_tensor = module.forward({ tensor }).toTensor();
    VU.tensorDIMS(out_tensor); // D=:[1, 3, 320, 480]
    out_tensor = out_tensor.to(torch::kFloat32).detach().cpu().squeeze(); //Remove batch dim, must convert back to torch::float
    VU.tensorDIMS(out_tensor); // D=:[1, 3, 320, 480]

    image=VU.torchToRayImage(out_tensor);

//    ImageFormat(&image, UNCOMPRESSED_R8G8B8A8);         // Format image to RGBA 32bit (required for texture update) <-- ISSUE
    Texture2D texture = LoadTextureFromImage(image);    // Image converted to texture, GPU memory (VRAM)

    int currentProcess = NONE;
    bool textureReload = false;

    Rectangle selectRecs[NUM_PROCESSES] = { 0 };

    for (int i = 0; i < NUM_PROCESSES; i++) selectRecs[i] = Rectangle{ 40.0f, (float)(50 + 32*i), 150.0f, 30.0f };

    SetTargetFPS(60);
    //---------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        if (IsKeyPressed(KEY_DOWN))
        {
            currentProcess++;
            if (currentProcess > 7) currentProcess = 0;
            textureReload = true;
        }
        else if (IsKeyPressed(KEY_UP))
        {
            currentProcess--;
            if (currentProcess < 0) currentProcess = 7;
            textureReload = true;
        }

        if (textureReload)
        {
            UnloadImage(image);                         // Unload current image data
            image = LoadImage("windmill.png"); // Re-load image data

            // NOTE: Image processing is a costly CPU process to be done every frame,
            // If image processing is required in a frame-basis, it should be done
            // with a texture and by shaders
            switch (currentProcess)
            {
                case COLOR_GRAYSCALE: ImageColorGrayscale(&image); break;
                case COLOR_TINT: ImageColorTint(&image, GREEN); break;
                case COLOR_INVERT: ImageColorInvert(&image); break;
                case COLOR_CONTRAST: ImageColorContrast(&image, -40); break;
                case COLOR_BRIGHTNESS: ImageColorBrightness(&image, -80); break;
                case FLIP_VERTICAL: ImageFlipVertical(&image); break;
                case FLIP_HORIZONTAL: ImageFlipHorizontal(&image); break;
                default: break;
            }

            Color *pixels = GetImageData(image);        // Get pixel data from image (RGBA 32bit)
            UpdateTexture(texture, pixels);             // Update texture with new image data
            free(pixels);                               // Unload pixels data from RAM

            textureReload = false;
        }
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        DrawText("IMAGE PROCESSING:", 40, 30, 10, DARKGRAY);

        // Draw rectangles
        for (int i = 0; i < NUM_PROCESSES; i++)
        {
            DrawRectangleRec(selectRecs[i], (i == currentProcess) ? SKYBLUE : LIGHTGRAY);
            DrawRectangleLines((int)selectRecs[i].x, (int) selectRecs[i].y, (int) selectRecs[i].width, (int) selectRecs[i].height, (i == currentProcess) ? BLUE : GRAY);
            DrawText( processText[i], (int)( selectRecs[i].x + selectRecs[i].width/2 - MeasureText(processText[i], 10)/2), (int) selectRecs[i].y + 11, 10, (i == currentProcess) ? DARKBLUE : DARKGRAY);
        }

        DrawTexture(texture, screenWidth - texture.width - 60, screenHeight/2 - texture.height/2, WHITE);
        DrawRectangleLines(screenWidth - texture.width - 60, screenHeight/2 - texture.height/2, texture.width, texture.height, BLACK);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    UnloadTexture(texture);       // Unload texture from VRAM
    UnloadImage(image);           // Unload image from RAM

    CloseWindow();                // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}