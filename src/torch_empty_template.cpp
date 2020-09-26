/*******************************************************************************************
*
*   raylib [core] example - Generate random values
*
*   This example has been created using raylib 1.1 (www.raylib.com)
*   raylib is licensed under an unmodified zlib/libpng license (View raylib.h for details)
*
*   Copyright (c) 2014 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

#include "raylib.h"
#include <torch/script.h>

// width/height/centerx/centery
static void midptellipse(int rx, int ry, int xc, int yc);
static int map[200][200]={0};

//int main(int argc, char* argv[])
//{
//    torch::Device device(torch::kCUDA);
//    torch::Tensor tensor = torch::eye(1).to(device);
//    std::cout<<tensor<<std::endl;
//
//    // Initialization
//    //--------------------------------------------------------------------------------------
//    const int screenWidth = 800;
//    const int screenHeight = 600;
//    InitWindow(screenWidth, screenHeight, "TorchRayLib: emprt template");
//    auto randValueTorch= (int)(1000 * (torch::rand(1).to(device).data().detach().item().toFloat()));
//    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
//    while (!WindowShouldClose())    // Detect window close button or ESC key
//    {
//        BeginDrawing();
//            ClearBackground(BLACK);
//            DrawText("Generate a random value on the GPU using PyTorch", 30, 100, 20, MAROON);
//        EndDrawing();
//    }
//
//    CloseWindow();        // Close window and OpenGL context
//    return 0;
//}

