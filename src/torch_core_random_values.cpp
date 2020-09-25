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

torch::Tensor sigmoid001(const torch::Tensor& x) {
    torch::Tensor sig = 1.0 / (1.0 + torch::exp((-x)));
    return sig;
}


int main(void)
{
    torch::Device device(torch::kCUDA);
    torch::Tensor tensor = torch::eye(3).to(device);
    std::cout<<tensor<<std::endl;

    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "TorchRayLib:PyTorch GPU random random values (c++17)");

    int framesCounter = 0;          // Variable used to count frames
    auto randValueTorch= (int)(1000 * (torch::rand(1).to(device).data().detach().item().toFloat()));

    int randValue=randValueTorch;
//    torch::Tensor t0 = torch::rand(1).to(device); // Allocate a tensor on the GPU
//    t0 = sigmoid001(t0);
    //Print (typeid(t0).name());
//    auto x = (t0).data().detach().item().toFloat(); // Move it to the CPU

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        framesCounter++;

        // Every two seconds (120 frames) a new random value is generated
        if (((framesCounter/60)%2) == 1)
        {
            randValue= (int)(10000 * (torch::rand(1).to(device).data().detach().item().toFloat()));
            framesCounter = 0;
        }
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(RAYWHITE);

            DrawText("Generate a random value on the GPU using PyTorch", 30, 100, 20, MAROON);

            DrawText(TextFormat("%i", randValue), 200, 180, 100, ORANGE);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}