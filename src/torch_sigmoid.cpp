
#include "raylib.h"
#include "../include/utils/vision_utils.hpp"
#include <math.h> // needed for cos and sin.
#include <torch/script.h> // One-stop header.

torch::Tensor sigmoid001(const torch::Tensor& x) {
    torch::Tensor sig = 1.0 / (1.0 + torch::exp((-x)));
    return sig;
}

int main(int argc, char* argv[])
{
    VisionUtils VU = VisionUtils();
    torch::Device device = VU.getDevice();

    torch::Tensor tensor = torch::rand(1).to(device);
    auto randValueTorch= (tensor.data().detach().item().toFloat());
    std::cout<<tensor<<std::endl;
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 600;
    const int screenHeight = 400;

    InitWindow(screenWidth, screenHeight, "TorchRayLib: The sigmoid function in PyTorch.");

    Vector2 position;
    position.x = screenWidth/2.0;
    position.y = screenHeight/2.0;

    Vector2 position_sig;
    position_sig.x = screenWidth/2.0;
    position_sig.y = screenHeight/2.0;

    auto angle= (int)(359 * (torch::rand(1).to(device).data().detach().item().toFloat()));

//    float angle=GetRandomValue(0,359);
 
    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        
        // cos and sin here.
        position.x += (float)cos(angle)*4;
        position.y += (float)sin(angle)*4;

        position_sig.x += 10 * torch::rand(1).to(device).data().detach().item().toFloat()*
                sigmoid001(torch::rand(1)).to(device).data().detach().item().toFloat();
        position_sig.y += - 12 * torch::rand(1).to(device).data().detach().item().toFloat();



        if((int)(100 * (torch::rand(1).to(device).data().detach().item().toFloat()))<5) {
            angle = (int) (359 * (torch::rand(1).to(device).data().detach().item().toFloat()));
        }
        if(position.x>screenWidth || position.x<0 || position.y<0 || position.y>screenHeight){
            position.x=screenWidth/2.0;
            position.y=screenHeight/2.0;
            }

        if(position_sig.x>screenWidth || position_sig.x<0 || position_sig.y<0 || position_sig.y>screenHeight){
            position_sig.x=screenWidth/4.0  * torch::rand(1).to(device).data().detach().item().toFloat();
            position_sig.y=screenHeight/14.0 * 10* torch::rand(1).to(device).data().detach().item().toFloat();
        }

        //----------------------------------------------------------------------------------
        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();
            ClearBackground(RAYWHITE);
            DrawCircle(position.x,position.y,
                       (int)(100 * (torch::rand(1).to(device).data().detach().item().toFloat())),
                       DARKPURPLE);
            DrawCircle(position_sig.x,position_sig.y,
                   (int)(20 * (torch::rand(1).to(device).data().detach().item().toFloat())),
                   GREEN);
        DrawCircle(position_sig.y,position_sig.x,
                   (int)(40 * (torch::rand(1).to(device).data().detach().item().toFloat())),
                   ORANGE);
        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;


}
