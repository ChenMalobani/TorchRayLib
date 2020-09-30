/*******************************************************************************************
*
*   raygui - image exporter
*
*   DEPENDENCIES:
*       raylib 2.1  - Windowing/input management and drawing.
*       raygui 2.0  - Immediate-mode GUI controls.
*
*   COMPILATION (Windows - MinGW):
*       gcc -o $(NAME_PART).exe $(FILE_NAME) -I../../src -lraylib -lopengl32 -lgdi32 -std=c99
*
*   LICENSE: zlib/libpng
*
*   Copyright (c) 2020 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

#include "raylib.h"
//#define RAYGUI_IMPLEMENTATION
//#define RAYGUI_SUPPORT_RICONS

#pragma warning( push, 0 )
#pragma warning( disable : 4576 )
#define RAYGUI_IMPLEMENTATION
#define RAYGUI_SUPPORT_RICONS
#include "../include/raygui/raygui.h"
#pragma warning( pop )

#include <stdlib.h>             // Required for: free()
//#include "texture_types.h"
#include <torch/script.h>
#include <torch/torch.h>
#include "../include/utils/vision_utils.hpp"

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------

Font gamefont;

int main()
{
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

    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1200;
    const int screenHeight = 800;

    const char * mainTitle="TorchRayLib: Super-Resolution";
    InitWindow(screenWidth, screenHeight, mainTitle);
    gamefont = LoadFont("GameCube.ttf");
    GuiSetFont(gamefont);
    GuiSetStyle(DEFAULT, TEXT_SIZE, 16);
    GuiFade(0.9f);
    GuiSetStyle(DEFAULT, TEXT_SPACING, 3);

    // Configure the GUI styles.
//    GuiLoadStyleDefault();

    // Cyber Style
    GuiSetStyle(0, 0, 0xf0f0f0ff);
    GuiSetStyle(0, 1, 0x868686ff);
    GuiSetStyle(0, 2, 0xe6e6e6ff);
    GuiSetStyle(0, 3, 0x929999ff);
    GuiSetStyle(0, 4, 0xeaeaeaff);
    GuiSetStyle(0, 5, 0x98a1a8ff);
    GuiSetStyle(0, 6, 0x3f3f3fff);
    GuiSetStyle(0, 7, 0xf6f6f6ff);
    GuiSetStyle(0, 8, 0x414141ff);
    GuiSetStyle(0, 9, 0x8b8b8bff);
    GuiSetStyle(0, 10, 0x777777ff);
    GuiSetStyle(0, 11, 0x959595ff);
    GuiSetStyle(0, 16, 0x00000010);
    GuiSetStyle(0, 17, 0x00000001);
    GuiSetStyle(0, 18, 0x9dadb1ff);
    GuiSetStyle(0, 19, 0x6b6b6bff);


//    GuiLoadStyleDefault();
//    GuiSetFont(GetFontDefault());
//    GuiSetStyle(TEXTBOX, TEXT_ALIGNMENT, GUI_TEXT_ALIGN_CENTER);

    // GUI controls initialization
    //----------------------------------------------------------------------------------
    Rectangle windowBoxRec = { screenWidth/2 - 110, screenHeight/2 - 100, 220, 190 };
    bool windowBoxActive = false;

    int fileFormatActive = 0;
    const char *fileFormatTextList[1] = { ".png"};

    int pixelFormatActive = 0;
    const char *pixelFormatTextList[1] = {"RGB"};

    bool textBoxEditMode = false;
    char fileName[64] = "ESRGAN";
    //--------------------------------------------------------------------------------------

    Image image = { 0 };
    Texture2D texture = { 0 };

    bool imageLoaded = false;
    float imageScale = 1.0f;
    Rectangle imageRec = { 0.0f };

    bool btnExport = false;

    SetTargetFPS(60);
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
//        DrawText("Drop a PNG image here to start the super resolution model.", 20, 75, 18, WHITE);
        // Update
        //----------------------------------------------------------------------------------
        if (IsFileDropped())
        {
            int fileCount = 0;
            char **droppedFiles = GetDroppedFiles(&fileCount);

            if (fileCount == 1)
            {
                Image imTemp = LoadImage(droppedFiles[0]);

                if (imTemp.data != NULL)
                {
                    UnloadImage(image);
                    image = imTemp;

                    UnloadTexture(texture);
                    texture = LoadTextureFromImage(image);

                    imageLoaded = true;
//                    pixelFormatActive = image.format - 1;
                    pixelFormatActive = UNCOMPRESSED_R8G8B8A8;

                    if (texture.height > texture.width) imageScale = (float)(screenHeight - 100)/(float)texture.height;
                    else imageScale = (float)(screenWidth - 100)/(float)texture.width;
                }
            }

            ClearDroppedFiles();
        }

        if (btnExport)
        {
            if (imageLoaded)
            {
//                ImageFormat(&image, pixelFormatActive + 1);
//                ImageFormat(&image,UNCOMPRESSED_R8G8B8A8);
                if ((GetExtension(fileName) == NULL) || (!IsFileExtension(fileName, ".png"))) strcat(fileName, ".png\0");     // No extension provided
//                image = LoadImage(fileName);   // Loaded in CPU memory (RAM)
                VU.applyModelOnImage(device, moduleUdnie,  image);
                ExportImage(image,fileName);
//                UnloadImage(image);
                UnloadTexture(texture);
                texture = LoadTextureFromImage(image);
//                ExportImage(image, fileName);
//                    UnloadImage(image);
//                    image=imageGAN;
            }
            windowBoxActive = false;
        }
        if (imageLoaded)
        {
            imageScale += (float)GetMouseWheelMove()*0.05f;   // Image scale control
            if (imageScale <= 0.1f) imageScale = 0.1f;
            else if (imageScale >= 5) imageScale = 5;

            imageRec = Rectangle{ screenWidth/2 - (float)image.width*imageScale/2,
                                  screenHeight/2 - (float)image.height*imageScale/2,
                                  (float)image.width*imageScale, (float)image.height*imageScale };
        }
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

//            ClearBackground(RAYWHITE);
        ClearBackground(GRAY);
        DrawText(mainTitle, 20, 20, 20, BLACK);

        if (texture.id > 0)
        {
            DrawTextureEx(texture, Vector2{ screenWidth/2 - (float)texture.width*imageScale/2, screenHeight/2 - (float)texture.height*imageScale/2 }, 0.0f, imageScale, WHITE);

            DrawRectangleLinesEx(imageRec, 1, CheckCollisionPointRec(GetMousePosition(), imageRec) ? RED : DARKGRAY);
            DrawText(FormatText("SCALE: %.2f%%", imageScale*100.0f), 20, screenHeight - 40, 20, GetColor(GuiGetStyle(DEFAULT, LINE_COLOR)));
        }
        else
        {
            DrawText("TO RUN A PYTORCH SUPER RESOLUTION MODEL, DRAG & DROP YOUR IMAGE", 30, 200, 20, BLACK);
            GuiDisable();
        }

        if (GuiButton(Rectangle{ screenWidth - 170, screenHeight - 50, 150, 30 }, "SR Image")) windowBoxActive = true;
        GuiEnable();

        // Draw window box: windowBoxName
        //-----------------------------------------------------------------------------
        if (windowBoxActive)
        {
            DrawRectangle(0, 0, screenWidth, screenHeight, Fade(GetColor(GuiGetStyle(DEFAULT, BACKGROUND_COLOR)), 0.75f));
            windowBoxActive = !GuiWindowBox(Rectangle{ windowBoxRec.x, windowBoxRec.y, 290, 220 }, "Super Resolution Options");

            GuiLabel(Rectangle{ windowBoxRec.x + 10, windowBoxRec.y + 35, 60, 25 }, "Format:");
            fileFormatActive = GuiComboBox(Rectangle{ windowBoxRec.x + 80, windowBoxRec.y + 35, 130, 25 },
                                           TextJoin(fileFormatTextList, 1, ";"), fileFormatActive);
            GuiLabel(Rectangle{ windowBoxRec.x + 10, windowBoxRec.y + 70, 63, 25 }, "Pixel:");
            pixelFormatActive = GuiComboBox(Rectangle{ windowBoxRec.x + 80, windowBoxRec.y + 70, 130, 25 },
                                            TextJoin(pixelFormatTextList, 1, ";"), pixelFormatActive);
            GuiLabel(Rectangle{ windowBoxRec.x + 10, windowBoxRec.y + 105, 50, 25 }, "Name:");
            if (GuiTextBox(Rectangle{ windowBoxRec.x + 80, windowBoxRec.y + 105, 130, 25 },
                           fileName, 64, textBoxEditMode)) textBoxEditMode = !textBoxEditMode;

            btnExport = GuiButton(Rectangle{ windowBoxRec.x + 10, windowBoxRec.y + 145, 260, 40 }, "Save Image");
        }
        else btnExport = false;

        if (btnExport) DrawText("Image exported!", 20, screenHeight - 20, 20, RED);
        //-----------------------------------------------------------------------------

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    UnloadImage(image);
    UnloadTexture(texture);

    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}
