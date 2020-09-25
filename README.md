
<h4 align="center">TorchRayLib++: A CMake based integration of the RayLib GUI library and the Libtorch C++ Deep Learning Library.</h4>
      
<p align="center">
  <a href="#about">About</a> •
  <a href="#credits">Credits</a> •
  <a href="#installation">Installation</a> •  
  <a href="#fexamples">Examples</a> •  
  <a href="#author">Author</a> •  
  <a href="#license">License</a>
</p>

<h1 align="center">  
  <img src="https://github.com/QuantScientist/PngTorch/blob/master/asstes/logo.png?raw=true" width="50%"></a>
</h1>

---

## About

<table>
<tr>
<td>
  
**TorchRayLib++** is a CMake based **integration** of the well-known **_RayLib GUI_** library 
with my favourite Deep Learning Library Libtorch: the **_PyTorch_** C++ frontend.

### RayLib 
RayLib is an amazing library which has been widely adopted by the gaming community. 
Read more about the raylib game framework here: https://www.raylib.com/
Or look up projects using it here:
https://www.google.com/search?q=raylib+site:github.com   

### PyTorch / Libtorch C++ 
PyTorch is a Python package that provides two high-level features, Tensor computation (like NumPy) with strong GPU acceleration
Deep neural networks built on a tape-based autograd system. In this project we use the C++ version entitled Libtorch. 
https://pytorch.org/ 
 
![TorchRayLib++ Code](https://github.com/QuantScientist/TorchRayLib/blob/master/asstes/torch_core_random_values.gif?raw=true)
 
<p align="right">
<sub>(Preview)</sub>
</p>

</td>
</tr>
</table>

## A simple example 
The folowing example create a ray window, allocates a `torch::tensor` on the GPU and draws the value 
into a ray window. 
 
```cpp
#include "raylib.h"
#include <torch/script.h>

int main(void)
{
    torch::Device device(torch::kCUDA);
    torch::Tensor tensor = torch::eye(3).to(device);
    std::cout<<tensor<<std::endl;
    const int screenWidth = 800;
    const int screenHeight = 450;
    InitWindow(screenWidth, screenHeight, "TorchRayLib:PyTorch GPU random random values (c++17)");

    int framesCounter = 0;          // Variable used to count frames
    auto randValueTorch= (int)(1000 * (torch::rand(1).to(device).data().detach().item().toFloat()));
    int randValue=randValueTorch;

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {        
        framesCounter++;    
        if (((framesCounter/60)%2) == 1)
        {
            randValue= (int)(10000 * (torch::rand(1).to(device).data().detach().item().toFloat()));
            framesCounter = 0;
        }        
        BeginDrawing();
            ClearBackground(RAYWHITE);
            DrawText("Generate a random value on the GPU using PyTorch", 30, 100, 20, MAROON);
            DrawText(TextFormat("%i", randValue), 200, 180, 100, ORANGE);
        EndDrawing();
    }
    CloseWindow();        // Close window and OpenGL context
    return 0;
}
```

## Credits 
* PyTorch CPP examples by koba-jon https://github.com/koba-jon/pytorch_cpp.
 
* PyTorch CPP examples + CMake build: https://github.com/prabhuomkar/pytorch-cpp/

* For the NeuralStyle transfer models which I traced to C++ see https://github.com/gnsmrky/pytorch-fast-neural-style-for-web 
and https://github.com/pytorch/examples/tree/master/fast_neural_style

* RayLib UI https://github.com/raysan5/raylib which is licensed under 
an unmodified zlib/libpng license (View raylib.h for details) Copyright (c) 2014 Ramon Santamaria (@raysan5) 


## Features

|                            | 🔰 TorchRayLib++ CMake  | |
| -------------------------- | :----------------: | :-------------:|
| PyTorch CPU tensor to PNG        |         ✔️                 
| PyTorch GPU tensors to PNG       |         ✔️                 
| Libtorch C++ 1.6           |         ✔️                 
| RayLib           |         ✔️                 


## Examples

### A Simple example, mainly for testing the integration. Allocates a tensor on the GPU without ray.

 
### Load a trained PyTorch NeuralStyle transfer model in C++ (**see pth folder**), load an Image in C++, run a trained pytorch model on it and save the output.
 ![TorchRayLib++ Code](https://github.com/QuantScientist/TorchRayLib/blob/master/asstes/amber.png_mosaic_cpp.pt-out.png?raw=true)


## Requirements:
* Windows 10 and Microsoft Visual C++ 2019 16.4, Linux is not supported at the moment.
* NVIDIA CUDA 10.2. I did not test with any other CUDA version. 
* PyTorch / LibTorch c++ version 1.6.  
* 64 bit only.  
* CMake 3.18  
* libpng, png++ 
* RayLib GUI

Please setup CLion as follows: 
![TorchRayLib++ Code](https://github.com/QuantScientist/TorchRayLib/blob/master/assets/clion.png?raw=true)

## Installation 

#### Downloading and installing steps LIBTORCH C++:
* **[Download]()** the latest version of Libtorch for Windows here: https://pytorch.org/.
![TorchRayLib++ Code](https://github.com/QuantScientist/TorchRayLib/blob/master/assets/libtorch16.png?raw=true)

* **Go** to the following path: `mysiv3dproject/`
* Place the **LiBtorch ZIP** folder (from .zip) inside the **project** folder as follows `mydproject/_deps/libtorch/`:

The **CMake file will download this automatically for you**. Note: only a GPU is supported.  
Credits: https://github.com/prabhuomkar/pytorch-cpp/
  

#### Downloading and installing steps lippng:
* **[Download]()** 
* Under the lib directory,I included the lib file for PNG and ZLIB for windows, 
the CMake file will link against them during runtime.

## A CMake example
   
```cmake
include(copy_torch_dlls)
################# EXAMPLE 001 ########################
# TARGET
set(EXAMPLE_001_EXE torch_ray_sanity)
add_executable(${EXAMPLE_001_EXE} src/${EXAMPLE_001_EXE}.cpp)
set(raylib_VERBOSE 1)
target_link_libraries(${EXAMPLE_001_EXE} raylib  ${TORCH_LIBRARIES})
#target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_dpes/libtorch/include/")
target_include_directories(${EXAMPLE_001_EXE} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/_deps/libtorch/include/torch/csrc/api/")
set_target_properties(${EXAMPLE_001_EXE} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES)
copy_torch_dlls(${EXAMPLE_001_EXE})
```
 
## Inference
For inference, you have to copy all the **Libtorch DLLs** to the location of the executable file. For instance:
This is **done automatically** for you in the CMake file. 

````cmake
function(copy_torch_dlls TARGET_NAME)    
    list(GET CMAKE_MODULE_PATH 0 CMAKE_SCRIPT_DIR)    
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND}
                       -D "TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}"
                       -D "DESTINATION_DIR=$<TARGET_FILE_DIR:${TARGET_NAME}>" 
                       -P "${CMAKE_SCRIPT_DIR}/create_torch_dll_hardlinks.cmake")
endfunction()
````
 
## Contributing

Feel free to report issues during build or execution. We also welcome suggestions to improve the performance of this application.

## Author
Shlomo Kashani, Author of the book _Deep Learning Interviews_ www.interviews.ai: entropy@interviews.ai 

## Citation

If you find the code or trained models useful, please consider citing:

```
@misc{TorchRayLib++,
  author={Kashani, Shlomo},
  title={TorchRayLib++2020},
  howpublished={\url{https://github.com/QuantScientist/TorchRayLib/}},
  year={2020}
}
```

## License

- Copyright © [Shlomo](https://github.com/QuantScientist/)

# References
- https://github.com/raysan5/raylib
- https://github.com/RobLoach/raylib-cpp
- https://github.com/koba-jon/pytorch_cpp 
- https://www.jianshu.com/p/6fe9214431c6
- https://github.com/lsrock1/maskrcnn_benchmark.cpp
- https://gist.github.com/Con-Mi/4d92af62adb784a5353ff7cf19d6d099
- https://lernapparat.de/pytorch-traceable-differentiable/
- http://lernapparat.de/static/artikel/pytorch-jit-android/thomas_viehmann.pytorch_jit_android_2018-12-11.pdf
- https://github.com/walktree/libtorch-yolov3
