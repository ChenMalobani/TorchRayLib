#include <iostream>
#include <chrono>
#include <torch/script.h> // One-stop header.
//#include <torch/torch.h>
#include <iostream>
#include <typeinfo>
//#include <png++/png.hpp>
#include <thread>
#include <future>
#include "raylib.h"

using namespace std;
using namespace std::chrono;
class VisionUtils {
public:
    VisionUtils();
    static void tensorDIMS(const torch::Tensor &tensor);
    torch::Tensor rayImageToTorch(const Image &image, c10::Device &device);
    Image torchToRayImage(torch::Tensor &tensor_);
//    torch::Tensor pngToTorch(png::image<png::rgb_pixel> &image, c10::Device &device);
//    png::image<png::rgb_pixel> torchToPng(torch::Tensor &tensor_);
//    torch::Tensor pngToTorchRGBA(png::image<png::rgba_pixel> &image, c10::Device &device);
//    png::image<png::rgba_pixel> torchToPngRGBA(torch::Tensor &tensor_);

};

VisionUtils::VisionUtils() {}

void VisionUtils::tensorDIMS(const torch::Tensor &tensor) {
    auto t0 = tensor.size(0);
    auto s = tensor.sizes();
    cout << "D=:" << s << "\n";
}

torch::Tensor VisionUtils::rayImageToTorch(const Image &image, c10::Device &device){
    size_t width = image.width;
    size_t height = image.height;
    auto pointer = new unsigned char[width * height * 3];

    const unsigned char* imagePointer = (unsigned char*)image.data;
    for (size_t j = 0; j < height; j++){
        size_t noAlpha = 0;
        for (size_t i = 0; i < width; i++){
            pointer[j * width * 3 + i * 3 + 0] = imagePointer[j*width+noAlpha]; ++noAlpha;
            pointer[j * width * 3 + i * 3 + 1] = imagePointer[j*width+noAlpha]; ++noAlpha;
            pointer[j * width * 3 + i * 3 + 2] = imagePointer[j*width+noAlpha]; noAlpha+=2; //we need to jump over alpha channels
        }
    }
    torch::Tensor tensor = torch::from_blob(pointer, {(int)width, (int)width, 3}, torch::kU8)
            .clone().permute({2, 0, 1}).to(device);  // copy
    delete[] pointer;
    return tensor;
}

Image VisionUtils::torchToRayImage(torch::Tensor &tensor_){
    torch::Tensor tensor = tensor_.squeeze().detach().cpu().permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
    tensor = tensor.clamp(0, 255);
    tensor = tensor.to(torch::kU8);
    size_t width = tensor.size(1);
    size_t height = tensor.size(0);
    auto torchPointer = tensor.data_ptr<unsigned char>();

//    const unsigned char* imagePointer = (unsigned char*)torchPointer;
    auto imagePointer = reinterpret_cast<unsigned char*>((torchPointer));

    for (size_t j = 0; j < height; j++){
        size_t noAlpha = 0;
        for (size_t i = 0; i < width; i++){
            imagePointer[j*width+noAlpha] = torchPointer[j * width * 3 + i * 3 + 0]; ++noAlpha;
            imagePointer[j*width+noAlpha] = torchPointer[j * width * 3 + i * 3 + 1]; ++noAlpha;
            imagePointer[j*width+noAlpha] = torchPointer[j * width * 3 + i * 3 + 2]; noAlpha+=1;
        }
    }
    return Image{
            imagePointer,
            (int)width,
            (int)height,
            1, //that line is mipmaps, keep as 1
            UNCOMPRESSED_R8G8B8}; //its an enum specyfing forma, 8 bit R, 8 bit G, 8 bit G, no alpha
}

//return Image{ data //u need to pass there pointer to data
//width,
//        height,
//1, //that line is mipmaps, keep as 1
//UNCOMPRESSED_R8G8B8}; //its an enum specyfing forma, 8 bit R, 8 bit G, 8 bit G, no alpha

//Adapted from https://github.com/koba-jon/pytorch_cpp/blob/master/utils/visualizer.cpp
//torch::Tensor VisionUtils::pngToTorch(png::image<png::rgb_pixel> &image, c10::Device &device){
//    size_t width = image.get_width();
//    size_t height = image.get_height();
//    auto pointer = new unsigned char[width * height * 3];
//    for (size_t j = 0; j < height; j++){
//        for (size_t i = 0; i < width; i++){
//            pointer[j * width * 3 + i * 3 + 0] = image[j][i].red;
//            pointer[j * width * 3 + i * 3 + 1] = image[j][i].green;
//            pointer[j * width * 3 + i * 3 + 2] = image[j][i].blue;
//        }
//    }
//    torch::Tensor tensor = torch::from_blob(pointer, {image.get_height(), image.get_width(), 3}, torch::kU8)
//            .clone().permute({2, 0, 1}).to(device);  // copy
////    .clone().to(torch::kFloat32).permute({ 2, 0, 1 }).div_(255).to(device);
////    torch::Tensor tensor = torch::from_blob(pointer, {image.get_height(), image.get_width(), 3}, torch::kUInt8).clone();  // copy
////    tensor = tensor.permute({2, 0, 1});  // {H,W,C} ===> {C,H,W}
//    delete[] pointer;
//    return tensor;
//}
//
//png::image<png::rgb_pixel> VisionUtils::torchToPng(torch::Tensor &tensor_){
//    torch::Tensor tensor = tensor_.squeeze().detach().cpu().permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
//    tensor = tensor.clamp(0, 255);
//    tensor = tensor.to(torch::kU8);
//    size_t width = tensor.size(1);
//    size_t height = tensor.size(0);
//    auto pointer = tensor.data_ptr<unsigned char>();
//    png::image<png::rgb_pixel> image(width, height);
//    for (size_t j = 0; j < height; j++){
//        for (size_t i = 0; i < width; i++){
//            image[j][i].red = pointer[j * width * 3 + i * 3 + 0];
//            image[j][i].green = pointer[j * width * 3 + i * 3 + 1];
//            image[j][i].blue = pointer[j * width * 3 + i * 3 + 2];
//        }
//    }
//    return image;
//}
//
//
//
//torch::Tensor VisionUtils::pngToTorchRGBA(png::image<png::rgba_pixel> &image, c10::Device &device){
//    size_t width = image.get_width();
//    size_t height = image.get_height();
//    auto pointer = new unsigned char[width * height * 4];
//    for (size_t j = 0; j < height; j++){
//        for (size_t i = 0; i < width; i++){
//            pointer[j * width * 4 + i * 4 + 0] = image[j][i].red;
//            pointer[j * width * 4 + i * 4 + 1] = image[j][i].green;
//            pointer[j * width * 4 + i * 4 + 2] = image[j][i].blue;
//            pointer[j * width * 4 + i * 4 + 3] = image[j][i].alpha;
//        }
//    }
//    torch::Tensor tensor = torch::from_blob(pointer, {image.get_height(), image.get_width(), 3}, torch::kUInt8).clone().to(device);  // copy
//    tensor = tensor.permute({2, 0, 1});  // {H,W,C} ===> {C,H,W}
//    delete[] pointer;
//    return tensor;
//}
//
//png::image<png::rgba_pixel> VisionUtils::torchToPngRGBA(torch::Tensor &tensor_){
//    torch::Tensor tensor = tensor_.detach().cpu().permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
////    torch::Tensor tensor = tensor_.permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
//    size_t width = tensor.size(1);
//    size_t height = tensor.size(0);
//    auto pointer = tensor.data_ptr<unsigned char>();
//    png::image<png::rgba_pixel> image(width, height);
//    for (size_t j = 0; j < height; j++){
//        for (size_t i = 0; i < width; i++){
//            image[j][i].red = pointer[j * width * 4 + i * 4 + 0];
//            image[j][i].green = pointer[j * width * 4 + i * 4 + 1];
//            image[j][i].blue = pointer[j * width * 4 + i * 4 + 2];
//            image[j][i].alpha = pointer[j * width * 4 + i * 4 + 3];
//        }
//    }
//    return image;
//}

//at::Tensor matToTensor(cv::Mat frame, int h, int w, int c) {
//    cv::cvtColor(frame, frame, CV_BGR2RGB);
//    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
//    auto input_tensor = torch::from_blob(frame.data, {1, h, w, c});
//    input_tensor = input_tensor.permute({0, 3, 1, 2});
//
//    torch::DeviceType device_type = torch::kCPU;
////    if (torch::cuda::is_available()) {
//    device_type = torch::kCUDA;
////    }
//    input_tensor = input_tensor.to(device_type);
//    return input_tensor;
//}


//cv::Mat VisionUtils::tensorToOpenCv(at::Tensor out_tensor, int h, int w, int c) {
//
//    out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
//    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
//    out_tensor = out_tensor.to(torch::kCPU);
//    cv::Mat resultImg(h, w, CV_8UC3);
//    // cv::Mat resultImg(h, w, CV_8UC1);
//    std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
//    return resultImg;
//}
//
//
//void VisionUtils::processVideo(const std::string &modelName, c10::Device device, const std::string &vidName) {
//    int kCHANNELS = 3;
//    std::thread::id this_id = std::this_thread::get_id();
//    stringstream ss_id;
//    ss_id << this_id;
//    string this_id_str = ss_id.str();
//    std::cout << "thread: " << this_id << std::endl;
//    std::cout << " >>> Loading " << modelName << std::endl;
//    auto module = torch::jit::load(modelName, device);
////    module->to(at::kCUDA);
//    if (!std::ifstream(modelName)) {
//        std::cout << "ERROR: Could not open the required tmodule file from path: "
//                  << modelName << std::endl;
//    }
//    assert(module != nullptr);
//    cv::VideoCapture video_reader;
//    cv::Mat frame;
//    if (!video_reader.open(vidName)) {
//        cout << "cannot open video " << endl;
//    }
//    long frame_h = int(video_reader.get(cv::CAP_PROP_FRAME_HEIGHT));
//    long frame_w = int(video_reader.get(cv::CAP_PROP_FRAME_WIDTH));
//    long nb_frames = int(video_reader.get(cv::CAP_PROP_FRAME_COUNT));
//    cv::VideoWriter videoWriter;
////    videoWriter.open(cv::VideoWriter::fourcc('I', 'Y', 'U', 'V'), 25, cv::Size(frame_w, frame_h));
//    cout << "WIDTH " << frame_w << "\n";
//    cout << "HEIGHT " << frame_h << "\n";
//    cout << "NB FRAMES " << nb_frames << "\n";
////    cv::Size imageSize = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
//
//    for (long num_frames = 0; num_frames < nb_frames; num_frames++) {
//        video_reader >> frame;
//        cv::imshow("orig", frame);
//
//        auto input_tensor = matToTensor(frame, frame_h, frame_w, kCHANNELS);
////        tensorDIMS(input_tensor); // 1/_3/_720/_1280
//        torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
////        tensorDIMS(out_tensor); // 1/_3/_720/_1280 , D=:[1, 64, 720, 1280] D=:[1, 512, 22, 40]
////        juce::Image juceImage=tensor_to_image(out_tensor);
////        savePNG(juceImage,);
////        out_tensor=image_to_tensor(juceImage);
//        auto resultImg = tensorToOpenCv(out_tensor, out_tensor.sizes()[2], out_tensor.sizes()[3],
//                                        kCHANNELS); // D=:[1, 3, 720, 1280]
////        auto resultImg= tensorToOpenCv(out_tensor, frame_h, frame_w, kCHANNELS);
//        // This is necessary to correctly apply softmax,
//        // last dimension should represent logits
////        auto full_prediction_flattned = full_prediction.squeeze(0)
////                .view({21, -1})
////                .transpose(0, 1);
////
////        // Converting logits to probabilities
////        auto softmaxed = torch::softmax(full_prediction_flattned).transpose(0, 1);
//
//        stringstream ss_nb;
//        ss_nb << num_frames;
//        string num_frames_str = ss_nb.str();
//
//        cv::putText(resultImg,
//                    this_id_str,
//                    cv::Point(50, 50), // Coordinates
//                    cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
//                    2.0, // Scale. 2.0 = 2x bigger
//                    cv::Scalar(255, 125, 0), // BGR Color
//                    1 // Line Thickness (Optional)
//        );
//
//        cv::putText(resultImg,
//                    num_frames_str,
//                    cv::Point(100, 100), // Coordinates
//                    cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
//                    1.0, // Scale. 2.0 = 2x bigger
//                    cv::Scalar(255, 255, 0), // BGR Color
//                    1 // Line Thickness (Optional)
//        );
//        cv::imshow(this_id_str, resultImg);
//        char key = cv::waitKey(10);
//        if (key == 27) // ESC
//            break;
//    }
//}