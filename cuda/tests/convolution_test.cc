#include "catch.hpp"

#include "convolution.h"

TEST_CASE( "Convolution constructor", "[cudnn::Convolution]" ) {
    cudnn::Context context;
    cudnn::Kernel kernel(3, 3, 3, 3);
    cudnn::Convolution convolution(context, kernel);
}

struct ConvolutionStruct {
    int pad_h = 0, pad_w = 0, stride_h = 0, stride_w = 0;
    int dilation_h = 0, dilation_w = 0;
    cudnnConvolutionMode_t mode;
    cudnnDataType_t compute_type;
};

struct ConvolutionStruct GetConvolutionStruct(const cudnn::Convolution &convolution)
{
    struct ConvolutionStruct ret;
    cudnnGetConvolution2dDescriptor(static_cast<cudnnConvolutionDescriptor_t>(convolution),
        &ret.pad_h,
        &ret.pad_w,
        &ret.stride_h,
        &ret.stride_w,
        &ret.dilation_h,
        &ret.dilation_w,
        &ret.mode,
        &ret.compute_type);
    return ret;
}

TEST_CASE( "Convolution set properties", "[cudnn::Convolution]" ) {
    cudnn::Context context;
    cudnn::Kernel kernel(3, 3, 3, 3);
    cudnn::Convolution convolution(context, kernel);

    SECTION("set mode") {
        convolution.SetMode(cudnn::ConvolutionMode::Convolution);
        auto ret = GetConvolutionStruct(convolution);
        REQUIRE(ret.mode == CUDNN_CONVOLUTION);
    }
    SECTION("set padding") {
        convolution.SetPadding(2, 3);
        auto ret = GetConvolutionStruct(convolution);
        REQUIRE(ret.pad_h == 2);
        REQUIRE(ret.pad_w == 3);
    }
    SECTION("set stride") {
        convolution.SetStride(2, 3);
        auto ret = GetConvolutionStruct(convolution);
        REQUIRE(ret.stride_h == 2);
        REQUIRE(ret.stride_w == 3);
    }
}

TEST_CASE( "Convolution forward", "[cudnn::Convolution]" ) {
    cudnn::Context context;

    cudnn::Kernel kernel(1, 1, 3, 3);
    auto kernel_data = kernel.CreateArray4f32();
    kernel_data = 1.0;

    cudnn::Tensor4d input_tensor(1, 1, 8, 8);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    cudnn::Tensor4d output_tensor(1, 1, 8, 8);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    cudnn::Convolution convolution(context, kernel, 1, 1);
    convolution.Forward(input_tensor, input_data, output_tensor, output_data, kernel_data);

    const float target[8][8] = {
        {4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 4.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {6.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 6.0},
        {4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 4.0},
    };

    for (int batch = 0; batch < output_data.dim(0); ++batch) {
        for (int ch = 0; ch < output_data.dim(1); ++ch) {
            for (int row = 0; row < output_data.dim(2); ++row) {
                for (int col = 0; col < output_data.dim(3); ++col) {
                    INFO("batch=" << batch << ", ch=" << ch << ", row=" << row << ", col=" << col);
                    CHECK(output_data(batch, ch, row, col) == Approx(target[row][col]));
                }
            }
        }
    }
}

TEST_CASE( "Conv1d", "[cudnn::Convolution]" ) {
    cudnn::Context context;

    cudnn::Kernel kernel(1, 1, 3, 3);
    auto kernel_data = kernel.CreateArray4f32();
    kernel_data = 1.0;

    cudnn::Tensor4d input_tensor(1, 1, 8, 8);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    cudnn::Tensor4d output_tensor(1, 1, 8, 8);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    cudnn::Convolution convolution(context, kernel, 1, 1);
    convolution.Forward(input_tensor, input_data, output_tensor, output_data, kernel_data);

}