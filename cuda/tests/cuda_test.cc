#include <iostream>

#include "catch.hpp"
#include "cudnn.h"

TEST_CASE( "CUDA & cuDNN Routine tests.", "[cuda][cudnn]" ) {
    SECTION("cudaMalloc & cudaMemset") {
        float* data = nullptr;
        REQUIRE(cudaMalloc(&data, 1024) == cudaSuccess);
        REQUIRE(cudaMemset(data, 0, 1024) == cudaSuccess);
        REQUIRE(cudaFree(data) == cudaSuccess);
    }

    SECTION("cudaMallocManaged & cudaMemset") {
        float* data = nullptr;
        REQUIRE(cudaMallocManaged(&data, 1024) == cudaSuccess);
        REQUIRE(cudaMemset(data, 0, 1024) == cudaSuccess);
        REQUIRE(cudaFree(data) == cudaSuccess);
    }
}

TEST_CASE("cuDNN convolution forward.", "[cudnn]") {
    cudnnHandle_t cudnn;
    REQUIRE(cudnnCreate(&cudnn) == CUDNN_STATUS_SUCCESS);

    int batch_size = 1;
    int channels = 1;
    int height = 8;
    int width = 8;

    cudnnTensorDescriptor_t input_descriptor;
    REQUIRE(cudnnCreateTensorDescriptor(&input_descriptor) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          batch_size,
                                          channels,
                                          height,
                                          width) == CUDNN_STATUS_SUCCESS);

    cudnnTensorDescriptor_t output_descriptor;
    REQUIRE(cudnnCreateTensorDescriptor(&output_descriptor) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        batch_size,
                                        channels,
                                        height,
                                        width) == CUDNN_STATUS_SUCCESS);

    int out_channels = 1;
    int in_channels = 1;
    int kernel_height = 3;
    int kernel_width = 3;
    cudnnFilterDescriptor_t kernel_descriptor;
    REQUIRE(cudnnCreateFilterDescriptor(&kernel_descriptor) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        out_channels,
                                        in_channels,
                                        kernel_height,
                                        kernel_width) == CUDNN_STATUS_SUCCESS);

    cudnnConvolutionDescriptor_t convolution_descriptor;
    REQUIRE(cudnnCreateConvolutionDescriptor(&convolution_descriptor) == CUDNN_STATUS_SUCCESS);
    REQUIRE(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                            /*pad_height=*/1,
                                            /*pad_width=*/1,
                                            /*vertical_stride=*/1,
                                            /*horizontal_stride=*/1,
                                            /*dilation_height=*/1,
                                            /*dilation_width=*/1,
                                            /*mode=*/CUDNN_CROSS_CORRELATION,
                                            /*computeType=*/CUDNN_DATA_FLOAT) == CUDNN_STATUS_SUCCESS);

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    REQUIRE(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                            /*memoryLimitInBytes=*/0,
                                            &convolution_algorithm) == CUDNN_STATUS_SUCCESS);
    size_t workspace_bytes = 0;
    REQUIRE(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    convolution_algorithm,
                                                    &workspace_bytes) == CUDNN_STATUS_SUCCESS);
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB" << std::endl;

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float* d_input = nullptr;
    REQUIRE(cudaMallocManaged(&d_input, image_bytes) == cudaSuccess);
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    size_t index = col + row * width + ch * width * height + batch * width * height * channels;
                    d_input[index] = 1.0;
                }
            }
        }
    }

    float* d_output = nullptr;
    REQUIRE(cudaMallocManaged(&d_output, image_bytes) == cudaSuccess);
    REQUIRE(cudaMemset(d_output, 0, image_bytes) == cudaSuccess);

    size_t kernel_size = out_channels * in_channels * kernel_height * kernel_width * sizeof(float);
    float* d_kernel = nullptr;
    REQUIRE(cudaMallocManaged(&d_kernel, kernel_size) == cudaSuccess);
    for (int ker = 0; ker < out_channels; ++ker) {
        for (int ch = 0; ch < in_channels; ++ch) {
            for (int row = 0; row < kernel_height; ++row) {
                for (int col = 0; col < kernel_width; ++col) {
                    size_t index = col + row * kernel_width + ch * kernel_width * kernel_height
                                        + ker * kernel_width * kernel_height * in_channels;
                    d_kernel[index] = 1.0;
                }
            }
        }
    }

    const float alpha = 1, beta = 0;
    REQUIRE(cudnnConvolutionForward(cudnn,
                                    &alpha,
                                    input_descriptor,
                                    d_input,
                                    kernel_descriptor,
                                    d_kernel,
                                    convolution_descriptor,
                                    convolution_algorithm,
                                    d_workspace,
                                    workspace_bytes,
                                    &beta,
                                    output_descriptor,
                                    d_output) == CUDNN_STATUS_SUCCESS);
    cudaDeviceSynchronize();

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

    for (int batch = 0; batch < batch_size; ++batch) {
        for (int ch = 0; ch < channels; ++ch) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    size_t index = col + row * width + ch * width * height + batch * width * height * channels;
                    INFO("batch=" << batch << ", ch=" << ch << ", row=" << row << ", col=" << col);
                    CHECK(d_output[index] == Approx(target[row][col]));
                }
            }
        }
    }

    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}