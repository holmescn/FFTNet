#include "catch.hpp"

#include "fftnet_block.h"

TEST_CASE( "FFTNet block without h forward test.", "[cudnn::Convolution]" ) {
    cudnn::Context context;
    FFTNetBlock block(context, 1, 2, 3);
    block.x_l_conv1d.weight_data = 1.0;
    block.x_l_conv1d.bias_data = 1.0;
    block.x_r_conv1d.weight_data = 1.0;
    block.x_r_conv1d.bias_data = 1.0;
    block.out_conv1d.weight_data = 1.0;
    block.out_conv1d.bias_data = 1.0;

    cudnn::Tensor4d input_tensor(1, 1, 1, 10);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    auto output_tensor = block.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();

    REQUIRE(output_tensor.batch_size == 1);
    REQUIRE(output_tensor.n_channels == 2);
    REQUIRE(output_tensor.width == 7);

    block.Forward(input_tensor, input_data, output_tensor, output_data);
    cudaDeviceSynchronize();

    for (int b = 0; b < output_data.dim(0); ++b) {
        for (int ch = 0; ch < output_data.dim(1); ++ch) {
            for (int row = 0; row < output_data.dim(2); ++row) {
                for (int col = 0; col < output_data.dim(3); ++col) {
                    CHECK(output_data(b, ch, row, col) == Approx(9.0));
                }
            }
        }
    }
}

TEST_CASE( "FFTNet block with h forward test.", "[cudnn::Convolution]" ) {
    cudnn::Context context;
    FFTNetBlock block(context, 1, 2, 3, 1);

    block.x_l_conv1d.weight_data = 1.0;
    block.x_l_conv1d.bias_data = 1.0;
    block.x_r_conv1d.weight_data = 1.0;
    block.x_r_conv1d.bias_data = 1.0;

    block.h_l_conv1d->weight_data = 1.0;
    block.h_l_conv1d->bias_data = 1.0;
    block.h_r_conv1d->weight_data = 1.0;
    block.h_r_conv1d->bias_data = 1.0;

    block.out_conv1d.weight_data = 1.0;
    block.out_conv1d.bias_data = 1.0;

    cudnn::Tensor4d input_tensor(1, 1, 1, 10);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    cudnn::Tensor4d h_tensor(1, 1, 1, 20);
    auto h_data = h_tensor.CreateArray4f32();
    h_data = 1.0;

    auto output_tensor = block.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();

    REQUIRE(output_tensor.batch_size == 1);
    REQUIRE(output_tensor.n_channels == 2);
    REQUIRE(output_tensor.width == 7);

    block.Forward(input_tensor, input_data,
                  h_tensor, h_data,
                  output_tensor, output_data);
    cudaDeviceSynchronize();

    for (int b = 0; b < output_data.dim(0); ++b) {
        for (int ch = 0; ch < output_data.dim(1); ++ch) {
            for (int row = 0; row < output_data.dim(2); ++row) {
                for (int col = 0; col < output_data.dim(3); ++col) {
                    CHECK(output_data(b, ch, row, col) == Approx(17.0));
                }
            }
        }
    }
}
