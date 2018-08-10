#include <iostream>
#include "catch.hpp"

#include "fftnet.h"

TEST_CASE( "FFTNet test.", "[cudnn::FFTNet]" ) {
    FFTNet fftnet(11);

    INFO("Intialize weights and biases");
    for(auto iter = fftnet.layers.begin(); iter != fftnet.layers.end(); iter++) {
        auto p_layer = *iter;

        p_layer->x_l_conv1d.weight_data.InitializeWithZeros();
        p_layer->x_l_conv1d.bias_data.InitializeWithZeros();
        p_layer->x_r_conv1d.weight_data.InitializeWithZeros();
        p_layer->x_r_conv1d.bias_data.InitializeWithZeros();

        if (p_layer->h_l_conv1d != nullptr) {
            p_layer->h_l_conv1d->weight_data.InitializeWithZeros();
            p_layer->h_l_conv1d->bias_data.InitializeWithZeros();
            p_layer->h_r_conv1d->weight_data.InitializeWithZeros();
            p_layer->h_r_conv1d->bias_data.InitializeWithZeros();
        }

        p_layer->out_conv1d.weight_data.InitializeWithZeros();
        p_layer->out_conv1d.bias_data.InitializeWithZeros();
    }
    fftnet.linear.bias_data.InitializeWithZeros();

    INFO("Create input");
    cudnn::Tensor4d x_tensor(1, 1, 1, 3000);
    auto x_data = x_tensor.CreateArray4f32();
    x_data.InitializeWithZeros();

    INFO("Create hideen");
    cudnn::Tensor4d h_tensor(1, 1, 1, 3500);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    INFO("Create output");
    auto output_tensor = fftnet.CreateOutputTensor(x_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    INFO("Forward");
    fftnet.Forward(x_tensor, x_data, h_tensor, h_data, output_tensor, output_data);

    INFO("Verify");
    for(int b = 0; b < output_tensor.batch_size; b++) {
        for(int ch = 0; ch < output_tensor.n_channels; ch++) {
            for(int row = 0; row < output_tensor.height; row++) {
                for(int col = 0; col < output_tensor.width; col++) {
                    CHECK(output_data(b, ch, row, col) == Approx(0.0));
                }
            }
        }
    }
}

TEST_CASE("FFTNet initialize with HDF5 file.", "[cudnn:FFTNet]") {
    FFTNet fftnet(11, 256, 256, 80);
    fftnet.InitializeWithHDF5("/tmp/model_ckpt.h5");

    cudnn::Tensor4d x_tensor(1, 1, 1, 2500);
    auto x_data = x_tensor.CreateArray4f32();
    x_data = 1.0;

    cudnn::Tensor4d h_tensor(1, 80, 1, 3000);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    auto out_tensor = fftnet.CreateOutputTensor(x_tensor);
    auto out_data = out_tensor.CreateArray4f32();
    out_data.InitializeWithZeros();
    REQUIRE( out_tensor.n_channels == 256 );
    REQUIRE( out_tensor.width == 453 );

    fftnet.Forward(x_tensor, x_data, h_tensor, h_data, out_tensor, out_data);

    CHECK(out_data(0, 0, 0, 0) == Approx(-102.0696));
    CHECK(out_data(0, 0, 0, 1) == Approx(-102.0696));
    CHECK(out_data(0, 1, 0, 1) == Approx(-101.2076));
}