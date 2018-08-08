#include <iostream>
#include "catch.hpp"

#include "fftnet.h"

TEST_CASE( "FFTNet test.", "[cudnn::Convolution]" ) {
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
    cudaDeviceSynchronize();

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
