#include "catch.hpp"

#include "conv1d.h"

TEST_CASE( "Convolution forward", "[cudnn::Convolution]" ) {
    cudnn::Context context;

    layers::Conv1D conv1d(context, 1, 1, 1);
    conv1d.weight_data = 2.0;
    conv1d.bias_data = 1.0;

    cudnn::Tensor4d input_tensor(1, 1, 1, 10);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    auto output_tensor = conv1d.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data = 0.0;

    conv1d.Forward(input_tensor, input_data, output_tensor, output_data);
    cudaDeviceSynchronize();

    for(int b = 0; b < output_data.dim(0); ++b) {
        for (int ch = 0; ch < output_data.dim(1); ++ch) {
            for (int row = 0; row < output_data.dim(2); ++row) {
                for (int col = 0; col < output_data.dim(3); ++col) {
                    float x = output_data(b, ch, row, col);
                    CHECK( x == Approx(3.0) );
                }
            }
        }
    }
}
