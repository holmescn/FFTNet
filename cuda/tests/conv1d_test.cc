#include "catch.hpp"

#include "conv1d.h"

TEST_CASE( "Conv1d", "[cudnn::Convolution]" ) {
    cudnn::Context context;

    cudnn::Tensor4d input_tensor(1, 1, 1, 8);
    auto input_data = input_tensor.CreateArray4f32();
    input_data = 1.0;

    cudnn::Tensor4d output_tensor(1, 1, 1, 8);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    layers::Conv1d conv1d(context, /* in_channels =*/1, /* out_channels =*/1, /* kernel_size =*/1);
    auto weights_data = conv1d.CreateWeightsArray4f32();
    weights_data = 2.0;

    conv1d(input_tensor, input_data, output_tensor, output_data, weights_data);

    for (int batch = 0; batch < output_data.dim(0); ++batch) {
        for (int ch = 0; ch < output_data.dim(1); ++ch) {
            for (int row = 0; row < output_data.dim(2); ++row) {
                for (int col = 0; col < output_data.dim(3); ++col) {
                    INFO("batch=" << batch << ", ch=" << ch << ", row=" << row << ", col=" << col);
                    CHECK(output_data(batch, ch, row, col) == Approx(2.0));
                }
            }
        }
    }
}
