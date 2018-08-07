#include "catch.hpp"

#include "fftnet_block.h"

TEST_CASE( "Convolution constructor", "[cudnn::Convolution]" ) {
    cudnn::Context context;
    FFTNetBlock block(context, 1, 2, 3);

    cudnn::Tensor4d input_tensor(1, 1, 1, 10);
    auto input_data = input_tensor.CreateArray4f32();
    auto output_tensor = block.GetOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();

    block.Forward(input_tensor, input_data, output_tensor, output_data);
}
