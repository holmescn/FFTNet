#include "catch.hpp"
#include "linear.h"

TEST_CASE("Linear layer test", "[layers::Linear") {
    cudnn::Context context;
    layers::Linear linear(context, 5, 3);
    for(int row = 0; row < linear.n_rows; ++row) {
        for (int col = 0; col < linear.n_cols; ++col) {
            linear.weight(row, col) = 1.0;
        }
    }
    linear.bias_data = 1.0;

    const size_t height = 2, width = 4;
    cudnn::Tensor4d input_tensor(1, 5, height, width);
    auto input_data = input_tensor.CreateArray4f32();
    for(int batch = 0; batch < input_tensor.batch_size; ++batch) {
        for(int ch = 0; ch < input_tensor.n_channels; ++ch) {
            for (int h = 0; h < input_tensor.height; ++h) {
                for (int w = 0; w < input_tensor.width; ++w) {
                    input_data(batch, ch, h, w) = 1.0 * w + 10.0 * h + 100.0 * ch;
                }
            }
        }
    }

    auto output_tensor = linear.CreateOutputTensor(input_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    linear.Forward(input_tensor, input_data, output_tensor, output_data);
    REQUIRE( cudaDeviceSynchronize() == cudaSuccess);

    float pat[height][width] = {
        {1001.0, 1006.0, 1011.0, 1016.0},
        {1051.0, 1056.0, 1061.0, 1066.0},
    };
    for(int batch = 0; batch < output_tensor.batch_size; ++batch) {
        for(int ch = 0; ch < output_tensor.n_channels; ++ch) {
            for(int h = 0; h < height; h++) {
                for (int w = 0; w < width; ++w) {
                    CHECK(output_data(batch, ch, h, w) == Approx(pat[h][w]));
                }
            }
        }
    }
}