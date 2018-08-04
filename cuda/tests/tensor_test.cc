#include "catch.hpp"

#include "tensor.h"

TEST_CASE( "Tensor4d is works", "[cudnn:Tensor4d]" ) {
    const float kernel_template[3][3] = {
        {1,  1, 1},
        {1, -8, 1},
        {1,  1, 1}
    };

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }

    float* d_kernel = nullptr;
    if( cudaMallocManaged(&d_kernel, sizeof(h_kernel)) != CUDNN_STATUS_SUCCESS ) {
        FAIL();
    }

    cudnn::Tensor4d tensor;
    tensor.batch_size = 3;
    tensor.n_channels = 3;
    tensor.height = 3;
    tensor.width = 3;
    tensor.initialize();

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    tensor.at<float>(d_kernel, kernel, channel, row, column) = kernel_template[row][column];
                }
            }
        }
    }

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    float d = tensor.at<float>(d_kernel, kernel, channel, row, column);
                    REQUIRE( h_kernel[kernel][channel][row][column] == d);
                }
            }
        }
    }
}
