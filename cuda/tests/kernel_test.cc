#include "catch.hpp"

#include "kernel.h"

TEST_CASE( "Kernel basic test.", "[cudnn:Kernel]" ) {
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

    cudnn::Kernel a_kernel(3, 3, 3, 3);

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    a_kernel.at(d_kernel, kernel, channel, row, column) = kernel_template[row][column];
                }
            }
        }
    }

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    float d = a_kernel.at(d_kernel, kernel, channel, row, column);
                    REQUIRE( h_kernel[kernel][channel][row][column] == d);
                }
            }
        }
    }
}
