#include "catch.hpp"

#include "array4d.h"
#include "kernel.h"
#include "tensor.h"

TEST_CASE("Array4f32 construct", "[cudnn::Array4f32") {
    SECTION("with raw dims") {
        cudnn::Array4f32 A(2, 2, 2, 2);
        SUCCEED();
    }
}

TEST_CASE( "Array4f32 assign and retrive.", "[cudnn:Array4f32]" ) {
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

    cudnn::Array4f32 a_kernel(3, 3, 3, 3);

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    a_kernel(kernel, channel, row, column) = kernel_template[row][column];
                }
            }
        }
    }

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    float d = a_kernel(kernel, channel, row, column);
                    REQUIRE( h_kernel[kernel][channel][row][column] == d);
                }
            }
        }
    }
}
