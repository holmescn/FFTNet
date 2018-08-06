#include "catch.hpp"

#include "tensor.h"

TEST_CASE( "Tensor4d is works", "[cudnn:Tensor4d]" ) {
    SECTION("constructor") {
        cudnn::Tensor4d t(3, 3, 3, 3);
        SUCCEED();
    }
    SECTION("create array4f32") {
        cudnn::Tensor4d t(3, 3, 3, 3);
        auto a = t.CreateArray4f32();
        SUCCEED();
    }
}
