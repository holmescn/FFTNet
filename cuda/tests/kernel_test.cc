#include "catch.hpp"

#include "kernel.h"

TEST_CASE( "Kernel basic test.", "[cudnn:Kernel]" ) {
    SECTION("constructor") {
        cudnn::Kernel k(3, 3, 3, 3);
        SUCCEED();
    }

    SECTION("create Array4f32") {
        cudnn::Kernel k(3, 3, 3, 3);
        auto a = k.CreateArray4f32();
        SUCCEED();
    }
}
