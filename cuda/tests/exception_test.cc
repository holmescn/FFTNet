#include "exception.h"

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"


TEST_CASE( "Assert cuDNN Success", "[assert_cudnn_success]" ) {
    SECTION("should success") {
        try {
            assert_cudnn_success(CUDNN_STATUS_SUCCESS);
        } catch (cudnn::Exception) {
            FAIL("should not fail.");
        }
    }

    SECTION("should fail") {
        try {
            assert_cudnn_success(CUDNN_STATUS_ALLOC_FAILED);
        } catch (cudnn::Exception) {
            SUCCEED("should success");
        }
    }
}

TEST_CASE( "Assert CUDA Success", "[assert_cuda_success]" ) {
    SECTION("should success") {
        try {
            assert_cuda_success(cudaSuccess);
        } catch (cuda::Exception) {
            FAIL("should not fail.");
        }
    }

    SECTION("should fail") {
        try {
            assert_cuda_success(cudaErrorUnknown);
        } catch (cuda::Exception) {
            SUCCEED("should success");
        }
    }
}