#include "catch.hpp"

#include "cudnn_context.h"

TEST_CASE( "cuDNN Context Wrapper", "[cudnn::Context]" ) {
    cudnn::Context context;
    
    SECTION( "convert to cudnnContext_t" ) {
        REQUIRE( (cudnnHandle_t)context != nullptr );
    }
}

TEST_CASE( "cuDNN Context Wrapper destructor", "[cudnn::Context]" ) {
    cudnn::Context *p_context = new cudnn::Context();
    delete p_context;

    REQUIRE( true );
}