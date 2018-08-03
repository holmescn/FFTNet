#include "cudnn_context.h"

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"


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