#include "catch.hpp"

#include "fftnet.h"

TEST_CASE( "FFTNet block without h forward test.", "[cudnn::Convolution]" ) {
    FFTNet fftnet(3);
}
