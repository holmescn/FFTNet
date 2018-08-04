#include <cassert>
#include <stdexcept>
#include "convolution.h"

cudnn::Convolution::Convolution(const Context &context, const struct ConvolutionParameters &param,
                                const Tensor4d &inputTensor, const Tensor4d &outputTensor)
: _filter_data(nullptr), _workspace(nullptr)
{
    
}