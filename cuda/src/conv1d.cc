#include "conv1d.h"
#include "exception.h"

layers::Conv1d::Conv1d(
    const cudnn::Context &context,
    int in_channels, int out_channels, int kernel_size, int stride,
    int padding, bool bias)
: _context(context),
    _weights(in_channels, out_channels, 1, kernel_size),
    _convolution(context, _weights, 0, padding, 1, stride)
{

}

layers::Conv1d::~Conv1d()
{

}

cudnn::Array4f32 layers::Conv1d::CreateWeightsArray4f32()
{
    return _weights.CreateArray4f32();
}

void layers::Conv1d::operator()(
    const cudnn::Tensor4d &input_tensor, const cudnn::Array4f32 &input_data,
    const cudnn::Tensor4d &output_tensor, const cudnn::Array4f32 &output_data,
    const cudnn::Array4f32 &weights_data)
{
    _convolution.Forward(input_tensor, input_data, output_tensor, output_data, weights_data);
}
