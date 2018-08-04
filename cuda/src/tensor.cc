#include <cassert>
#include <stdexcept>
#include <iostream>
#include "tensor.h"
#include "exception.h"

cudnn::Tensor4d::Tensor4d()
: format(TensorFormat::ChannelsFirst), dataType(DataType::Float32),
  batch_size(1), n_channels(1), height(1), width(1),
  batch_stride(0), channels_stride(0), height_stride(0), width_stride(0),
  _initialized(false)
{}

cudnn::Tensor4d::~Tensor4d()
{
    if (_initialized) {
        assert_cudnn_success( cudnnDestroyTensorDescriptor(_descriptor) );
    }
}

void cudnn::Tensor4d::initialize()
{
    assert(! _initialized);
    assert(batch_size > 0);
    assert(n_channels > 0);
    assert(height > 0);
    assert(width > 0);

    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );

    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
        static_cast<cudnnTensorFormat_t>(format),
        static_cast<cudnnDataType_t>(dataType),
        batch_size, n_channels, height, width
    ) );

    switch(format) {
        case TensorFormat::ChannelsFirst:
            width_stride = size_of_data_type(dataType);
            height_stride = width_stride * width;
            channels_stride = height_stride * height;
            batch_stride = channels_stride * n_channels;
            break;
        case TensorFormat::ChannelsLast:
            channels_stride = size_of_data_type(dataType);
            width_stride = channels_stride * n_channels;
            height_stride = width_stride * width;
            batch_stride = height_stride * height;
            break;
        default:
            throw std::runtime_error("Unknown tensor format.");
    }

    _initialized = true;
}
