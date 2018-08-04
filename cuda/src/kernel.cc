#include "kernel.h"
#include "exception.h"

cudnn::Kernel::Kernel(int out_channels, int in_channels, int height, int width,
                     TensorFormat format,
                     DataType dataType)
: _out_channels(out_channels), _in_channels(in_channels),
  _kernel_height(height), _kernel_width(width), _format(format), _dataType(dataType)
{
    assert_cudnn_success( cudnnCreateFilterDescriptor(&_descriptor) );
    assert_cudnn_success( cudnnSetFilter4dDescriptor(_descriptor,
        static_cast<cudnnDataType_t>(dataType),
        static_cast<cudnnTensorFormat_t>(format),
        out_channels,
        in_channels,
        height,
        width
    ));
}

cudnn::Kernel::~Kernel()
{
    assert_cudnn_success( cudnnDestroyFilterDescriptor(_descriptor) );
}