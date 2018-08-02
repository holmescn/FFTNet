#include "tensor.h"
#include "exception.h"

cudnn::Tensor4d::Tensor4d(TensorFormat format, DataType dataType,
                          int batch_size, int n_channels, int height, int width)
: _format(format), _dataType(dataType), _data(nullptr), _size(0)
{
    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );
    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
                                static_cast<cudnnTensorFormat_t>(format),
                                static_cast<cudnnDataType_t>(dataType),
                                batch_size,
                                n_channels,
                                height,
                                width) );
}

cudnn::Tensor4d::~Tensor4d()
{
    cudnnDestroyTensorDescriptor(_descriptor);
    if (_data != nullptr) {
        cudaFree(_data);
    }
}

cudnn::Tensor4d::Tensor4d(const Tensor4d& other)
: _format(other._format), _dataType(other._dataType), _data(nullptr), _size(0)
{
    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );

    cudnnDataType_t dataType;
    int batch_size = 0, n_channels = 0, height = 0, width = 0,
        nStride = 0, cStride = 0, hStride = 0, wStride = 0;
    assert_cudnn_success( cudnnGetTensor4dDescriptor(other._descriptor,
                                &dataType,
                                &batch_size,
                                &n_channels,
                                &height,
                                &width,
                                &nStride,
                                &cStride,
                                &hStride,
                                &wStride) );

    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
                                static_cast<cudnnTensorFormat_t>(other._format),
                                static_cast<cudnnDataType_t>(other._dataType),
                                batch_size,
                                n_channels,
                                height,
                                width) );

    if (other._data != nullptr) {
        assert_cuda_success( cudaMallocManaged(&this->_data, other._size) );
        assert_cuda_success( cudaMemcpy(this->_data, other._data, other._size, cudaMemcpyDefault) );
        this->_size = other._size;
    }
}

cudnn::Tensor4d::Tensor4d(Tensor4d&& other)
: _format(other._format), _dataType(other._dataType), _data(nullptr), _size(0)
{
    assert_cudnn_success( cudnnCreateTensorDescriptor(&_descriptor) );

    cudnnDataType_t dataType;
    int batch_size = 0, n_channels = 0, height = 0, width = 0,
        nStride = 0, cStride = 0, hStride = 0, wStride = 0;
    assert_cudnn_success( cudnnGetTensor4dDescriptor(other._descriptor,
                                &dataType,
                                &batch_size,
                                &n_channels,
                                &height,
                                &width,
                                &nStride,
                                &cStride,
                                &hStride,
                                &wStride) );

    assert_cudnn_success( cudnnSetTensor4dDescriptor(_descriptor,
                                static_cast<cudnnTensorFormat_t>(other._format),
                                static_cast<cudnnDataType_t>(other._dataType),
                                batch_size,
                                n_channels,
                                height,
                                width) );

    if (other._data != nullptr) {
        this->_data = other._data;
        this->_size = other._size;
        other._data = nullptr;
        other._size = 0;
    }
}

cudnn::Tensor4d::Tensor4d&
cudnn::Tensor4d::operator=(const cudnn::Tensor4d& other)
{
    return *this = Tensor4d(other);
}

cudnn::Tensor4d::Tensor4d&
cudnn::Tensor4d::operator=(cudnn::Tensor4d&& other)
 {
     // TODO
     return *this;
 }
