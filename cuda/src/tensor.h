#pragma once
#ifndef __CUDNN_TENSOR_H__
#define __CUDNN_TENSOR_H__

#include "cudnn.h"

namespace cudnn {
    enum class TensorFormat {
        ChannelsFirst = CUDNN_TENSOR_NCHW,
        ChannelsLast = CUDNN_TENSOR_NHWC
    };

    enum class DataType {
        Float = CUDNN_DATA_FLOAT,
        Double = CUDNN_DATA_DOUBLE,
        Half = CUDNN_DATA_HALF
    };

    class Tensor4d {
        cudnnTensorDescriptor_t _descriptor;
        TensorFormat _format;
        DataType _dataType;

        void *_data;
        size_t _size;
    public:
        Tensor4d(TensorFormat format, DataType dataType,
                int batch_size, int n_channels, int height, int width);

        ~Tensor4d();

        Tensor4d(const Tensor4d& other);
        Tensor4d(Tensor4d&& other);
        Tensor4d& operator=(const Tensor4d& other);
        Tensor4d& operator=(Tensor4d&& other);
    };
}

#endif // __CUDNN_TENSOR_H__
