#pragma once
#ifndef __CUDNN_TENSOR_H__
#define __CUDNN_TENSOR_H__

#include "cudnn.h"
#include "data_type.h"
#include "tensor_format.h"

namespace cudnn {
    class Tensor4d {
        cudnnTensorDescriptor_t _descriptor;
        bool _initialized;

        size_t batch_stride, channels_stride, height_stride, width_stride;
    public:
        TensorFormat format;
        DataType dataType;
        int batch_size, n_channels, height, width;

    public:
        Tensor4d();
        ~Tensor4d();
        Tensor4d(const Tensor4d& other) = delete;
        Tensor4d(Tensor4d&& other) = delete;
        Tensor4d& operator=(const Tensor4d& other) = delete;
        Tensor4d& operator=(Tensor4d&& other) = delete;

        explicit operator cudnnTensorDescriptor_t() const noexcept { return _descriptor; }

        void initialize();

        template<typename T>
        T at(void* data, int batch, int channel, int height, int width) const noexcept {
            uint8_t *p = static_cast<uint8_t*>(data);
            T *r = static_cast<T*>(p + width * width_stride + height * height_stride + channel * channels_stride + batch * batch_stride);
            return *r;
        }

        template<typename T>
        T& at(void* data, int batch, int channel, int height, int width) noexcept {
            uint8_t *p = static_cast<uint8_t*>(data);
            T *r = static_cast<T*>(p + width * width_stride + height * height_stride + channel * channels_stride + batch * batch_stride);
            return *r;
        }

        size_t size() const noexcept { return size_of_data_type(dataType) * batch_size * n_channels * height * width; }
    };
}

#endif // __CUDNN_TENSOR_H__
