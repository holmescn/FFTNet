#pragma once
#ifndef __CUDNN_KERNEL_H__
#define __CUDNN_KERNEL_H__

#include <cassert>
#include "cudnn.h"
#include "data_type.h"
#include "tensor_format.h"

namespace cudnn {
    class Kernel {
        cudnnFilterDescriptor_t _descriptor;

        TensorFormat _format;
        DataType _dataType;
        int _out_channels, _in_channels, _kernel_height, _kernel_width;
    public:

    public:
        Kernel(int out_channels, int in_channels, int height, int width,
                TensorFormat format = TensorFormat::ChannelsFirst,
                DataType dataType = DataType::Float32);
        ~Kernel();

        Kernel(const Kernel& other) = delete;
        Kernel(Kernel&& other) = delete;
        Kernel& operator=(const Kernel& other) = delete;
        Kernel& operator=(Kernel&& other) = delete;

        explicit operator cudnnFilterDescriptor_t() const noexcept { return _descriptor; }

        template<typename T>
        T at(T* data, int kernel, int channel, int height, int width) const noexcept {
            assert( is_valid_type<T>(_dataType) );

            T *p = static_cast<T*>(data);
            return *(p + width +
                     _kernel_width * height +
                     _kernel_width * _kernel_height * channel +
                     _kernel_width * _kernel_height * _in_channels * kernel);
        }

        template<typename T>
        T& at(T* data, int kernel, int channel, int height, int width) noexcept {
            assert( is_valid_type<T>(_dataType) );

            T *p = static_cast<T*>(data);
            return *(p + width +
                     _kernel_width * height +
                     _kernel_width * _kernel_height * channel +
                     _kernel_width * _kernel_height * _in_channels * kernel);
        }

        size_t size() const noexcept {
            return size_of_data_type(_dataType) * _in_channels * _out_channels * _kernel_height * _kernel_width;
        }
    };
}

#endif // __CUDNN_KERNEL_H__
