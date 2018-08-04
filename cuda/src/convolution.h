#pragma once
#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "cudnn.h"
#include "data_type.h"
#include "tensor.h"
#include "cudnn_context.h"

namespace cudnn {
    enum class ConvolutionMode {
        CrossCorrelation = CUDNN_CROSS_CORRELATION,
        Convolution = CUDNN_CONVOLUTION
    };

    struct ConvolutionParameters {
        DataType DataType;
        TensorFormat tensorFormat;
        ConvolutionMode mode;
        int in_channels, out_channels, kernel_height, kernel_width;
        int pad_height, pad_width;
        int height_stride, width_stride;
        int dilation_height, dilation_width;
    };

    class Convolution {
        cudnnFilterDescriptor_t _filter_descriptor;
        cudnnConvolutionDescriptor_t _convolution_descriptor;
        cudnnConvolutionFwdAlgo_t _convolution_forward_algo;

        size_t workspace_size;

        void *_workspace;
        void *_filter_data;
    public:
        Convolution(const Context &context, const struct ConvolutionParameters &param,
                    const Tensor4d &inputTensor, const Tensor4d &outputTensor);
        ~Convolution();
        Convolution(const Convolution& other);
        Convolution(Convolution&& other);
        Convolution& operator=(const Convolution& other); // copy assignment
        Convolution& operator=(Convolution&& other) noexcept; // move assignment

        void Forward(const Tensor4d &inputTensor, const Tensor4d &outputTensor);
    };
}

#endif // __CONVOLUTION_H__