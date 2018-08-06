#pragma once
#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include "cudnn.h"
#include "data_type.h"
#include "tensor.h"
#include "kernel.h"
#include "array4d.h"
#include "cudnn_context.h"

namespace cudnn {
    enum class ConvolutionMode {
        CrossCorrelation = CUDNN_CROSS_CORRELATION,
        Convolution = CUDNN_CONVOLUTION,
        Invalid
    };

    class Convolution {
        cudnnConvolutionDescriptor_t _convolution_descriptor;
        cudnnConvolutionFwdAlgo_t _convolution_forward_algo;

        size_t _workspace_size;
        void *_workspace;
        const Context &_context;
        const Kernel &_kernel;
        void _SetConvolutionProperties(int pad_height,
                                        int pad_width,
                                        int height_stride,
                                        int width_stride,
                                        int dilation_height,
                                        int dilation_width,
                                        cudnn::ConvolutionMode mode,
                                        cudnn::DataType conpute_type);
        void _PrepareWorkspace(const Tensor4d &input_tensor, const Tensor4d &output_tensor);
    public:
        Convolution(const Context &context, const Kernel &kernel,
                    int pad_height = 0, int pad_width = 0,
                    int height_stride = 1, int width_stride = 1,
                    int dilation_height = 1, int dilation_width = 1);
        ~Convolution();
        Convolution(const Convolution& other) = delete;
        Convolution(Convolution&& other) = delete;
        Convolution& operator=(const Convolution& other) = delete;
        Convolution& operator=(Convolution&& other) noexcept = delete;

        void SetMode(ConvolutionMode mode);
        void SetPadding(int pad_h, int pad_w);
        void SetStride(int stride_h, int stride_w);

        void Forward(const Tensor4d &input_tensor, const Array4f32 &input_data,
                     const Tensor4d &output_tensor, const Array4f32 &output_data,
                     const Array4f32 &kernel_data);
        explicit operator cudnnConvolutionDescriptor_t() const noexcept { return _convolution_descriptor; }
    };
}

#endif // __CONVOLUTION_H__