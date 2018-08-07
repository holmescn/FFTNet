#pragma once
#ifndef __FFTNET_BLOCK_H__
#define __FFTNET_BLOCK_H__

#include "cudnn_context.h"
#include "convolution.h"

class FFTNetBlock {
    cudnnActivationDescriptor_t _activation_descriptor;

    const cudnn::Context &_context;
    cudnn::Kernel _x_kernel;
    cudnn::Convolution _x_conv;
    cudnn::Array4f32 _x_l_weight;
    cudnn::Array4f32 _x_l_bias;
    cudnn::Array4f32 _x_r_weight;
    cudnn::Array4f32 _x_r_bias;
    cudnn::Kernel _out_kernel;
    cudnn::Convolution _out_conv;
    cudnn::Array4f32 _out_weight;
    cudnn::Array4f32 _out_bias;
    cudnn::Kernel *_h_kernel;
    cudnn::Convolution *_h_conv;
    cudnn::Array4f32 *_h_l_weight;
    cudnn::Array4f32 *_h_l_bias;
    cudnn::Array4f32 *_h_r_weight;
    cudnn::Array4f32 *_h_r_bias;
    int _local_condition_channels;
    int _shift;

    void _SplitByShift(const cudnn::Tensor4d &tensor, const cudnn::Array4f32 &input_data,
                        cudnn::Array4f32 &out_l, cudnn::Array4f32 &out_r) const;
    void _AddTensor(const cudnn::Tensor4d &tensor,
                    const cudnn::Array4f32 &data,
                    cudnn::Array4f32 &output);
    void _ReLU(const cudnn::Tensor4d &tensor, const cudnn::Array4f32 &data, cudnn::Array4f32 &output);
public:
    FFTNetBlock(const cudnn::Context &context,
                int in_channels, 
                int out_channels, 
                int shift,
                int local_condition_channels=-1);
    ~FFTNetBlock();

    FFTNetBlock(const FFTNetBlock& other) = delete;
    FFTNetBlock(FFTNetBlock&& other) = delete;
    FFTNetBlock& operator=(const FFTNetBlock& other)  = delete;
    FFTNetBlock& operator=(FFTNetBlock&& other) = delete;

    cudnn::Tensor4d GetOutputTensor(const cudnn::Tensor4d &tensor);
    void Forward(const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
                    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
    void Forward(const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
                    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
                    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
};

#endif // __FFTNET_BLOCK_H__