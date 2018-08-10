#pragma once
#ifndef __FFTNET_H__
#define __FFTNET_H__

#include <vector>
#include <memory>
#include "cudnn_context.h"
#include "fftnet_block.h"
#include "linear.h"
#include "H5Cpp.h"

class FFTNet {
    const cudnn::Context _context;
    int _n_stacks;
    int _fft_channels;
    int _quantization_channels;
    int _local_condition_channels;

    std::vector<int> _window_shifts;

    void Forward_Impl(
        std::vector<std::shared_ptr<FFTNetBlock>>::iterator iter,
        const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
        const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
        const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
    cudnn::Tensor4d CreateOutputTensor_Impl(
        std::vector<std::shared_ptr<FFTNetBlock>>::iterator iter,
        const cudnn::Tensor4d &input_tensor);

    void _InitializeConv1DLayer(const H5::H5File &hdf_file,
                                int layer_index,
                                const layers::Conv1D &layer,
                                const char *layer_name);
    void _InitializeLinearLayer(const H5::H5File &hdf_file);
public:
    layers::Linear linear;
    std::vector<std::shared_ptr<FFTNetBlock>> layers;
    int receptive_field;

public:
    FFTNet(int n_stacks=11,
           int fft_channels=256,
           int quantization_channels=256,
           int local_condition_channels=-1);
    ~FFTNet();

    FFTNet(const FFTNet& other) = delete;
    FFTNet(FFTNet&& other) = delete;
    FFTNet& operator=(const FFTNet& other)  = delete;
    FFTNet& operator=(FFTNet&& other) = delete;

    void InitializeWithHDF5(const char *full_path);

    cudnn::Tensor4d CreateOutputTensor(const cudnn::Tensor4d &input_tensor);
    void Forward(const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
                 const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
                 const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data);
};

#endif // __FFTNET_H__