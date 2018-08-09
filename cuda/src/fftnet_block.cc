#include <iostream>
#include "fftnet_block.h"
#include "exception.h"

FFTNetBlock::FFTNetBlock(
    const cudnn::Context &context,
    int in_channels,
    int out_channels,
    int shift,
    int local_condition_channels)
: _context(context),
  _relu(context),
  _local_condition_channels(local_condition_channels),
  _shift(shift),
  x_l_conv1d(context, in_channels, out_channels, 1),
  x_r_conv1d(context, in_channels, out_channels, 1),
  h_l_conv1d(nullptr),
  h_r_conv1d(nullptr),
  out_conv1d(context, out_channels, out_channels, 1)
{
    if (local_condition_channels > 0) {
        h_l_conv1d = new layers::Conv1D(context, local_condition_channels, out_channels, 1);
        h_r_conv1d = new layers::Conv1D(context, local_condition_channels, out_channels, 1);
    }
}

FFTNetBlock::~FFTNetBlock()
{
    if (_local_condition_channels > 0) {
        delete h_l_conv1d;
        delete h_r_conv1d;
    }
}

cudnn::Tensor4d FFTNetBlock::CreateOutputTensor(const cudnn::Tensor4d &input_tensor)
{
    return out_conv1d.CreateOutputTensor(
        cudnn::Tensor4d(
            input_tensor.batch_size,
            input_tensor.n_channels,
            1, input_tensor.width - this->_shift));
}

void FFTNetBlock::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    // x_l = x[:, :, :-self.shift]
    // x_r = x[:, :, self.shift:]
    cudnn::Tensor4d x_shift_tensor(x_tensor.batch_size, x_tensor.n_channels, 1, x_tensor.width - this->_shift);
    auto x_l_data = x_shift_tensor.CreateArray4f32();
    auto x_r_data = x_shift_tensor.CreateArray4f32();
    _SplitXByShift(x_tensor, x_data, x_l_data, x_r_data);

    // x_l = self.x_l_conv(x_l) 
    // x_r = self.x_r_conv(x_r) 
    auto x_conv_out_tensor = x_l_conv1d.CreateOutputTensor(x_shift_tensor);
    auto x_l_conv_out_data = x_conv_out_tensor.CreateArray4f32();
    auto x_r_conv_out_data = x_conv_out_tensor.CreateArray4f32();
    x_l_conv_out_data.InitializeWithZeros();
    x_r_conv_out_data.InitializeWithZeros();

    x_l_conv1d.Forward(x_shift_tensor, x_l_data, x_conv_out_tensor, x_l_conv_out_data);
    x_r_conv1d.Forward(x_shift_tensor, x_r_data, x_conv_out_tensor, x_r_conv_out_data);

    // x_r = F.relu(x_l + x_r)
    _AddTensor(x_conv_out_tensor, x_l_conv_out_data, x_r_conv_out_data);

    _relu(x_conv_out_tensor, x_r_conv_out_data, x_l_conv_out_data);

    // output = F.relu(self.output_conv(x_r))
    auto out_conv_out_data = out_tensor.CreateArray4f32();
    out_conv_out_data.InitializeWithZeros();

    out_conv1d.Forward(x_conv_out_tensor, x_l_conv_out_data, out_tensor, out_conv_out_data);
    _relu(out_tensor, out_conv_out_data, out_data);
}

void FFTNetBlock::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    // x_l = x[:, :, :-self.shift]
    // x_r = x[:, :, self.shift:]
    cudnn::Tensor4d x_shift_tensor(x_tensor.batch_size, x_tensor.n_channels, 1, x_tensor.width - this->_shift);
    auto x_l_data = x_shift_tensor.CreateArray4f32();
    auto x_r_data = x_shift_tensor.CreateArray4f32();
    _SplitXByShift(x_tensor, x_data, x_l_data, x_r_data);

    // h = h[:, :, -x.size(-1):]
    // h_l = h[:, :, :-self.shift]
    // h_r = h[:, :, self.shift:]
    cudnn::Tensor4d h_shift_tensor(h_tensor.batch_size, h_tensor.n_channels, 1, x_tensor.width - this->_shift);
    auto h_l_data = h_shift_tensor.CreateArray4f32();
    auto h_r_data = h_shift_tensor.CreateArray4f32();
    _SplitHByShift(h_tensor, h_data, x_tensor.width, h_l_data, h_r_data);

    // x_l = self.x_l_conv(x_l) 
    // x_r = self.x_r_conv(x_r) 
    auto x_conv_out_tensor = x_l_conv1d.CreateOutputTensor(x_shift_tensor);
    auto x_l_conv_out_data = x_conv_out_tensor.CreateArray4f32();
    auto x_r_conv_out_data = x_conv_out_tensor.CreateArray4f32();
    x_l_conv_out_data.InitializeWithZeros();
    x_r_conv_out_data.InitializeWithZeros();
    x_l_conv1d.Forward(x_shift_tensor, x_l_data, x_conv_out_tensor, x_l_conv_out_data);
    x_r_conv1d.Forward(x_shift_tensor, x_r_data, x_conv_out_tensor, x_r_conv_out_data);

    // h_l = self.h_l_conv(h_l)
    // h_r = self.h_r_conv(h_r) 
    auto h_conv_out_tensor = h_l_conv1d->CreateOutputTensor(x_shift_tensor);
    auto h_l_conv_out_data = h_conv_out_tensor.CreateArray4f32();
    auto h_r_conv_out_data = h_conv_out_tensor.CreateArray4f32();
    h_l_conv_out_data.InitializeWithZeros();
    h_r_conv_out_data.InitializeWithZeros();
    h_l_conv1d->Forward(h_shift_tensor, h_l_data, h_conv_out_tensor, h_l_conv_out_data);
    h_r_conv1d->Forward(h_shift_tensor, h_r_data, h_conv_out_tensor, h_r_conv_out_data);

    // z_x = x_l + x_r
    _AddTensor(x_conv_out_tensor, x_l_conv_out_data, x_r_conv_out_data);

    // z_h = h_l + h_r
    _AddTensor(h_conv_out_tensor, h_l_conv_out_data, h_r_conv_out_data);

    // z = F.relu(z_x + z_h)
    _AddTensor(h_conv_out_tensor, h_r_conv_out_data, x_r_conv_out_data);
    _relu(h_conv_out_tensor, x_r_conv_out_data, x_l_conv_out_data);

    // output = F.relu(self.output_conv(z))
    auto out_conv_out_data = out_tensor.CreateArray4f32();
    out_conv_out_data.InitializeWithZeros();

    out_conv1d.Forward(x_conv_out_tensor, x_l_conv_out_data, out_tensor, out_conv_out_data);
    _relu(out_tensor, out_conv_out_data, out_data);
}

void FFTNetBlock::_SplitXByShift(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    cudnn::Array4f32 &x_l_data, cudnn::Array4f32 &x_r_data) const
{
    for(int batch = 0; batch < x_tensor.batch_size; ++batch) {
        for (int ch = 0; ch < x_tensor.n_channels; ++ch) {
            for (int row = 0; row < x_tensor.height; ++row) {
                for (int col = 0; col < x_tensor.width - this->_shift; ++col) {
                    x_l_data(batch, ch, row, col) = x_data(batch, ch, row, col);
                    x_r_data(batch, ch, row, col) = x_data(batch, ch, row, col + this->_shift);                    
                }
            }
        }
    }
}

void FFTNetBlock::_SplitHByShift(
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    int x_width,
    cudnn::Array4f32 &h_l_data,
    cudnn::Array4f32 &h_r_data)
{
    for(int batch = 0; batch < h_tensor.batch_size; ++batch) {
        for (int ch = 0; ch < h_tensor.n_channels; ++ch) {
            for (int row = 0; row < h_tensor.height; ++row) {
                int col_start = h_tensor.width - x_width;
                for (int col = 0; col < x_width - this->_shift; ++col) {
                    h_l_data(batch, ch, row, col) = h_data(batch, ch, row, col + col_start);
                    h_r_data(batch, ch, row, col) = h_data(batch, ch, row, col + col_start + this->_shift);                    
                }
            }
        }
    }
}

void FFTNetBlock::_AddTensor(
    const cudnn::Tensor4d &tensor,
    const cudnn::Array4f32 &data,
    cudnn::Array4f32 &output)
{
    const float alpha = 1.0, beta = 1.0;
    assert_cudnn_success( cudnnAddTensor(
        static_cast<cudnnHandle_t>(_context),
        &alpha,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        data.data(),
        &beta,
        static_cast<cudnnTensorDescriptor_t>(tensor),
        output.data()) );
}
