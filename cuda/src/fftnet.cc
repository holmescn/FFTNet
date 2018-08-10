#include <cassert>
#include <cstring>
#include <iostream>
#include "fftnet.h"

FFTNet::FFTNet(
    int n_stacks,
    int fft_channels, 
    int quantization_channels, 
    int local_condition_channels)
: _context(),
  _n_stacks(n_stacks),
  _fft_channels(fft_channels),
  _quantization_channels(quantization_channels),
  _local_condition_channels(local_condition_channels),
  linear(_context, fft_channels, quantization_channels)
{
    // self.window_shifts = [2 ** i for i in range(self.n_stacks)]
    // self.receptive_field = sum(self.window_shifts) + 1
    receptive_field = 1;
    for(int i = 0; i < n_stacks; i++) {
        int shift = 1 << i;
        _window_shifts.push_back(shift);
        receptive_field += shift;
    }

    auto last_shift = _window_shifts.back();
    for(auto iter = _window_shifts.rbegin(); iter != _window_shifts.rend(); iter ++) {
        int shift = *iter;
        int in_channels = (shift == last_shift ? 1 : fft_channels);
        auto fftnet_block = std::make_shared<FFTNetBlock>(_context, in_channels, fft_channels, shift, local_condition_channels);
        layers.push_back(fftnet_block);
    }

    // self.linear = nn.Linear(fft_channels, quantization_channels)
}

FFTNet::~FFTNet()
{
    
}

void FFTNet::Forward(
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    // Suppose the input is channels_first.
    Forward_Impl(layers.begin(), x_tensor, x_data, h_tensor, h_data, out_tensor, out_data);
    cudaDeviceSynchronize();
}

void FFTNet::Forward_Impl(
    std::vector<std::shared_ptr<FFTNetBlock>>::iterator iter,
    const cudnn::Tensor4d &x_tensor, const cudnn::Array4f32 &x_data,
    const cudnn::Tensor4d &h_tensor, const cudnn::Array4f32 &h_data,
    const cudnn::Tensor4d &out_tensor, cudnn::Array4f32 &out_data)
{
    if (iter != layers.end()) {
        auto p_layer = *iter;
        auto layer_out_tensor = p_layer->CreateOutputTensor(x_tensor);
        auto layer_out_data = layer_out_tensor.CreateArray4f32();

        if (_local_condition_channels > 0) {
            p_layer->Forward(x_tensor, x_data, h_tensor, h_data, layer_out_tensor, layer_out_data);
        } else {
            p_layer->Forward(x_tensor, x_data, layer_out_tensor, layer_out_data);
        }
        cudaDeviceSynchronize();

        Forward_Impl(std::next(iter), layer_out_tensor, layer_out_data,
                          h_tensor, h_data, out_tensor, out_data);
    } else {
        // output = self.linear(output.transpose(1, 2))
        // return output.transpose(1, 2)
        linear.Forward(x_tensor, x_data, out_tensor, out_data);
    }
}

cudnn::Tensor4d
FFTNet::CreateOutputTensor(const cudnn::Tensor4d &input_tensor)
{
    return CreateOutputTensor_Impl(layers.begin(), input_tensor);
}

cudnn::Tensor4d
FFTNet::CreateOutputTensor_Impl(
    std::vector<std::shared_ptr<FFTNetBlock>>::iterator iter,
    const cudnn::Tensor4d &input_tensor)
{
    if (iter != layers.end()) {
        auto p_layer = *iter;
        auto layer_out_tensor = p_layer->CreateOutputTensor(input_tensor);
        return CreateOutputTensor_Impl(std::next(iter), layer_out_tensor);
    } else {
        return linear.CreateOutputTensor(input_tensor);
    }
}

void FFTNet::InitializeWithHDF5(const char *full_path)
{
    H5::H5File hdf_file( full_path, H5F_ACC_RDONLY );

    int layer_index = 0;
    for(auto iter = layers.begin(); iter != layers.end(); iter++, layer_index++) {
        auto p_layer = *iter;
        _InitializeConv1DLayer(hdf_file, layer_index, p_layer->x_l_conv1d, "x_l_conv");
        _InitializeConv1DLayer(hdf_file, layer_index, p_layer->x_r_conv1d, "x_r_conv");
        if (p_layer->h_l_conv1d != nullptr) {
            _InitializeConv1DLayer(hdf_file, layer_index, *(p_layer->h_l_conv1d), "h_l_conv");
            _InitializeConv1DLayer(hdf_file, layer_index, *(p_layer->h_r_conv1d), "h_r_conv");
        }
        _InitializeConv1DLayer(hdf_file, layer_index, p_layer->out_conv1d, "output_conv");
    }

    _InitializeLinearLayer(hdf_file);
}

void FFTNet::_InitializeConv1DLayer(
    const H5::H5File &hdf_file,
    int layer_index,
    const layers::Conv1D &layer,
    const char *layer_name)
{
    char str_buf[1024] = { 0 };

    snprintf(str_buf, 1024, "layers/%d/%s/weight", layer_index, layer_name);
    H5std_string ds_weight_name = str_buf;

    snprintf(str_buf, 1024, "layers/%d/%s/bias", layer_index, layer_name);
    H5std_string ds_bias_name = str_buf;

    H5::DataSet ds_weight = hdf_file.openDataSet( ds_weight_name );
    H5::DataSet ds_bias = hdf_file.openDataSet( ds_bias_name );

    H5::DataSpace dataspace_weight = ds_weight.getSpace();
    hsize_t weight_dims[4];
    int ndims_weight = dataspace_weight.getSimpleExtentDims( weight_dims, NULL);
    assert(weight_dims[0] == layer.out_channels);
    assert(weight_dims[1] == layer.in_channels);
    assert(weight_dims[2] == layer.kernel_size);

    H5::DataSpace dataspace_bias = ds_bias.getSpace();
    hsize_t bias_dims[1];
    int ndims_bias = dataspace_bias.getSimpleExtentDims( bias_dims, NULL);
    assert(bias_dims[0] == layer.out_channels);

    ds_weight.read(layer.weight_data.data(), H5::PredType::NATIVE_FLOAT);
    ds_bias.read(layer.bias_data.data(), H5::PredType::NATIVE_FLOAT);
}

void FFTNet::_InitializeLinearLayer(const H5::H5File &hdf_file)
{
    H5std_string ds_weight_name = "linear/weight";
    H5std_string ds_bias_name = "linear/bias";

    H5::DataSet ds_weight = hdf_file.openDataSet( ds_weight_name );
    H5::DataSet ds_bias = hdf_file.openDataSet( ds_bias_name );

    H5::DataSpace dataspace_weight = ds_weight.getSpace();
    hsize_t weight_dims[2];
    int ndims_weight = dataspace_weight.getSimpleExtentDims( weight_dims, NULL);
    assert(weight_dims[0] == linear.out_features);
    assert(weight_dims[1] == linear.in_features);

    H5::DataSpace dataspace_bias = ds_bias.getSpace();
    hsize_t bias_dims[1];
    int ndims_bias = dataspace_bias.getSimpleExtentDims( bias_dims, NULL);
    assert(bias_dims[0] == linear.out_features);

    float *weight_data = new float[linear.size()];
    ds_weight.read(weight_data, H5::PredType::NATIVE_FLOAT);
    ds_bias.read(linear.bias_data.data(), H5::PredType::NATIVE_FLOAT);

    linear.weight(weight_data);
    delete[] weight_data;
}
