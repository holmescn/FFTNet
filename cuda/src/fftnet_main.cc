#include <chrono>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include "fftnet.h"
#include "hparams.h"
#include "H5Cpp.h"

using namespace H5;

void LoadLocalCondition(int receptive_field, int upsample_factor,
    const DataSet& lc_dataset,
    hsize_t lc_dims[2],
    cudnn::Array4f32 &local_condition)
{
    cudnn::Array4f32 raw(1, 1, lc_dims[0], lc_dims[1]);
    lc_dataset.read(raw.data(), H5::PredType::NATIVE_FLOAT);

    for(int ch = 0; ch < local_condition.dim(2); ch++) {
        for(int w = receptive_field; w < local_condition.dim(3); w++) {
            int ww = (w - receptive_field) / upsample_factor;
            local_condition(0, 0, ch, w) = raw(0, 0, ww, ch);
        }
    }
}

float MuLawDecode(float signal, int quantization_channels)
{
    int mu = quantization_channels - 1;
    float x = 2 * (signal / mu) - 1;
    float sign = x < 0 ? -1.0 : 1.0;
    float y = sign * (1.0 / mu) * (pow(1.0 + mu, fabs(x)) - 1.0);
    return y;
}

void WriteToFile(const char* filename, const std::vector<float> &samples)
{
    H5File ofile( filename, H5F_ACC_TRUNC );

    float fillvalue = 0;   /* Fill value for the dataset */
    DSetCreatPropList plist;
    plist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue);

    hsize_t fdim[1] = { 0 };
    fdim[0] = samples.size();
    DataSpace fspace( 1, fdim );
    fspace.selectAll();

    auto dataset = ofile.createDataSet("samples", PredType::NATIVE_FLOAT, fspace, plist);

    hsize_t mdim[1] = { 0 };
    mdim[0] = samples.size();
    DataSpace mspace( 1, mdim );

    dataset.write(samples.data(), PredType::NATIVE_FLOAT, mspace, fspace);

    dataset.close();
    ofile.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <checkpoint_file> "
                  << " <lc_file> "
                  << " <output_file> "
                  << std::endl;
        return 1;
    }

    HParams hparams;
    int upsample_factor = hparams.frame_shift_ms / 1000.0 * hparams.sample_rate;

    // ONLY melspc SUPPORTTED
    int lc_channel =  hparams.num_mels;
    FFTNet fftnet(hparams.n_stacks,
                  hparams.fft_channels,
                  hparams.quantization_channels,
                  lc_channel);

    fftnet.InitializeWithHDF5(argv[1]);

    H5File file( argv[2], H5F_ACC_RDONLY );
    const H5std_string DATASET_NAME( "local_condition" );
    auto ds_local_condition = file.openDataSet( DATASET_NAME );

    auto dataspace_lc = ds_local_condition.getSpace();
    hsize_t lc_dims[2];
    int ndims_lc = dataspace_lc.getSimpleExtentDims( lc_dims, NULL);

    cudnn::Array4f32 local_condition(1, 1, lc_dims[1],
                                     lc_dims[0] * upsample_factor + fftnet.receptive_field);
    local_condition.InitializeWithZeros();
    LoadLocalCondition(fftnet.receptive_field, upsample_factor, ds_local_condition, lc_dims, local_condition);

    auto start = std::chrono::system_clock::now();
    std::vector<float> samples(fftnet.receptive_field, 0.0);

    cudnn::Tensor4d x_tensor(1, 1, 1, fftnet.receptive_field);
    auto x_data = x_tensor.CreateArray4f32();
    x_data.InitializeWithZeros();

    cudnn::Tensor4d h_tensor(1, lc_dims[1], 1, fftnet.receptive_field);
    auto h_data = h_tensor.CreateArray4f32();
    h_data.InitializeWithZeros();

    auto output_tensor = fftnet.CreateOutputTensor(x_tensor);
    auto output_data = output_tensor.CreateArray4f32();
    output_data.InitializeWithZeros();

    cudnn::Tensor4d softmax_tensor(1, 1, 1, output_tensor.n_channels);
    auto softmax_in_data = softmax_tensor.CreateArray4f32();
    auto softmax_out_data = softmax_tensor.CreateArray4f32();
    softmax_in_data.InitializeWithZeros();
    softmax_out_data.InitializeWithZeros();

    int max_steps = local_condition.dim(3) - fftnet.receptive_field;
    for(int i = 0; i < max_steps; i++) {
        auto iter = samples.rbegin();
        for(int w = 0; w < fftnet.receptive_field; w++, iter++) {
            x_data(0, 0, 0, fftnet.receptive_field - w - 1) = *iter;
        }

        for(int ch = 0; ch < h_tensor.n_channels; ch++) {
            for(int w = 0; w < h_tensor.width; w++) {
                h_data(0, ch, 0, w) = local_condition(0, 0, ch, w + i + 1);
            }
        }

        fftnet.Forward(x_tensor, x_data, h_tensor, h_data, output_tensor, output_data);

        for(int ch = 0; ch < softmax_tensor.width; ch++) {
            softmax_in_data(0, 0, 0, ch) = output_data(0, ch, 0, output_tensor.width - 1);
        }

        fftnet.Softmax(softmax_tensor, softmax_in_data, softmax_out_data);

        float max_val = softmax_out_data(0, 0, 0, 0);
        size_t max_index = 0;
        for(int w = 0; w < softmax_tensor.width; w++) {
            float val = softmax_out_data(0, 0, 0, w);
            if (val > max_val) {
                max_val = val;
                max_index = w;
            }
        }

        float sample = MuLawDecode(max_index * 1.0, hparams.quantization_channels);
        samples.push_back(sample);

        std::cout << "step (" << i << "/" << max_steps << ")\r";
    }
    std::cout << std::endl;

    WriteToFile(argv[3], samples);

    auto now = std::chrono::system_clock::now();
    auto dt_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start);
    size_t mins = dt_in_seconds.count() / 60;
    size_t secs = dt_in_seconds.count() % 60;
    std::cout << "Takes: " << mins << ":" << secs << std::endl;

    return 0;
}