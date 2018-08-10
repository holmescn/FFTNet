#pragma once
#ifndef __HPARAMS_H__
#define __HPARAMS_H__

struct HParams {

    // Audio:
    const int num_mels = 80;
    const int num_freq = 1025;
    const int mcep_dim = 24;
    const float mcep_alpha = 0.41;
    const int minf0 = 40;
    const int maxf0 = 500;
    const int sample_rate = 16000;
    const char *feature_type = "melspc"; // mcc or melspc
    const int frame_length_ms = 25;
    const int frame_shift_ms = 10;
    const float preemphasis = 0.97;
    const int min_level_db = -100;
    const int ref_level_db = 20;
    const bool noise_injecting = true;

    // Training:
    const bool use_cuda = true;
    const bool use_local_condition = true;
    const int batch_size = 8;
    const int sample_size = 16000;
    const float learning_rate = 2e-4;
    const int training_steps = 200000;
    const int checkpoint_interval = 5000;

    // Model
    const int n_stacks = 11;
    const int fft_channels = 256;
    const int quantization_channels = 256;
};

#endif // __HPARAMS_H__