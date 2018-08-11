import argparse
import h5py
import numpy as np
import scipy.io.wavfile


def write_wav(wav, sample_rate, filename):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    scipy.io.wavfile.write(filename, sample_rate, wav.astype(np.int16))
    print('Write wav file to {}'.format(filename))


def main(opts):
    with h5py.File(opts.input_file) as in_file:
        samples = np.asarray(in_file['samples'])
        print(samples.shape)
        write_wav(samples, opts.sample_rate, opts.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True,
        help='Input file.')
    parser.add_argument('-o', "--output_file", type=str, required=True,
        help='Output file.')
    parser.add_argument("--sample_rate", type=int, default=16000,
        help='Sample rate.')
    args = parser.parse_args()
    main(args)
