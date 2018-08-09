"""Convert the checkpoint and lc file to HDF5.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import torch
import h5py
import numpy as np


def convert_checkpoint(checkpoint_file, data_dir):
    ckpt = torch.load(checkpoint_file, map_location='cpu')
    full_path = os.path.join(data_dir, 'model_ckpt.h5')
    with h5py.File(full_path, 'w') as hdf:
        for k, v in ckpt['model'].items():
            data = v.numpy()
            dset = hdf.create_dataset(k.replace('.', '/'), data.shape, data.dtype)
            dset[...] = data


def convert_lc(lc_file, data_dir):
    local_condition = np.load(lc_file)
    full_path = os.path.join(data_dir, 'lc_file.h5')
    with h5py.File(full_path, 'w') as hdf:
        dset = hdf.create_dataset('local_condition', local_condition.shape, local_condition.dtype)
        dset[...] = local_condition


def main(opts):
    convert_checkpoint(opts.checkpoint, opts.data_dir)
    convert_lc(opts.lc_file, opts.data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
        help='Checkpoint path to restore model')
    parser.add_argument('--lc_file', type=str, required=True,
        help='Local condition file path.')
    parser.add_argument('--data_dir', type=str, default='training_data',
        help='data dir')
    args = parser.parse_args()
    main(args)
