from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

import librosa
import numpy as np
import os
import time
import scipy.io.wavfile
from sklearn.preprocessing import StandardScaler

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from torch.nn import functional as F
from tqdm import tqdm
from utils import infolog
from fftnet_gaussian import FFTNet, UpSampleConv
from hparams import hparams, hparams_debug_string
from utils.utils import mu_law_encode, mu_law_decode, write_wav
from train1 import sample_gaussian,save_waveplot
from utils import audio


def upsample_condition(model,z):

    z = z.view(1, *z.size())
    z = z.unsqueeze(1)
    z = model.upsample_conv(z)
    z = z.squeeze(1)
    return z


def prepare_data(lc_file, model, hparams, read_fn=lambda x: x, feat_transform=None):
    samples = [0.0] * model.receptive_field
    local_condition = read_fn(lc_file)
    uv = local_condition[:, 0]
    if feat_transform is not None:
        local_condition = feat_transform(local_condition)
    uv = np.repeat(uv, hparams.upsample_factor, axis=0)
    uv = np.pad(uv, [model.receptive_field, 0], 'constant')
    if not hparams.upsample_network:
        local_condition = np.repeat(local_condition, hparams.upsample_factor, axis=0)
        local_condition = np.pad(local_condition, [[model.receptive_field, 0], [0, 0]], 'constant')
        local_condition = local_condition[np.newaxis, :, :]
        local_condition = torch.FloatTensor(local_condition).transpose(1, 2)
    else:
        local_condition = torch.from_numpy(local_condition.transpose(1, 0)).float()
        local_condition = upsample_condition(model,local_condition)

    return samples, local_condition, uv


def extract_audio_mels(audio_path):
    wav = audio.load_wav(audio_path)
    mels = audio.melspectrogram(wav)
    return mels


def create_model(hparams):
    if hparams.feature_type == 'mcc':
        lc_channel = hparams.mcep_dim + 3
    else:
        lc_channel = hparams.num_mels

    return FFTNet(n_stacks=hparams.n_stacks,
                  fft_channels=hparams.fft_channels,
                  quantization_channels=hparams.quantization_channels,
                  upsample_network=hparams.upsample_network,
                  out_channels=2,
                  local_condition_channels=lc_channel)





def generate_fn(args):
    device = torch.device("cuda" if hparams.use_cuda else "cpu")
    upsample_factor = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    model = create_model(hparams)

    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    if torch.cuda.device_count() > 1:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.to(device)
    model.eval()

    if hparams.feature_type == "mcc":
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(args.data_dir, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(args.data_dir, 'scale.npy'))
        feat_transform = transforms.Compose([lambda x: scaler.transform(x)])
    else:
        feat_transform = None

    with torch.no_grad():
        samples, local_condition, uv = prepare_data(lc_file=args.lc_file,
                                                    model=model,
                                                    hparams=hparams,
                                                    read_fn=lambda x: np.load(x),
                                                    feat_transform=feat_transform)
        print(local_condition.shape)
        start = time.time()
        for i in tqdm(range(local_condition.size(-1) - model.receptive_field)):
            sample = torch.FloatTensor(np.array(samples[-model.receptive_field:]).reshape(1, -1, 1))
            h = local_condition[:, :, i + 1: i + 1 + model.receptive_field]
            sample, h = sample.to(device), h.to(device)
            output = model.inference(sample, h)

            if hparams.feature_type == "mcc":
                if uv[i + model.receptive_field] == 0:
                    output = output[0, :, -1]
                    outprob = F.softmax(output, dim=0).cpu().numpy()
                    sample = np.random.choice(np.arange(hparams.quantization_channels),p=outprob)
                else:
                    output = output[0, :, -1] * 2
                    outprob = F.softmax(output, dim=0).cpu().numpy()
                    sample = outprob.argmax(0)
            else:
                # I tested sampling, but it will produce more noise,
                # so I use argmax in this time.
                if hparams.out_type == 'Gaussian':
                    output =output.transpose(1,2)
                    sample = sample_gaussian(output)
                else:
                    output = output[0, :, -1]
                    outprob = F.softmax(output, dim=0).cpu().numpy()
                    sample = outprob.argmax(0)
                    sample = mu_law_decode(sample, hparams.quantization_channels)
            samples.append(sample)
        samples = samples[model.receptive_field:]
        write_wav(np.asarray(samples), hparams.sample_rate,
                  os.path.join(os.path.dirname(args.checkpoint),
                               "generated-{}.wav".format(os.path.basename(args.checkpoint))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/home/jinqiangzeng/work/pycharm/FFTNet/checkpoints_gaussian/model.ckpt-102000.pt',
                        help='Checkpoint path to restore model')
    parser.add_argument('--lc_file', type=str,
                        default='/home/jinqiangzeng/work/pycharm/FFTNet/training_data/mels/arctic_a0001.npy',
                        help='Local condition file path.')
    parser.add_argument('--audio_file', type=str,
                        default='/home/jinqiangzeng/work/pycharm/FFTNet/training_data/wavs/arctic_a0001.npy',
                        help='original file path.')
    parser.add_argument('--data_dir', type=str, default='training_data',
                        help='data dir')
    parser.add_argument('--hparams', default='upsample_network=false',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    hparams.parse(args.hparams)
    generate_fn(args)
