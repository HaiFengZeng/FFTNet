from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.nn import ConvTranspose2d
from hparams import hparams


class UpSampleConv(nn.Module):
    def __init__(self):
        super(UpSampleConv, self).__init__()
        self.upsample_conv = nn.ModuleList()
        for s in hparams.upsample_scales:
            freq_axis_padding = (hparams.freq_axis_kernel_size - 1) // 2
            convt = ConvTranspose2d(1, 1, (hparams.freq_axis_kernel_size, s),
                                    padding=(freq_axis_padding, 0),
                                    dilation=1, stride=[1, s])
            self.upsample_conv.append(weight_norm(convt))
            self.upsample_conv.append(nn.LeakyReLU(inplace=True, negative_slope=0.4))

    def forward(self, c):
        for f in self.upsample_conv:
            c = f(c)
        return c


class FFTNetBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shift,
                 local_condition_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift = shift
        self.local_condition_channels = local_condition_channels
        self.x_l_conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        self.x_r_conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        if local_condition_channels is not None:
            self.h_l_conv = weight_norm(nn.Conv1d(local_condition_channels, out_channels, kernel_size=1))
            self.h_r_conv = weight_norm(nn.Conv1d(local_condition_channels, out_channels, kernel_size=1))
        self.output_conv = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x, h=None):
        x_l = self.x_l_conv(x[:, :, :-self.shift])
        x_r = self.x_r_conv(x[:, :, self.shift:])
        if h is None:
            z = F.relu(x_l + x_r)
        else:
            h = h[:, :, -x.size(-1):]
            h_l = self.h_l_conv(h[:, :, :-self.shift])
            h_r = self.h_r_conv(h[:, :, self.shift:])
            z_x = x_l + x_r
            z_h = h_l + h_r
            z = F.relu(z_x + z_h)
        output = F.relu(self.output_conv(z))

        return output


class FFTNet(nn.Module):
    """Implements the FFTNet for vocoder

    Reference: FFTNet: a Real-Time Speaker-Dependent Neural Vocoder. ICASSP 2018

    Args:
        n_stacks: the number of stacked fft layer
        fft_channels:
        quantization_channels:
        local_condition_channels:
    """

    def __init__(self,
                 n_stacks=11,
                 fft_channels=256,
                 quantization_channels=256,
                 out_channels=2,
                 local_condition_channels=None,
                 upsample_network=False,
                 out_type='Gaussian'):
        super().__init__()
        self.n_stacks = n_stacks
        self.fft_channels = fft_channels
        self.quantization_channels = quantization_channels
        self.local_condition_channels = local_condition_channels
        self.window_shifts = [2 ** i for i in range(self.n_stacks)]
        self.upsample_network = upsample_network
        self.receptive_field = sum(self.window_shifts) + 1
        self.linear = nn.Linear(fft_channels, quantization_channels)
        self.layers = nn.ModuleList()
        self.output_layer = weight_norm(
            nn.Conv1d(fft_channels, out_channels, kernel_size=1)) if out_type == 'Gaussian' else self.linear
        # using nv-wavenet's way handle
        if self.upsample_network:
            self.upsample_conv = UpSampleConv()
        else:
            self.upsample_conv = None

        for shift in reversed(self.window_shifts):
            if shift == self.window_shifts[-1]:
                in_channels = 1
            else:
                in_channels = fft_channels
            fftlayer = FFTNetBlock(in_channels, fft_channels, shift, local_condition_channels)
            self.layers.append(fftlayer)

    def forward(self, x, h):

        if self.upsample_conv:
            h = h.transpose(1, 2)
            h = h.unsqueeze(1)
            h = self.upsample_conv(h)
            h = h.squeeze(1)
            # [B,C,T]
            h = F.pad(h, (self.receptive_field, 0))
        h = h[:, :, 1:]

        output = x.transpose(1, 2)
        for fft_layer in self.layers:
            output = fft_layer(output, h)

        output = self.output_layer(F.relu(output))
        return output

    def inference(self, x, h):
        output = x.transpose(1, 2)
        for fft_layer in self.layers:
            output = fft_layer(output, h)

        output = self.output_layer(F.relu(output))
        return output
