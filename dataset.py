from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import io
import numpy as np
import os
import torch

from torch.utils.data import Dataset
from utils.utils import mu_law_encode

class CustomDataset(Dataset):

    def __init__(self,
                 meta_file,
                 receptive_field,
                 sample_size=5000,
                 upsample_factor=200,
                 quantization_channels=256,
                 use_local_condition=True):
        with open(meta_file, encoding='utf-8') as f:
            self.metadata = [line.strip().split('|') for line in f]
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.upsample_factor = upsample_factor
        self.quantization_channels = quantization_channels
        self.use_local_condition = use_local_condition

        self.audio_buffer, self.local_condition_buffer = self._load_data(
                           self.metadata, use_local_condition, post_fn=lambda x: np.load(x))

    def __len__(self):
        return len(self.audio_buffer)

    def __getitem__(self, index):
        audios = self.audio_buffer[index]
        rand_pos = np.random.randint(0, len(audios) - self.sample_size - self.receptive_field)

        if self.use_local_condition:
            local_condition = self.local_condition_buffer[index]

            lc_pad = int(self.receptive_field / float(self.upsample_factor + 0.5))
            audio_pad = lc_pad * self.upsample_factor
            audios = np.pad(audios, [[audio_pad, 0], [0, 0]], 'constant')
            local_condition = np.pad(local_condition, [[lc_pad, 0], [0, 0]], 'constant')

            local_condition = np.repeat(local_condition, self.upsample_factor, axis=0)
            local_condition = local_condition[rand_pos : rand_pos + self.sample_size + self.receptive_field]
        else:
            audios = np.pad(audios, [[self.receptive_field, 0], [0, 0]], 'constant')
            local_condition = None

        audios = audios[rand_pos : rand_pos + self.sample_size + self.receptive_field]
        audios = mu_law_encode(audios, self.quantization_channels)

        return torch.LongTensor(audios), torch.FloatTensor(local_condition)
    
    def _load_data(self, metadata, use_local_condition, post_fn=lambda x: x):
        audio_buffer = []
        local_condition_buffer = []
        for x in metadata:
            audio_buffer.append(post_fn(x[0]))
            if use_local_condition:
                local_condition_buffer.append(post_fn(x[1]))
        return audio_buffer, local_condition_buffer
