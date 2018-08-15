from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import re
import sys
import scipy.io.wavfile
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import DataLoader
from fftnet import FFTNet
from dataset import CustomDataset
from utils.utils import apply_moving_average, ExponentialMovingAverage, mu_law_decode, write_wav
from utils import infolog
from hparams import hparams, hparams_debug_string
from tensorboardX import SummaryWriter
import math, random
import librosa
import librosa.display
from tqdm import tqdm
import torch.nn.functional as F

log = infolog.log


class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, y_hat, y):
        B, T, _ = y.size()
        log_scale_min = -7
        mean, log_scale = y_hat[:, :, :1], y_hat[:, :, 1:]
        scales = torch.exp(torch.clamp(log_scale, min=log_scale_min))

        loss = (y - mean) ** 2 / (2 * scales ** 2) + torch.log(scales) + math.log(math.sqrt(2) * math.pi)
        return torch.sum(loss) / (B * T)


def sample_gaussian(y_hat):
    mean, log_scale = y_hat[:, :, :1], y_hat[:, :, 1:]
    scales = torch.exp(torch.clamp(log_scale, min=-7))
    from torch.distributions import Normal
    normal = Normal(loc=mean, scale=scales)
    x = normal.sample()
    return x



def save_waveplot(path, y_predict, y_target, writer=None, step=0, name='train'):
    sr = hparams.sample_rate

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('target')
    librosa.display.waveplot(y_target, sr=sr)
    plt.subplot(2, 1, 2)
    plt.title('predict')
    librosa.display.waveplot(y_predict, sr=sr)
    plt.tight_layout()
    if path:
        plt.savefig(path, format="png")
    if writer:
        import io
        from PIL import Image
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        plt.close()
        buff.seek(0)
        im = np.array(Image.open(buff))
        writer.add_image(name, im, global_step=step)
    plt.close()


def save_checkpoint(device, hparams, model, optimizer, step, checkpoint_dir, test_step=0, ema=None):
    model = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_test_step": test_step,
        "steps": step}
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}.pt".format(step))
    torch.save(checkpoint_state, checkpoint_path)
    log("Saved checkpoint: {}".format(checkpoint_path))

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, hparams, model, ema)
        averaged_checkpoint_state = {
            "model": averaged_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_test_step": test_step,
            "steps": step}
        checkpoint_path = os.path.join(
            checkpoint_dir, "model.ckpt-{}.ema.pt".format(step))
        torch.save(averaged_checkpoint_state, checkpoint_path)
        log("Saved averaged checkpoint: {}".format(checkpoint_path))


def eval_model(epoch, writer, x, condition, model, device):
    index = random.randint(0, x.size(0) - 1)
    x_input = x[index:index + 1, :, :].transpose(1, 2)
    condition = condition[index:index + 1, :, :].transpose(1, 2)
    samples = list(x_input[:, :, :model.receptive_field].view(-1).data.cpu().numpy())
    for i in tqdm(range(condition.size(-1) - model.receptive_field)):
        sample = torch.FloatTensor(np.array(samples[-model.receptive_field:]).reshape(1, -1, 1))
        h = condition[:, :, i + 1: i + 1 + model.receptive_field]
        sample, h = sample.to(device), h.to(device)
        output = model(sample, h)

        if hparams.out_type == 'Gaussian':
            output = output.transpose(1, 2)
            sample = sample_gaussian(output)
        else:
            output = output[0, :, -1]
            outprob = F.softmax(output, dim=0).cpu().numpy()
            sample = outprob.argmax(0)
            sample = mu_law_decode(sample, hparams.quantization_channels)
        samples.append(sample.item())
    predict = np.asarray(samples[model.receptive_field:])
    target = x_input[:, :, model.receptive_field:].view(-1).data.cpu().numpy()
    path = 'log/eval/eval_{}_epoch.png'.format(epoch)
    save_waveplot(path, predict, target, writer, epoch, name='eval')


def clone_as_averaged_model(device, hparams, model, ema):
    assert ema is not None
    averaged_model = create_model(hparams).to(device)
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def create_model(hparams):
    if hparams.feature_type == 'mcc':
        lc_channel = hparams.mcep_dim + 3
    else:
        lc_channel = hparams.num_mels

    return FFTNet(n_stacks=hparams.n_stacks,
                  fft_channels=hparams.fft_channels,
                  quantization_channels=hparams.quantization_channels,
                  local_condition_channels=lc_channel,
                  out_channels=2,
                  upsample_network=hparams.upsample_network,
                  out_type='Gaussian')


def train_fn(args):
    device = torch.device("cuda" if hparams.use_cuda else "cpu")


    model = create_model(hparams)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

    if args.resume is not None:
        log("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint['steps']
        global_test_step = checkpoint.get('global_test_step', 0)
        epoch = checkpoint.get('epoch', 0)
    else:
        global_step = 0
        global_test_step = 0
        epoch = 0

    log("receptive field: {0} ({1:.2f}ms)".format(
        model.receptive_field, model.receptive_field / hparams.sample_rate * 1000))

    if hparams.feature_type == "mcc":
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(args.data_dir, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(args.data_dir, 'scale.npy'))
        feat_transform = transforms.Compose([lambda x: scaler.transform(x)])
    else:
        feat_transform = None

    dataset = CustomDataset(meta_file=os.path.join(args.data_dir, 'train.txt'),
                            receptive_field=model.receptive_field,
                            sample_size=hparams.sample_size,
                            upsample_factor=hparams.upsample_factor,
                            upsample_network=hparams.upsample_network,
                            quantization_channels=hparams.quantization_channels,
                            use_local_condition=hparams.use_local_condition,
                            noise_injecting=hparams.noise_injecting,
                            feat_transform=feat_transform)
    test_dataset = CustomDataset(meta_file=os.path.join(args.data_dir, 'test.txt'),
                                 receptive_field=model.receptive_field,
                                 sample_size=hparams.sample_size,
                                 upsample_factor=hparams.upsample_factor,
                                 upsample_network=hparams.upsample_network,
                                 quantization_channels=hparams.quantization_channels,
                                 use_local_condition=hparams.use_local_condition,
                                 noise_injecting=hparams.noise_injecting,
                                 feat_transform=feat_transform)
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 pin_memory=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = GaussianLoss()

    ema = ExponentialMovingAverage(args.ema_decay)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    writer = SummaryWriter(args.log_dir)

    while epoch < hparams.epoches:
        model.train()
        for i, data in enumerate(dataloader, 0):
            audio, target, local_condition = data
            local_condition = local_condition.transpose(1, 2)
            audio, target, h = audio.to(device), target.to(device), local_condition.to(device)

            optimizer.zero_grad()
            output = model(audio[:, :-1, :], h).transpose(1, 2)
            target_ = audio[:, 2048:, :]
            loss = torch.sum(criterion(output, target_))
            log('train step [%3d]: loss: %.3f' % (global_step, loss.item()))
            writer.add_scalar('train loss', loss.item(), global_step)

            loss.backward()
            optimizer.step()

            # update moving average
            if ema is not None:
                apply_moving_average(model, ema)

            global_step += 1

            if global_step % hparams.checkpoint_interval == 0:
                save_checkpoint(device, hparams, model, optimizer, global_step, args.checkpoint_dir, global_test_step,
                                ema)
            if global_step % 200 == 0:
                out = output[:1, :, :]
                target = audio[:1, :, :].view(-1).data.cpu().numpy()[model.receptive_field:]
                predict = sample_gaussian(out).view(-1).data.cpu().numpy()
                save_waveplot('log/predict/{}.png'.format(global_step), predict, target, writer, global_step)
        eval_audio, eval_condition, eval_target = None, None, None
        for i, data in enumerate(test_dataloader, 0):
            audio, target, local_condition = data
            eval_audio, eval_target, eval_condition = data
            target = target.squeeze(-1)
            local_condition = local_condition.transpose(1, 2)
            audio, target, h = audio.to(device), target.to(device), local_condition.to(device)
            output = model(audio[:, :-1, :], h[:, :, 1:]).transpose(1, 2)
            target_ = audio[:, model.receptive_field:, :]
            loss = torch.sum(criterion(output, target))
            loss.backward()
            log('test step [%3d]: loss: %.3f' % (global_test_step, loss.item()))
            writer.add_scalar('test loss', loss.item(), global_test_step)
            global_test_step += 1
        epoch += 1
        # eval_model every epoch
        if epoch % 10==0:
            model.eval()
            eval_model(epoch, writer, eval_audio, eval_condition, model, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--data_dir', default='training_data',
                        help='Metadata file which contains the keys of audio and melspec')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='Moving average decay rate.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt_gaussian_upsample/',
                        help='Directory to save checkpoints.')
    parser.add_argument('--log_dir', type=str, default='log/gaussian_upsample',
                        help='Directory to save checkpoints.')
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    infolog.init(os.path.join(args.checkpoint_dir, 'train.log'), 'FFTNET')
    hparams.parse(args.hparams)
    train_fn(args)
