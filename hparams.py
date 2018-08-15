import tensorflow as tf


# Default hyperparameters:
hparams = tf.contrib.training.HParams(

  # Audio:
  num_mels=80,
  num_freq=1025,
  mcep_dim=24,
  mcep_alpha=0.41,
  hop=160,
  minf0=40,
  maxf0=500,
  sample_rate=16000,
  feature_type='melspc', # mcc or melspc
  frame_length_ms=25,
  frame_shift_ms=10,
  preemphasis=0.97,
  min_level_db=-100,
  ref_level_db=20,
  noise_injecting=True,
  upsample_factor=200,

  # Training:
  use_cuda=True,
  use_local_condition=True,
  batch_size=8,
  sample_size=16000,
  learning_rate=5e-4,
  training_steps=200000,
  checkpoint_interval=2000,
  epoches=2000,

  # Model
  n_stacks=11,
  fft_channels=256,
  quantization_channels=256,
  out_channels=2,
  out_type='Gaussian',
  upsample_network=True,
  freq_axis_kernel_size=3,
  upsample_scales=[10,16],# eq hop
  # Generate
  generate_path='generate'
)


def hparams_debug_string():
  values = hparams.values()
  hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
  return 'Hyperparameters:\n' + '\n'.join(hp)
