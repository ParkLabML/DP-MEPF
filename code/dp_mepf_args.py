import argparse
import os.path

import numpy as np
import torch as pt
from dp_analysis import find_single_release_sigma, find_two_release_sigma, \
  find_train_val_sigma_m1, find_train_val_sigma_m1m2

from util_logging import get_base_log_dir
from typing import NamedTuple


def get_args():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'lsun', 'celeba',
                                                               'dmnist', 'fmnist'])
  parser.add_argument('--n_iter', type=int, default=100_000, help='number of generator updates')
  parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
  parser.add_argument('--net_enc', nargs='*', default=['models/vgg19.pt'],
                      help="path to net_enc (to continue training)")
  parser.add_argument('--net_enc_type', nargs='*', default=['vgg19'],
                      help="feature extractors to use: 'encoder | vgg19 [default]| resnet18'")
  parser.add_argument('--net_gen_type', default='resnet',
                      help="type of generator/decoder to use: 'dcgan | resnet[default]'")
  parser.add_argument('--exp_name', default=None, type=str, help='more concise way to set log_dir. '
                                                                 'overwrites log_dir if set')
  parser.add_argument('--log_dir', default='../logs/temp',
                      help='folder to output images and model checkpoints')
  parser.add_argument('--labeled', action='store_true', help='enables labeled data generation')
  parser.add_argument('--pretrain_dataset', default=None,
                      choices=[None, 'imagenet', 'svhn', 'cifar10'], help='set automatically')

  parser.add_argument('--gen_output', type=str, default='linear', choices=['tanh', 'linear'],
                      help='tanh: output in range [-1,1]. linear: output unbounded')
  parser.add_argument('--data_scale', type=str, default='normed', choices=['bounded', 'normed'],
                      help='bounded: in range [-1,1]. normed: scaled to imagenet mean and var')

  parser.add_argument('--pytorch_encoders', action='store_true',
                      help="Use torchvision model encoders rather than the ones from the paper")
  parser.add_argument('--select_on_train_data', action='store_true',
                      help="if true, use train data for feature selection (debug mode)")
  parser.add_argument('--keep_best_syn_data', action='store_true',
                      help="if true, don't delete highest scoring synthetic dataset")

  # MOMENT MATCHING OPTIONS
  parser.add_argument('--matched_moments', type=str, default='m1_and_m2',
                      choices=['mean_and_var', 'mean', 'm1_and_m2'],
                      help='mean_and_var is used in the paper - no DP support yet. '
                           'm1_and_m2 matches first and second moment')
  parser.add_argument('--n_split_layers', type=int, default=None,
                      help='if set, use only n last layers for class-conditional embedding')

  # FEATURE SELECTION
  parser.add_argument('--channel_filter_rate', type=float, default=1.,
                      help='fraction of channels in each layer to keep')
  parser.add_argument('--channel_filter_mode', type=str, default="maxnorm",
                      help='how to select channels')
  parser.add_argument('--n_filter_select_batches', type=int, default=10,
                      help='#batches to use for selecting the pruned features')
  parser.add_argument('--hsic_block_size', type=int, default=100, help='')
  parser.add_argument('--hsic_reps', type=int, default=3, help='')
  parser.add_argument('--hsic_max_samples', type=int, default=100_000, help='')
  parser.add_argument('--n_pca_iter', type=int, default=10,
                      help='number of iterations for randomized PCA to converge')

  # PRIVACY PARAMS
  parser.add_argument('--dp_tgt_eps', type=float, default=None,
                      help='DP Epsilon. if set, this overwrites dp_noise with a new value')
  parser.add_argument('--dp_tgt_delta', type=float, default=1e-6,
                      help='DP parameter Delta: should be <1/N, but 1e-6 is a decent guess')
  parser.add_argument('--dp_noise', type=float, default=None, help='noise for gaussian mechanism')
  parser.add_argument('--dp_mean_bound', type=float, default=1., help='bound mean sensitivity')
  parser.add_argument('--dp_var_bound', type=float, default=1., help='bound var/m2 sensitivity')
  parser.add_argument('--dp_sens_bound_type', type=str, default='norm',
                      choices=['norm', 'clip', 'norm_layer', 'clip_layer'],
                      help='if *layer mode, each layer is bounded wit dp_sens_bound and overall'
                           'sensitivity increases by factor sqrt(n_layers) accordingly')
  parser.add_argument('--dp_scale_var_sigma', type=float, default=1.,
                      help='relative scale of second moment noise to first moment noise parameter')

  # LOGGING AND IO FREQUENCY
  parser.add_argument('--stdout_file', type=str, default='out.io', help='stdout is saved here')
  parser.add_argument('--stderr_file', type=str, default='err.io', help='stderr is saved here')
  parser.add_argument('--log_importance_level', type=str, default='info',
                      choices=['debug', 'info', 'warning'],
                      help="minimum importance leven at which to display messages")
  parser.add_argument('--tensorboard_log_iter', type=int, default=100, help="log every n steps")
  parser.add_argument('--restart_iter', type=int, default=100_000,
                      help="Number of batches after which the model is saved into a NEW file.")
  parser.add_argument('--new_ckpt_iter', type=int, default=500_000,
                      help="Number of batches after which the model is saved into a NEW file.")
  parser.add_argument('--ckpt_iter', type=int, default=10_000,
                      help="Number of epochs after which the model is saved (to the same file).")
  parser.add_argument('--valid_iter', type=int, default=10_000,
                      help='Number of batches after which the validation is applied')
  parser.add_argument('--syn_eval_iter', '--fid_log_steps', type=int, default=500_000,
                      help='create fid score every n iterations')
  parser.add_argument('--load_generator', type=str, default='if_exists',
                      choices=['False', 'True', 'if_exists'],
                      help='whether to load a generator (i.e. continue run)')

  # FID SCORE AND SYNTH DATA
  parser.add_argument('--synth_dataset_size', type=int, default=None,
                      help="if none take fid_data for unlabeled or downstream_data for labeled")
  parser.add_argument('--fid_dataset_size', type=int, default=5_000,
                      help="create a synthetic dataset of at least this size after training")
  parser.add_argument('--downstream_dataset_size', type=int, default=50_000,
                      help="create a dataset of at least this size after training if labeled")

  # VALIDATION AND EARLY STOPPING
  parser.add_argument('--validation_mode', type=str, default='off',
                      choices=['off', 'mavg', 'static'],
                      help="enables static of moving average validation loss")
  parser.add_argument('--val_enc', nargs='*', default=None,
                      help='if not none, choose separate enoder for val loss i.e. fid_features')
  parser.add_argument('--val_data', type=str, default=None,
                      choices=['test', 'train', 'split'],
                      help='data to use for validation. test set, train set or split train set')
  parser.add_argument('--dp_val_noise', type=float, default=None, help='noise for gm')
  parser.add_argument('--dp_val_noise_scaling', type=float, default=2.,
                      help='noise scale relative to noise used in training (likely higher)')

  # TURN OFF FUNCTIONALY THAT IS ON BY DEFAULT
  parser.add_argument('--no_io_files', action='store_true', help='disables logging to files')
  parser.add_argument('--no_cuda', action='store_true', help='disables cuda')
  parser.add_argument('--no_tensorboard', action='store_true', help="disables tensorboard logs")

  parser.add_argument('--do_embedding_norms_analysis', action='store_true',
                      help="if true, log L2 norms of embeddings to tensorflow then exit")

  # ADDITIONAL PARAMS WHERE DEFAULTS ARE PROBABLY FINE FOR A START
  parser.add_argument('--dp_sens_bound', type=float, default=None, help='sets, mean and var bound')
  parser.add_argument('--log_messages', type=list, default=[],
                      help='stores error messages before logger, then outputs them')
  parser.add_argument('--dataroot', help='path to dataset', default="../data/")
  parser.add_argument('--n_workers', type=int, help='number of data loading workers', default=2)
  parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
  parser.add_argument('--image_size', type=int, default=32,
                      help='the height / width of the input image to network.reset to 28 for mnist')
  parser.add_argument('--center_crop_size', type=int, default=0,
                      help='Size to use when performing center cropping.')
  parser.add_argument('--z_dim', type=int, default=100, help='size of the latent z vector')
  # parser.add_argument('--ngf', type=int, default=64)  # hyperparams governing number of gen params
  # parser.add_argument('--ndf', type=int, default=64)
  parser.add_argument('--first_batch_id', type=int, default=None,
                      help='sequential number to be used in the first batch')
  parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, default=0.0002')
  parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
  parser.add_argument('--manual_seed', '--seed', type=int, help='manual seed', default=None)
  parser.add_argument('--n_matching_layers', type=int, default=16,
                      help='Number of layers of the feature extractor used for feature matching')
  parser.add_argument('--match_with_top_layers', action='store_true',
                      help="Uses top <--n_matching_layers> layers to perform feature matching.")
  parser.add_argument('--m_avg_alpha', type=float, default=1.0,
                      help='Term used to balance the trade-off in the regular moving average.')
  parser.add_argument('--m_avg_lr', type=float, default=1e-5,
                      help='Learning rate for moving average, default=0.0002')
  parser.add_argument('--update_type', type=str, default='adam_ma',
                      choices=['adam_ma', 'regular_ma', 'no_ma'],
                      help="Use regular moving average instead of Adam-based moving average")
  parser.add_argument('--n_classes_in_enc', type=int, default=1_000,
                      help="Number of classes in the feature extractor classifier.")
  # DE FACTO CONSTANTS HAVE BEEN MOVED TO model_builder.model_constants()
  # parser.add_argument('--n_enc_extra_layers', type=int, default=2,
  #                     help="Number of extra layers in the encoder.")
  # parser.add_argument('--n_dec_extra_layers', type=int, default=2,
  #                     help="Number of extra layers in the generator.")
  # parser.add_argument('--vgg_classifier_depth', type=int, default=1,
  #                     help="Number of fully connected layers in the VGG classifier.")
  # parser.add_argument('--no_adapt_filter_size', action='store_true',
  #                     help="Does not use a different number of filters for each conv. layer "
  #                          "[Resnet generator only].")
  # parser.add_argument('--use_conv_at_gen_skip_conn', action='store_true',
  #                     help="For Resnet generator, applies a conv. layer to the input "
  #                          "before upsampling in the skip connection.")
  # parser.add_argument('--save_feature_extractor', action='store_true',
  #                help="Saves the feature extractor model to directory specified by --log_dir.")

  arg = parser.parse_args()
  arg.log_messages = set_arg_dependencies(arg)
  return arg


def set_dp_noise(arg, log_messages):
  assert arg.dp_tgt_delta is not None, 'tgt_delta must be set to calculate noise'
  assert arg.dp_noise is None, "Don't set dp_noise if setting dp_target_eps!"
  if arg.validation_mode == 'off':
    if arg.matched_moments == 'mean':
      eps, sig = find_single_release_sigma(arg.dp_tgt_eps, arg.dp_tgt_delta)
    elif arg.matched_moments == 'm1_and_m2':
      eps, sig = find_two_release_sigma(arg.dp_tgt_eps, arg.dp_tgt_delta, arg.dp_scale_var_sigma)
    else:
      raise NotImplementedError(f'matched moments {arg.matched_moments} not supported yet')
    log_messages.append(('info', f'Target eps ({arg.dp_tgt_eps}): using sig={sig} gives ({eps}, '
                                 f'{arg.dp_tgt_delta})-DP'))
    arg.dp_noise = sig
  elif arg.val_data == 'train':
    if arg.matched_moments == 'mean':
      eps, sig_t, sig_v = find_train_val_sigma_m1(arg.dp_tgt_eps, arg.dp_tgt_delta,
                                                  arg.dp_val_noise_scaling)
    elif arg.matched_moments == 'm1_and_m2':
      eps, sig_t, sig_v = find_train_val_sigma_m1m2(arg.dp_tgt_eps, arg.dp_tgt_delta,
                                                    arg.dp_scale_var_sigma,
                                                    arg.dp_val_noise_scaling)
    else:
      raise NotImplementedError(f'matched moments {arg.matched_moments} not supported yet')
    log_messages.append(('info', f'Target eps ({arg.dp_tgt_eps}): using sig={sig_t, sig_v} '
                                 f'gives ({eps}, {arg.dp_tgt_delta})-DP'))
    arg.dp_noise = sig_t
    arg.dp_val_noise = sig_v
  elif arg.validation_mode == 'test':
    raise NotImplementedError
  elif arg.validation_mode == 'split':
    raise NotImplementedError
  else:
    raise ValueError


def set_arg_dependencies(arg):
  log_messages = []  # as logger may not be configured yet
  if arg.exp_name is not None:
    base_dir = get_base_log_dir()
    assert base_dir is not None
    arg.log_dir = os.path.join(base_dir, arg.exp_name)

  if arg.synth_dataset_size is None:
    assert arg.fid_dataset_size is not None
    if arg.labeled:
      assert arg.downstream_dataset_size is not None
      arg.synth_dataset_size = max([arg.fid_dataset_size, arg.downstream_dataset_size])
    else:
      arg.synth_dataset_size = arg.fid_dataset_size

  if arg.dp_tgt_eps is not None:
    set_dp_noise(arg, log_messages)

  if arg.dp_sens_bound is not None:
    assert arg.dp_mean_bound is None
    assert arg.dp_var_bound is None
    arg.dp_mean_bound = arg.dp_sens_bound
    arg.dp_var_bound = arg.dp_sens_bound

  if arg.dp_mean_bound <= 0:
    arg.dp_mean_bound = None
  if arg.dp_var_bound <= 0:
    arg.dp_var_bound = None

  if arg.dp_noise is not None:
    assert arg.dp_mean_bound is not None
    if arg.matched_moments != 'mean':
      assert arg.dp_var_bound is not None
  if arg.dp_mean_bound is not None:
    assert arg.matched_moments != 'mean_and_var'

  if arg.center_crop_size == 0:
    arg.center_crop_size = arg.image_size
  if arg.synth_dataset_size is None or arg.synth_dataset_size < 5000:
    log_messages.append(('warning', f'Synth dataset size to small for FID '
                                    f'(was {arg.synth_dataset_size}, setting to 5000'))
    arg.synth_dataset_size = 5000

  if arg.restart_iter is not None and 0 < arg.restart_iter:
    if arg.restart_iter < arg.ckpt_iter:
      log_messages.append(('warning', f'ckpt_iter ({arg.ckpt_iter} < restart_iter '
                                      f'({arg.restart_iter}) set to {arg.restart_iter}'))
      arg.ckpt_iter = arg.restart_iter
    assert arg.restart_iter % arg.ckpt_iter == 0, 'restart_iter must be multiple of ckpt_iter'
  assert arg.new_ckpt_iter % arg.ckpt_iter == 0, 'new_ckpt_iter must be multiple of ckpt_iter'

  pretrain_assignments = {'cifar10': 'imagenet', 'lsun': 'imagenet', 'celeba': 'imagenet',
                          'dmnist': 'svhn', 'fmnist': 'cifar10_pretrain'}
  if arg.pretrain_dataset is None:
    arg.pretrain_dataset = pretrain_assignments[arg.dataset]
  else:
    assert arg.pretrain_dataset == pretrain_assignments[arg.dataset]

  if arg.dataset in {'dmnist', 'fmnist'}:
    arg.net_gen_type = 'condconvgen'
    arg.image_size = 28

  if arg.n_split_layers is not None:
    assert arg.labeled

  if arg.val_enc is not None:  # pruning and layer splitting not supported
    assert arg.n_split_layers is None
    assert arg.channel_filter_rate == 1.
    assert arg.val_data in {'train', 'test'}
    assert arg.valid_iter == arg.syn_eval_iter

  if arg.net_gen_type.startswith('static'):
    dset_samples = int(arg.net_gen_type[len('static'):])
    assert dset_samples == arg.batch_size
    assert arg.dataset == 'cifar10'
    assert arg.labeled

  return log_messages


def get_imagenet_norm_min_and_range(device):
  imagenet_norm_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
  imagenet_norm_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
  imagenet_norm_min = -imagenet_norm_mean / imagenet_norm_std
  imagenet_norm_max = (1.0 - imagenet_norm_mean) / imagenet_norm_std
  imagenet_norm_range = imagenet_norm_max - imagenet_norm_min

  imagenet_norm_min = pt.tensor(imagenet_norm_min, dtype=pt.float32, device=device)
  imagenet_norm_range = pt.tensor(imagenet_norm_range, dtype=pt.float32, device=device)

  imagenet_norm_min.resize_(1, 3, 1, 1)
  imagenet_norm_range.resize_(1, 3, 1, 1)
  return imagenet_norm_min, imagenet_norm_range


class DPParams(NamedTuple):
  tgt_eps: float
  tgt_delta: float
  noise: float
  mean_bound: float
  var_bound: float
  bound_type: str
  scale_var_sigma: bool


class EventSteps(NamedTuple):
  final: int
  restart: int
  ckpt: int
  new_ckpt: int
  valid: int
  tb_log: int
  syn_eval: int


def get_param_group_tuples(ar):
  dp_params = DPParams(ar.dp_tgt_eps, ar.dp_tgt_delta, ar.dp_noise,
                       ar.dp_mean_bound, ar.dp_var_bound, ar.dp_sens_bound_type,
                       ar.dp_scale_var_sigma)
  val_dp_params = DPParams(ar.dp_tgt_eps, ar.dp_tgt_delta, ar.dp_val_noise,
                           ar.dp_mean_bound, ar.dp_var_bound, ar.dp_sens_bound_type,
                           ar.dp_scale_var_sigma)
  event_steps = EventSteps(ar.n_iter, ar.restart_iter, ar.ckpt_iter, ar.new_ckpt_iter,
                           ar.valid_iter, ar.tensorboard_log_iter, ar.syn_eval_iter)
  return dp_params, val_dp_params, event_steps
