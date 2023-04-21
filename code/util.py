import os
import os.path
import random
from typing import NamedTuple
import torch as pt
import torch.optim as optim
from dataclasses import dataclass

from data_loading import load_dataset
from util_logging import LOG, BestResult
from models.encoders_class import Encoders


@dataclass
class LossAccumulator:
    gen_mean = 0.0
    gen_var = 0.0
    mean = 0.0
    var = 0.0


@dataclass
class MovingAverages:
  mean = None
  var = None


@dataclass
class Embeddings:
  real_means = None
  real_vars = None
  fake_feats = None
  fake_feats_sqrd = None
  means_valid = None
  vars_valid = None


class Optimizers(NamedTuple):
  gen: pt.optim.Optimizer
  mean: pt.optim.Optimizer
  var: pt.optim.Optimizer


def get_image_size_n_feats(net, verbose=False, image_size=32, use_cuda=False):
  """
  return two list:
  - list of size of output (image) for each layer
  - list of size of total number of features (nFeatMaps*featmaps_height,featmaps_width)
  """
  if use_cuda:
    _, layers = net(pt.randn(1, 3, image_size, image_size).cuda())
  else:
    _, layers = net(pt.randn(1, 3, image_size, image_size))

  layer_img_size = []
  layer_num_feats = []
  for layer in reversed(layers):
    if len(layer.size()) == 4:
      layer_img_size.append(layer.size(2))
      layer_num_feats.append(layer.size(1)*layer.size(2)*layer.size(3))
    elif len(layer.size()) == 2:
      layer_img_size.append(1)
      layer_num_feats.append(layer.size(1))
    else:
      assert 0, f'not sure how to handle this layer size {layer.size()}'
  if verbose:
    LOG.info("# Layer img sizes: {}".format(layer_img_size))
    LOG.info("# Layer num feats: {}".format(layer_num_feats))

  return layer_img_size, layer_num_feats


def set_random_seed(manual_seed):
  if manual_seed is None:
    manual_seed = random.randint(1, 10000)
  LOG.info("Random Seed: {}".format(manual_seed))
  random.seed(manual_seed)
  pt.manual_seed(manual_seed)
  # if not no_cuda:
  #   pt.cuda.manual_seed_all(manual_seed)


def create_checkpoint(feat_emb, net_gen, encoders, m_avg, optimizers,
                      step, fixed_noise, log_dir, update_type, new_ckpt_iter,
                      m_avg_valid, optimizers_valid, best_result: BestResult,
                      best_proxy_result: BestResult):
  # saving models
  checkpoint_dict = dict()
  checkpoint_dict['iter'] = step
  checkpoint_dict['fixed_noise'] = fixed_noise
  checkpoint_dict['feat_means'] = feat_emb.real_means
  checkpoint_dict['feat_vars'] = feat_emb.real_vars
  checkpoint_dict['best_score'] = best_result.score
  checkpoint_dict['best_step'] = best_result.step
  checkpoint_dict['best_syn_data_file'] = best_result.data_file
  checkpoint_dict['first_best_score'] = best_result.first_local_optimum_score
  checkpoint_dict['first_best_step'] = best_result.first_local_optimum_step
  checkpoint_dict['first_best_syn_data_file'] = best_result.first_local_optimum_data_file

  if best_proxy_result is not None:
    checkpoint_dict['best_proxy_score'] = best_proxy_result.score
    checkpoint_dict['best_proxy_step'] = best_proxy_result.step
    checkpoint_dict['best_proxy_syn_data_file'] = best_proxy_result.data_file
    checkpoint_dict['first_best_proxy_score'] = best_proxy_result.first_local_optimum_score
    checkpoint_dict['first_best_proxy_step'] = best_proxy_result.first_local_optimum_step
    checkpoint_dict['first_best_proxy_syn_data_file'] = best_proxy_result.first_local_optimum_data_file

  if feat_emb.means_valid is not None:
    checkpoint_dict['feat_means_valid'] = feat_emb.means_valid
    checkpoint_dict['feat_vars_valid'] = feat_emb.vars_valid

  checkpoint_dict['net_gen'] = net_gen.state_dict()
  checkpoint_dict['opt_gen'] = optimizers.gen.state_dict()

  if update_type == 'adam_ma':
    checkpoint_dict['net_mean'] = m_avg.mean.state_dict()
    checkpoint_dict['net_var'] = m_avg.var.state_dict()
    checkpoint_dict['opt_mean'] = optimizers.mean.state_dict()
    checkpoint_dict['opt_var'] = optimizers.var.state_dict()

    if m_avg_valid.mean is not None:
      checkpoint_dict['net_mean_valid'] = m_avg_valid.mean.state_dict()
      checkpoint_dict['net_var_valid'] = m_avg_valid.var.state_dict()
      checkpoint_dict['opt_mean_valid'] = optimizers_valid.mean.state_dict()
      checkpoint_dict['opt_var_valid'] = optimizers_valid.var.state_dict()

  # save feature selection from encoders (only support Encoders class)
  if isinstance(encoders, Encoders) and (encoders.prune_mode is not None):

    checkpoint_dict['encoder_channels'] = encoders.channel_ids
    checkpoint_dict['encoder_weights'] = encoders.weight_ids
    checkpoint_dict['encoder_nfeats'] = encoders.n_feats_by_layer
    if len(encoders.pca_maps) > 0:
      pca_state_dicts = {name: pca.to_param_tuple() for (name, pca) in encoders.pca_maps.items()}
      LOG.warning(f'{pca_state_dicts}')
      checkpoint_dict['encoder_pca'] = pca_state_dicts
    else:
      checkpoint_dict['encoder_pca'] = dict()

  pt.save(checkpoint_dict, os.path.join(log_dir, 'ckpt.pt'))
  # saving model with a different suffix
  if step % new_ckpt_iter == 0:
    pt.save(checkpoint_dict, os.path.join(log_dir, f'ckpt_{step}.pt'))

  LOG.info(f'created checkopint at step {step}')


def debug_data_embedding(net_enc, n_matching_layers, x_in, arg, writer):
  assert writer is not None
  dataloader = load_dataset(arg.dataset, arg.image_size, arg.center_crop_size, arg.dataroot,
                            arg.batch_size, arg.n_workers, arg.tanh_output, arg.labeled)

  def extract_l2_norms(data_batch, net_enc_, n_matching_layers_, match_with_top_layers):
    """
    Applies feature extractor. Concatenate feature vectors from all selected layers.
    """
    # gets features from each layer of net_enc
    l2_norms = []
    tensor_shape_list = []
    for enc in net_enc_:
      feats_per_layer = enc(data_batch)[1]
      for layer_id in range(1, n_matching_layers_ + 1):
        corrected_layer_id = layer_id - 1  # gets features in forward order
        if match_with_top_layers:
          corrected_layer_id = -layer_id  # gets features in backward order (last layers first)

        layer_feats = feats_per_layer[corrected_layer_id]
        layer_feats = layer_feats.view(layer_feats.size()[0], -1).detach()
        l2_norms.append(pt.linalg.norm(layer_feats, dim=1))
        tensor_shape_list.append(feats_per_layer[corrected_layer_id].shape)

    return l2_norms, tensor_shape_list

  l2_norms_list = None
  tensor_shapes = None

  n_examples = 0.0
  LOG.info("Computing l2 features from TRUE data for debugging")
  for i, data in enumerate(dataloader, 1):
    # gets real images
    real_cpu, _ = data
    if not arg.no_cuda:
      real_cpu = real_cpu.cuda()

    x_in.resize_as_(real_cpu).copy_(real_cpu)
    n_examples += x_in.size()[0]

    # extracts features for TRUE data
    l2_batch, tensor_shapes = extract_l2_norms(x_in, net_enc, n_matching_layers,
                                               arg.match_with_top_layers)
    if l2_norms_list is None:
      l2_norms_list = l2_batch
    else:
      l2_norms_list = [pt.cat((a, b), dim=0) for (a, b) in zip(l2_norms_list, l2_batch)]
      del l2_batch
    # l2_norms_list.append(l2_batch)

  for idx, tensor in enumerate(l2_norms_list):
    writer.add_histogram(f'layer_norms/t_{idx}_shape_{tensor_shapes[idx]}', tensor, global_step=0)

  assert 0


def load_checkpoint(log_dir, selected_iteration=None, rank=None):
  """
  loads checkpoint of format 'ckpt.pt' or 'ckpt_(selected_iteration).pt'.
  ckpt.pt contains the most recent iteration.
  returns: checkpoint dict and iteration
  """
  if selected_iteration is not None:
    ckpt_path = os.path.join(log_dir, f'ckpt_{selected_iteration}.pt')
  else:
    ckpt_path = os.path.join(log_dir, 'ckpt.pt')

  if not os.path.exists(ckpt_path):
    if rank == 0:
      LOG.info('Checkpoint: starting from scratch')
    return None, 0
  else:
    if rank is not None:
      ckpt = pt.load(ckpt_path, map_location=f'cuda:{rank}')
    else:
      ckpt = pt.load(ckpt_path)

    if rank == 0:
      LOG.info(f"Checkpoint: loaded ckpt iter {ckpt['iter']}")
    return ckpt, ckpt['iter']


def get_optimizers(net_gen, gen_lr, m_avg, m_avg_lr, beta1, ckpt):
  optimizer_gen = optim.Adam(net_gen.parameters(), lr=gen_lr, betas=(beta1, 0.999))
  # optimizer_gen = optim.SGD(net_gen.parameters(), lr=gen_lr)
  if ckpt is not None:
    optimizer_gen.load_state_dict(ckpt['opt_gen'])
    # prep_opt_dict_capturable(optimizer_gen)
  # if set_capturable:
  #   optimizer_gen.param_groups[0]['capturable'] = True

  if m_avg.mean is not None and m_avg.var is not None:
    optimizer_mean = optim.Adam(m_avg.mean.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    optimizer_var = optim.Adam(m_avg.var.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    if ckpt is not None:
      optimizer_mean.load_state_dict(ckpt['opt_mean'])
      optimizer_var.load_state_dict(ckpt['opt_var'])
      # prep_opt_dict_capturable(optimizer_mean)
      # prep_opt_dict_capturable(optimizer_var)
    # if set_capturable:
    #   optimizer_mean.param_groups[0]['capturable'] = True
    #   optimizer_var.param_groups[0]['capturable'] = True
  else:
    optimizer_mean, optimizer_var = None, None

  return Optimizers(optimizer_gen, optimizer_mean, optimizer_var)


def prep_opt_dict_capturable(optimizer: pt.optim.Adam):
  # fix for this issue: https://github.com/pytorch/pytorch/issues/80809
  steps = 0
  errs = 0
  for group in optimizer.param_groups:
    for p in group['params']:
      state = optimizer.state[p]
      if len(state) > 0:
        if 'step' in state:
          value = state['step']
          steps += 1
          try:
            tst = value.cpu()
            assert pt.all(tst == value)
          except:
            errs += 1
            continue
          state['step'] = tst
  if steps > 0 or errs > 0:
    print(f'steps={steps}, errs={errs}')


def get_validation_optimizers(m_avg, m_avg_lr, beta1, ckpt):

  if m_avg.mean is not None and m_avg.var is not None:
    optimizer_mean = optim.Adam(m_avg.mean.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    optimizer_var = optim.Adam(m_avg.var.parameters(), lr=m_avg_lr, betas=(beta1, 0.999))
    if ckpt is not None:
      optimizer_mean.load_state_dict(ckpt['opt_mean_valid'])
      optimizer_var.load_state_dict(ckpt['opt_var_valid'])
      # prep_opt_dict_capturable(optimizer_mean)
      # prep_opt_dict_capturable(optimizer_var)
  else:
    optimizer_mean, optimizer_var = None, None

  return Optimizers(None, optimizer_mean, optimizer_var)


class GeneratorNoiseMaker:
  def __init__(self, default_batch_size, z_dim, device, n_classes, ckpt):
    if n_classes is None:
      normal_noise_dim = z_dim
      label_one_hots = None
    else:
      normal_noise_dim = z_dim - n_classes
      label_one_hots = pt.eye(n_classes, dtype=pt.float32, device=device)
    self.normal_noise_dim = normal_noise_dim
    self.one_hots = label_one_hots
    self.default_batch_size = default_batch_size
    self.n_classes = n_classes
    self.device = device
    if ckpt is not None:
      self.fix_noise = ckpt['fixed_noise']
    else:
      self.fix_noise = None

  def noise_fun(self, labels=None, batch_size=None, return_labels=False):
    if labels is not None:
      batch_size = labels.shape[0]
    elif batch_size is None:
      batch_size = self.default_batch_size

    normal_noise = pt.randn(batch_size, self.normal_noise_dim, 1, 1, dtype=pt.float32,
                            device=self.device)
    if self.n_classes is None:
      return normal_noise
    else:
      if labels is None:
        labels = self.one_hots[pt.randint(high=self.n_classes, size=(batch_size,))]

      noise_batch = pt.cat([labels[:, :, None, None], normal_noise], dim=1)
      if return_labels:
        return noise_batch, labels
      else:
        return noise_batch

  def generator_noise(self):
    # creates noise
    if self.n_classes is None:
      gen_noise = self.noise_fun()
      gen_labels = None
    else:
      balanced_labels = self.one_hots.repeat_interleave(self.default_batch_size // self.n_classes,
                                                        dim=0)
      random_labels = self.one_hots[pt.randint(high=self.n_classes,
                                               size=(self.default_batch_size % self.n_classes,))]
      gen_labels = pt.cat([balanced_labels, random_labels], dim=0)
      gen_noise = self.noise_fun(labels=gen_labels)
    return gen_noise, gen_labels

  def get_fixed_noise(self, batch_size=100):
    if self.fix_noise is None:

      if self.n_classes is not None:
        fixed_labels = self.one_hots.repeat_interleave(10, dim=0)
        self.fix_noise = self.noise_fun(labels=fixed_labels, batch_size=batch_size)
      else:
        self.fix_noise = self.noise_fun(batch_size=batch_size)
    return self.fix_noise
