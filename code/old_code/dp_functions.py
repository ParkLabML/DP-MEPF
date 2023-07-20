import torch as pt
import math
from util_logging import log_noise_norm


def bound_sensitivity_per_sample(x, c, bound_type):
  """
  takes minibatch tensor x and ensures L2 norm of each x_i along first dimension <= c
  x: list of minibatch tensors with samples along dim 0 for each layer
  c: sensitivity bound (sensitivity is 2c for replacement relation or c for inclusion/exclusion)
  bound_type: specifies how to bound. either norm or clip for now.
  """
  assert bound_type in {'norm', 'clip', 'norm_layer', 'clip_layer'}
  if bound_type == 'norm':
    x = pt.cat(x, dim=1)
    l2_norms = pt.linalg.norm(x, dim=1)
    x_bounded = x * (c / l2_norms)[:, None]
  elif bound_type == 'clip':
    x = pt.cat(x, dim=1)
    l2_norms = pt.linalg.norm(x, dim=1)
    x_bounded = x * pt.clamp_max(c / l2_norms, 1.)[:, None]
  elif bound_type == 'norm_layer':
    l2_norms = [pt.linalg.norm(k, dim=1) for k in x]
    x_bounded = [k * (c / n)[:, None] for k, n in zip(x, l2_norms)]
    x_bounded = pt.cat(x_bounded, dim=1)
  elif bound_type == 'clip_layer':
    l2_norms = [pt.linalg.norm(k, dim=1) for k in x]
    x_bounded = [k * pt.clamp_max(c / n, 1.)[:, None] for k, n in zip(x, l2_norms)]
    x_bounded = pt.cat(x_bounded, dim=1)
  else:
    raise ValueError
  return x_bounded, l2_norms


def dp_feature_release(feats, noise_sigma, l2_norm_bound, n_samples, n_bounded_components,
                       return_noise=False, neighbouring_relation='swap'):
  assert neighbouring_relation in {'swap', 'add_remove'}
  sens_factor = 2. if neighbouring_relation == 'swap' else 1.
  effective_bound = l2_norm_bound * math.sqrt(n_bounded_components)
  sens = sens_factor * effective_bound / n_samples
  if not isinstance(sens, float) and len(sens.shape) == 1:
    sens = sens[:, None]
  noise_vec = pt.randn_like(feats) * (noise_sigma * sens)
  feats_pert = feats + noise_vec
  if not return_noise:
    return feats_pert
  else:
    return feats_pert, noise_vec


def dp_dataset_feature_release(feature_embeddings, feat_l2_norms, dp_params, matched_moments,
                               n_samples, writer):
  n_bounded_components = len(feat_l2_norms) if isinstance(feat_l2_norms, list) else 1
  data_norm = pt.linalg.norm(feature_embeddings.real_means)
  feat_means, noise_vec = dp_feature_release(feature_embeddings.real_means, dp_params.noise,
                                             dp_params.mean_bound, n_samples,
                                             n_bounded_components,
                                             return_noise=True)
  log_noise_norm(noise_vec, data_norm, writer, prefix='DP_mean')

  if matched_moments == 'm1_and_m2':
    second_moment_sens = dp_params.var_bound / 2  # / 2 because var is positive
    data_norm = pt.linalg.norm(feature_embeddings.real_vars)
    var_noise = dp_params.noise * dp_params.scale_var_sigma
    feat_vars, noise_vec = dp_feature_release(feature_embeddings.real_vars, var_noise,
                                              second_moment_sens, n_samples,
                                              n_bounded_components,
                                              return_noise=True)
    log_noise_norm(noise_vec, data_norm, writer, prefix='DP_var')
  else:
    feat_vars = None

  return feat_means, feat_vars
