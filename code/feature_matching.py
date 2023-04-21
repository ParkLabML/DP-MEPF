import torch as pt
from torch.nn.functional import mse_loss
from util import GeneratorNoiseMaker, Embeddings
from util_logging import LOG, log_real_feature_norms
from dp_functions import bound_sensitivity_per_sample
from data_loading import load_dataset
from models.encoders_class import Encoders
from models.model_builder import StaticDataset


def regular_match_loss(fake_moment, real_moment, fake_m_avg, m_avg_alpha):
  if fake_m_avg is None:
    fake_m_avg = fake_moment.clone().detach()
  fake_m_avg = (1.0 - m_avg_alpha) * fake_m_avg.detach() + m_avg_alpha * fake_moment
  loss = mse_loss(fake_m_avg, real_moment.detach())
  return loss, fake_m_avg


def regular_moving_average_update(feat_emb, m_avg, m_avg_alpha, acc_losses, optimizer_g,
                                  matched_moments):

  fake_data_feat_mean = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0), feat_emb.fake_feats)
  mean_loss, m_avg.mean = regular_match_loss(fake_data_feat_mean, feat_emb.real_means,
                                             m_avg.mean, m_avg_alpha)

  if matched_moments == 'mean':
    var_loss = 0.
  elif matched_moments == 'mean_and_var':
    fake_data_feat_var = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    var_loss, m_avg.var = regular_match_loss(fake_data_feat_var, feat_emb.real_vars,
                                             m_avg.var, m_avg_alpha)
  else:
    fake_data_feat_var = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                     feat_emb.fake_feats_sqrd)
    var_loss, m_avg.var = regular_match_loss(fake_data_feat_var, feat_emb.real_vars,
                                             m_avg.var, m_avg_alpha)

  loss_net_g = mean_loss + var_loss

  acc_losses.gen_mean += loss_net_g.item()
  loss_net_g.backward()
  optimizer_g.step()


def adam_match_loss(moment_net, fake_moment, real_moment, avg_loss_net_mom, opt_moment,
                    avg_loss_net_gen):
  moment_diff_m_avg = moment_net(None)
  diff = real_moment.detach() - fake_moment.detach()
  loss_net_moment = mse_loss(moment_diff_m_avg, diff)
  avg_loss_net_mom += loss_net_moment.item()
  loss_net_moment.backward()
  opt_moment.step()  # update moment net
  diff_true = pt.sum(moment_diff_m_avg * real_moment).detach()
  diff_fake = pt.sum(moment_diff_m_avg * fake_moment)
  # diff_true = pt.matmul(moment_diff_m_avg, real_moment.view(-1, 1)).detach()
  # diff_fake = pt.matmul(moment_diff_m_avg, fake_moment.view(-1, 1))
  loss_net_gen = (diff_true - diff_fake)  # compute loss for generator
  avg_loss_net_gen += loss_net_gen.item()
  return avg_loss_net_mom, avg_loss_net_gen, loss_net_gen


def adam_moving_average_update(feat_emb, acc_losses, m_avg, optimizers, matched_moments):

  fake_data_feat_mean = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                    feat_emb.fake_feats)
  res_mean = adam_match_loss(m_avg.mean, fake_data_feat_mean, feat_emb.real_means,
                             acc_losses.mean, optimizers.mean, acc_losses.gen_mean)
  acc_losses.mean, acc_losses.gen_mean, loss_net_g_mean = res_mean

  if matched_moments == 'mean':
    loss_net_g_var = 0.
  elif matched_moments == 'mean_and_var':
    fake_data_feat_var = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    # fake_data_feat_var = pt.var(fake_data_feats, 0)
    res_var = adam_match_loss(m_avg.net_var, fake_data_feat_var, feat_emb.real_vars,
                              acc_losses.var, optimizers.var, acc_losses.gen_var)
    acc_losses.var, acc_losses.gen_var, loss_net_g_var = res_var
  else:
    fake_data_feat_var = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                     feat_emb.fake_feats_sqrd)
    res_var = adam_match_loss(m_avg.var, fake_data_feat_var, feat_emb.real_vars,
                              acc_losses.var, optimizers.var, acc_losses.gen_var)
    acc_losses.var, acc_losses.gen_var, loss_net_g_var = res_var

  loss_net_g = loss_net_g_mean + loss_net_g_var
  loss_net_g.backward()
  optimizers.gen.step()


def reduce_feats_list_or_tensor(reduce_op, feats):
  if isinstance(feats, list):
    return pt.stack([reduce_op(k) for k in feats], dim=0)
  else:
    return reduce_op(feats)


def var_reduce_op(x):
  return pt.sum(x - pt.sum(pt.sqrt(x)), dim=0) / (x.shape[0] - 1)


def extract_features(data_batch, net_enc, channel_ids_by_enc, n_matching_layers,
                     match_with_top_layers, detach_output):
  feats_list = []
  for enc, channel_ids in zip(net_enc, channel_ids_by_enc):
    feats_per_layer = enc(data_batch)[1]
    for layer_id in range(1, n_matching_layers + 1):
      corrected_layer_id = layer_id - 1  # gets features in forward order
      if match_with_top_layers:
        corrected_layer_id = -layer_id  # gets features in backward order (last layers first)
      layer_feats = feats_per_layer[corrected_layer_id]
      if channel_ids is not None:
        layer_feats = layer_feats[:, channel_ids[corrected_layer_id], :, :]
      layer_feats = layer_feats.view(layer_feats.shape[0], -1)
      if detach_output:
        feats_list.append(layer_feats.detach())
      else:
        feats_list.append(layer_feats)
  return feats_list


def extract_features_with_hooks(data_batch, encoders: Encoders):
  # feats_list = []
  for encoder_name in encoders.models:
    encoders.models[encoder_name](data_batch)
  if encoders.prune_mode is not None:
    pruned_feats = []
    for layer_name in encoders.layer_feats:
      feats = encoders.layer_feats[layer_name]
      if encoders.prune_mode == 'channels':
        channel_ids = encoders.channel_ids[layer_name]
        pruned_feats.append(feats[:, channel_ids, :, :])
      elif encoders.prune_mode == 'weights':
        weight_ids = encoders.weight_ids[layer_name]
        pruned_feats.append(feats.contiguous().view(feats.shape[0], -1)[:, weight_ids])
      else:
        return ValueError
    # if flatten_feats:
    return [k.contiguous().view(k.shape[0], -1) for k in pruned_feats]
    # else:
    #   return [k.contiguous() for k in pruned_feats]
  else:
    # LOG.warn(f'shapes {[k.shape for k in encoders.layer_feats.values()]}')
    # if flatten_feats:
    return [k.contiguous().view(k.shape[0], -1) for k in encoders.layer_feats.values()]
    # else:
    #   return [k.contiguous() for k in encoders.layer_feats.values()]


def extract_and_bound_features(data_batch, net_enc, n_matching_layers, match_with_top_layers,
                               dp_params, channel_ids_by_enc, do_second_moment, detach_output=False,
                               compute_norms=False):
  """
  Applies feature extractor. Concatenate feature vectors from all selected layers.
  """
  # channel_mean_norms, channel_sqrd_norms = None, None
  # gets features from each layer of net_enc
  if isinstance(net_enc, list):
    feats_list = extract_features(data_batch, net_enc, channel_ids_by_enc, n_matching_layers,
                                  match_with_top_layers, detach_output)
  else:
    feats_list = extract_features_with_hooks(data_batch, net_enc)

  if dp_params.mean_bound is not None:
    feats, l2_norms = bound_sensitivity_per_sample(feats_list, dp_params.mean_bound,
                                                   dp_params.bound_type)
    if dp_params.var_bound is not None and do_second_moment:
      feats_sqrd, l2_norms_sqrd = bound_sensitivity_per_sample([f**2 for f in feats_list],
                                                               dp_params.var_bound,
                                                               dp_params.bound_type)
    else:
      feats_sqrd, l2_norms_sqrd = None, None
  else:
    feats = pt.cat(feats_list, dim=1)
    feats_sqrd = feats**2 if do_second_moment else None
    if compute_norms:
      l2_norms = pt.linalg.norm(feats.view(feats.shape[0], -1), dim=1)
      if feats_sqrd is not None:
        l2_norms_sqrd = pt.linalg.norm(feats.view(feats_sqrd.shape[0], -1), dim=1)
      else:
        l2_norms_sqrd = None
    else:
      l2_norms, l2_norms_sqrd = None, None

  return feats, feats_sqrd, l2_norms, l2_norms_sqrd


def compute_data_embedding(net_enc, n_matching_layers, device, dataloader, writer,
                           channel_ids_by_enc, dp_params, no_cuda, match_with_top_layers,
                           matched_moments, n_classes):
  if n_classes is None:
    return compute_unlabeled_data_embedding(net_enc, n_matching_layers, device, dataloader, writer,
                                            channel_ids_by_enc, no_cuda, match_with_top_layers,
                                            dp_params, matched_moments)
  elif isinstance(net_enc, Encoders) and (net_enc.n_split_layers is not None):
    return compute_hybrid_labeled_data_embedding(net_enc, n_matching_layers, device, dataloader,
                                                 writer, channel_ids_by_enc, no_cuda,
                                                 match_with_top_layers, dp_params, matched_moments,
                                                 n_classes)
  else:
    return compute_labeled_data_embedding(net_enc, n_matching_layers, device, dataloader, writer,
                                          channel_ids_by_enc, no_cuda, match_with_top_layers,
                                          dp_params, matched_moments, n_classes)


def compute_labeled_data_embedding(net_enc, n_matching_layers, device, dataloader, writer,
                                   channel_ids_by_enc, no_cuda,
                                   match_with_top_layers, dp_params, matched_moments, n_classes):
  do_second_moment = matched_moments in {'mean_and_var', 'm1_and_m2'}
  feat_sum = None
  feat_sqrd_sum = None
  l2_norms_list = []
  l2_norms_sqrd_list = []

  to_one_hot = pt.eye(n_classes, device=device)
  n_examples = pt.zeros(n_classes, device=device)
  LOG.info("Computing mean features from TRUE data")
  for i, data in enumerate(dataloader, 1):
    # gets real images
    x_in, y_in = data
    if not no_cuda:
      x_in = x_in.to(device)
      y_in = y_in.to(device)
    y_one_hot = to_one_hot[y_in]

    n_examples += pt.sum(y_one_hot, dim=0)  # n_examples per class
    # extracts features for TRUE data
    ef_res = extract_and_bound_features(x_in, net_enc, n_matching_layers, match_with_top_layers,
                                        dp_params, channel_ids_by_enc, do_second_moment,
                                        detach_output=True, compute_norms=True)
    feats_batch, feats_batch_sqrd, l2_batch, l2_batch_sqrd = ef_res
    l2_norms_list.append(l2_batch)
    l2_norms_sqrd_list.append(l2_batch_sqrd)

    batch_sum, batch_sqrd_sum, l2_batch = sum_feats_by_label(feats_batch, feats_batch_sqrd,
                                                             l2_batch, y_one_hot)

    feat_sum, feat_sqrd_sum = acc_batch_feats(batch_sum, batch_sqrd_sum, feat_sum, feat_sqrd_sum)

  if isinstance(l2_norms_list[0], pt.Tensor):  # global l2 norms
    l2_norms = pt.cat(l2_norms_list, dim=0)
    if isinstance(l2_norms_sqrd_list[0], pt.Tensor):
      l2_norms_sqrd = pt.cat(l2_norms_sqrd_list, dim=0)
    else:
      l2_norms_sqrd = None
    log_real_feature_norms(l2_norms, l2_norms_sqrd, writer)

  else:  # l2 norms per layer
    l2_norms = []
    for idx in range(len(l2_norms_list[0])):
      l2_norms.append(pt.cat([n[idx] for n in l2_norms_list], dim=0))

  LOG.info("Normalizing sum of features with denominator: {}".format(n_examples))

  n_examples_view = n_examples[:, None]  # (n_exmpl) -> (n_feats x n_exmpl)
  feat_mean = feat_sum / n_examples_view
  if matched_moments == 'mean_and_var':
    feat_vars = (feat_sqrd_sum - (feat_sum ** 2) / n_examples_view) / (n_examples_view - 1)
  elif matched_moments == 'm1_and_m2':
    feat_vars = feat_sqrd_sum / n_examples_view
  else:
    feat_vars = None

  return feat_mean, feat_vars, n_examples, l2_norms


def compute_unlabeled_data_embedding(net_enc, n_matching_layers, device, dataloader, writer,
                                     channel_ids_by_enc, no_cuda,
                                     match_with_top_layers, dp_params, matched_moments):
  do_second_moment = matched_moments in {'mean_and_var', 'm1_and_m2'}
  feat_sum = None
  feat_sqrd_sum = None
  l2_norms_list = []
  l2_norms_sqrd_list = []
  # channel_mean_norms_acc, channel_sqrd_norms_acc = [], []
  n_examples = 0.
  LOG.info("Computing mean features from TRUE data")
  for i, data in enumerate(dataloader, 1):
    # gets real images
    x_in, y_in = data
    if not no_cuda:
      x_in = x_in.to(device)

    n_examples += x_in.size()[0]

    # extracts features for TRUE data
    ef_res = extract_and_bound_features(x_in, net_enc, n_matching_layers, match_with_top_layers,
                                        dp_params, channel_ids_by_enc,
                                        do_second_moment, detach_output=True, compute_norms=True)
    feats_batch, feats_batch_sqrd, l2_batch, l2_batch_sqrd, = ef_res
    l2_norms_list.append(l2_batch)
    l2_norms_sqrd_list.append(l2_batch_sqrd)

    batch_sum = pt.sum(feats_batch, dim=0).detach()
    if do_second_moment:
      batch_sqrd_sum = pt.sum(feats_batch_sqrd, dim=0).detach()
    else:
      batch_sqrd_sum = None

    feat_sum, feat_sqrd_sum = acc_batch_feats(batch_sum, batch_sqrd_sum, feat_sum, feat_sqrd_sum)

  # if isinstance(l2_norms_list[0], pt.Tensor):  # global l2 norms
    # l2_norms = pt.cat(l2_norms_list, dim=0)
  if isinstance(l2_norms_list[0], pt.Tensor):  # global l2 norms
    l2_norms = pt.cat(l2_norms_list, dim=0)
    if isinstance(l2_norms_sqrd_list[0], pt.Tensor):
      l2_norms_sqrd = pt.cat(l2_norms_sqrd_list, dim=0)
    else:
      l2_norms_sqrd = None
    log_real_feature_norms(l2_norms, l2_norms_sqrd, writer)
  else:  # l2 norms per layer
    l2_norms = []
    for idx in range(len(l2_norms_list[0])):
      l2_norms.append(pt.cat([n[idx] for n in l2_norms_list], dim=0))

  LOG.info("Normalizing sum of features with denominator: {}".format(n_examples))

  feat_mean = feat_sum / n_examples
  if matched_moments == 'mean_and_var':
    feat_vars = (feat_sqrd_sum - (feat_sum ** 2) / n_examples) / (n_examples - 1)
  elif matched_moments == 'm1_and_m2':
    feat_vars = feat_sqrd_sum / n_examples
  else:
    feat_vars = None

  return feat_mean, feat_vars, n_examples, l2_norms


def l2_norm_feat_analysis(feat_mean, feat_vars, n_feats_by_layer, layer_feats, writer):
  # ensure do_second_moment is true, dp_mean_bound and dp_var_bound should be none.
  # to sort into layers, we still need the features per layer, which encoders has
  n_feats_by_l = [k for k in n_feats_by_layer.values()]
  layer_names = [k for k in n_feats_by_layer.keys()]
  last_feat_by_l = [sum(n_feats_by_l[:k + 1]) for k in range(len(n_feats_by_l))]
  id_tuples = list(zip([0] + last_feat_by_l[:-1], last_feat_by_l))

  layer_shapes = [[j for j in k.shape[1:]] for k in layer_feats.values()]

  print(id_tuples, last_feat_by_l)
  mean_norms_acc = []
  vars_norms_acc = []
  for (id_first, id_last), l_name, l_shape in zip(id_tuples, layer_names, layer_shapes):
    print((id_first, id_last), l_name, l_shape)
    l_feats_mean = pt.reshape(feat_mean[id_first:id_last], [-1] + l_shape[1:])  # n_channels varies
    l_feats_vars = pt.reshape(feat_vars[id_first:id_last], [-1] + l_shape[1:])  # n_channels varies
    n_channels = l_feats_mean.shape[0]
    print(n_channels)
    l_feats_mean_norm = pt.norm(pt.reshape(l_feats_mean, (n_channels, -1)), dim=1)
    l_feats_vars_norm = pt.norm(pt.reshape(l_feats_vars, (n_channels, -1)), dim=1)
    print(l_feats_mean_norm.shape)
    writer.add_histogram(f'real_channel_norms_mean/{l_name}', l_feats_mean_norm, global_step=0)
    writer.add_histogram(f'real_channel_norms_var/{l_name}', l_feats_vars_norm, global_step=0)

    mean_norms_acc.append(l_feats_mean_norm)
    vars_norms_acc.append(l_feats_vars_norm)
  mean_norms_acc = pt.cat(mean_norms_acc)
  vars_norms_acc = pt.cat(vars_norms_acc)
  writer.add_histogram(f'real_channel_norms_mean/all_layers', mean_norms_acc,
                       global_step=0)
  writer.add_histogram(f'real_channel_norms_var/all_layers', vars_norms_acc,
                       global_step=0)

  writer.add_histogram(f'real_channel_norms_mean/all_layers_log', pt.log(mean_norms_acc),
                       global_step=0)
  writer.add_histogram(f'real_channel_norms_var/all_layers_log', pt.log(vars_norms_acc),
                       global_step=0)
  mean_norms_bounded = pt.clamp(mean_norms_acc,
                                min=pt.quantile(mean_norms_acc, 0.01),
                                max=pt.quantile(mean_norms_acc, 0.99))
  vars_norms_bounded = pt.clamp(vars_norms_acc,
                                min=pt.quantile(vars_norms_acc, 0.01),
                                max=pt.quantile(vars_norms_acc, 0.99))
  writer.add_histogram(f'real_channel_norms_mean/all_layers_98%', mean_norms_bounded,
                       global_step=0)
  writer.add_histogram(f'real_channel_norms_var/all_layers_98%', vars_norms_bounded,
                       global_step=0)
  writer.close()
  assert 1 % 1 == 1




def hybrid_labeled_batch_embedding(encoders, feats_batch, feats_batch_sqrd, l2_norms,
                                   do_second_moment, n_classes, y_one_hot):
  # ef_res = extract_and_bound_features(x_in, encoders, n_matching_layers, match_with_top_layers,
  #                                     dp_params, channel_ids_by_enc, do_second_moment,
  #                                     detach_output=True, compute_norms=True)
  n_shared_features = encoders.n_feats_total - encoders.n_split_features * n_classes
  n_shared_split = [n_shared_features, encoders.n_split_features]
  feats_batch_shared, feats_batch_split = pt.split(feats_batch, n_shared_split, dim=1)
  if feats_batch_sqrd is not None:
    feats_batch_shared_sqrd, feats_batch_sqrd_split = pt.split(feats_batch_sqrd, n_shared_split,
                                                               dim=1)
  else:
    feats_batch_shared_sqrd, feats_batch_sqrd_split = None, None
  batch_sum_shared = pt.sum(feats_batch_shared, dim=0).detach()

  batch_sum_split, batch_sqrd_sum_split, _ = sum_feats_by_label(feats_batch_split,
                                                                feats_batch_sqrd_split,
                                                                l2_norms, y_one_hot)

  LOG.debug(f'mean embedding - shared shape: {batch_sum_shared.shape} '
            f'/ split shape {batch_sum_split.shape}')
  batch_sum = pt.cat([batch_sum_shared, batch_sum_split.flatten()], dim=0)
  if do_second_moment:
    batch_sqrd_sum_shared = pt.sum(feats_batch_shared_sqrd, dim=0).detach()
    batch_sqrd_sum = pt.cat([batch_sqrd_sum_shared, batch_sqrd_sum_split.flatten()], dim=0)
  else:
    batch_sqrd_sum = None
  return batch_sum, batch_sqrd_sum


def compute_hybrid_labeled_data_embedding(encoders: Encoders, n_matching_layers, device, dataloader,
                                          writer, channel_ids_by_enc, no_cuda,
                                          match_with_top_layers, dp_params, matched_moments,
                                          n_classes):
  assert encoders.n_split_layers > 0

  LOG.info(f'using last {encoders.n_split_features} features for labeled training')
  do_second_moment = matched_moments in {'mean_and_var', 'm1_and_m2'}
  feat_sum = None
  feat_sqrd_sum = None
  l2_norms_list = []
  l2_norms_sqrd_list = []

  to_one_hot = pt.eye(n_classes, device=device)
  n_examples = pt.zeros(n_classes, device=device)
  LOG.info("Computing mean features from TRUE data")
  for i, data in enumerate(dataloader, 1):
    # gets real images
    x_in, y_in = data
    if not no_cuda:
      x_in = x_in.to(device)
      y_in = y_in.to(device)
    y_one_hot = to_one_hot[y_in]

    n_examples += pt.sum(y_one_hot, dim=0)  # n_examples per class
    # extracts features for TRUE data
    ef_res = extract_and_bound_features(x_in, encoders, n_matching_layers, match_with_top_layers,
                                        dp_params, channel_ids_by_enc, do_second_moment,
                                        detach_output=True, compute_norms=True)
    feats_batch, feats_batch_sqrd, l2_norms, l2_norms_sqrd = ef_res
    ff, ff_sq = hybrid_labeled_batch_embedding(encoders, feats_batch, feats_batch_sqrd, l2_norms,
                                               do_second_moment, n_classes, y_one_hot)
    batch_sum, batch_sqrd_sum = ff, ff_sq
    l2_norms_list.append(l2_norms)
    l2_norms_sqrd_list.append(l2_norms_sqrd)
    feat_sum, feat_sqrd_sum = acc_batch_feats(batch_sum, batch_sqrd_sum, feat_sum, feat_sqrd_sum)

  if isinstance(l2_norms_list[0], pt.Tensor):  # global l2 norms
    l2_norms = pt.cat(l2_norms_list, dim=0)
    if isinstance(l2_norms_sqrd_list[0], pt.Tensor):
      l2_norms_sqrd = pt.cat(l2_norms_sqrd_list, dim=0)
    else:
      l2_norms_sqrd = None
    log_real_feature_norms(l2_norms, l2_norms_sqrd, writer)

  else:  # l2 norms per layer
    l2_norms = []
    for idx in range(len(l2_norms_list[0])):
      l2_norms.append(pt.cat([n[idx] for n in l2_norms_list], dim=0))

  LOG.info("Normalizing sum of features with denominator: {}".format(n_examples))

  n_examples_view = n_examples[:, None]  # (n_exmpl) -> (n_feats x n_exmpl)
  feat_mean = feat_sum / n_examples_view
  if matched_moments == 'mean_and_var':
    feat_vars = (feat_sqrd_sum - (feat_sum ** 2) / n_examples_view) / (n_examples_view - 1)
  elif matched_moments == 'm1_and_m2':
    feat_vars = feat_sqrd_sum / n_examples_view
  else:
    feat_vars = None

  return feat_mean, feat_vars, n_examples, l2_norms


def acc_batch_feats(batch_sum, batch_sqrd_sum, feat_sum, feat_sqrd_sum):
  if feat_sum is None:
    feat_sum = batch_sum
  else:
    feat_sum += batch_sum

  if batch_sqrd_sum is not None:
    if feat_sqrd_sum is None:
      feat_sqrd_sum = batch_sqrd_sum
    else:
      feat_sqrd_sum += batch_sqrd_sum
  return feat_sum, feat_sqrd_sum


def sort_feats_by_label(feats, y_one_hot):
  return [feats[pt.argmax(y_one_hot, dim=1) == k] for k in range(y_one_hot.shape[1])]


def sum_feats_by_label(feats, feats_sqrd, l2_batch, y_one_hot):

  batch_sum = pt.einsum('ki,kj->ij', [y_one_hot, feats])
  if feats_sqrd is not None:
    batch_sqrd_sum = pt.einsum('ki,kj->ij', [y_one_hot, feats_sqrd])
  else:
    batch_sqrd_sum = None
  l2_batch_by_label = pt.einsum('ki,k->ki', [y_one_hot, l2_batch])

  return batch_sum, batch_sqrd_sum, l2_batch_by_label


def get_test_data_embedding(net_enc, n_matching_layers, device,
                            channel_ids_by_enc, no_cuda, match_with_top_layers,
                            dp_params, matched_moments, data_scale,
                            n_classes, dataset, image_size, center_crop_size, dataroot,
                            batch_size, n_workers, labeled, val_data):
  assert val_data in {'train', 'test'}
  use_test_set = val_data == 'test'
  test_loader, _ = load_dataset(dataset, image_size, center_crop_size, dataroot,
                                batch_size, n_workers, data_scale, labeled,
                                test_set=use_test_set)
  emb_res = compute_data_embedding(net_enc, n_matching_layers, device, test_loader, None,
                                   channel_ids_by_enc, dp_params, no_cuda, match_with_top_layers,
                                   matched_moments, n_classes)
  test_feat_means, test_feat_vars, _, _ = emb_res
  return test_feat_means, test_feat_vars


def regular_moving_average_valid(feat_emb, m_avg, m_avg_alpha, acc_losses, matched_moments):

  fake_data_feat_mean = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                    feat_emb.fake_feats)
  mean_loss, m_avg.mean = regular_match_loss(fake_data_feat_mean, feat_emb.real_means,
                                             m_avg.mean, m_avg_alpha)

  if matched_moments == 'mean':
    var_loss = 0.
  elif matched_moments == 'mean_and_var':
    fake_data_feat_var = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    var_loss, m_avg.var = regular_match_loss(fake_data_feat_var, feat_emb.real_vars,
                                             m_avg.var, m_avg_alpha)
  else:
    fake_data_feat_var = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                     feat_emb.fake_feats_sqrd)
    var_loss, m_avg.var = regular_match_loss(fake_data_feat_var, feat_emb.real_vars,
                                             m_avg.var, m_avg_alpha)
  acc_losses.gen_mean += (mean_loss + var_loss).item()


def adam_moving_average_valid(feat_emb, acc_losses, m_avg, optimizers, matched_moments):

  fake_data_feat_mean = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                    feat_emb.fake_feats)
  res_mean = adam_match_loss(m_avg.mean, fake_data_feat_mean, feat_emb.real_means,
                             acc_losses.mean, optimizers.mean, acc_losses.gen_mean)
  acc_losses.mean, acc_losses.gen_mean, _ = res_mean

  if matched_moments == 'mean':
    pass
  elif matched_moments == 'mean_and_var':
    fake_data_feat_var = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    # fake_data_feat_var = pt.var(fake_data_feats, 0)
    res_var = adam_match_loss(m_avg.var, fake_data_feat_var, feat_emb.real_vars,
                              acc_losses.var, optimizers.var, acc_losses.gen_var)
    acc_losses.var, acc_losses.gen_var, _ = res_var
  else:
    fake_data_feat_var = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
                                                     feat_emb.fake_feats_sqrd)
    res_var = adam_match_loss(m_avg.var, fake_data_feat_var, feat_emb.real_vars,
                              acc_losses.var, optimizers.var, acc_losses.gen_var)
    acc_losses.var, acc_losses.gen_var, _ = res_var


def get_static_losses_valid(means_valid, vars_valid, gen, encoders, noise_maker: GeneratorNoiseMaker,
                            batch_size, n_samples, n_classes, device, n_matching_layers,
                            match_with_top_layers, dp_params, channel_ids_by_enc, do_second_moment,
                            writer, step):
  # make a small synthetic dataset and collect embeddings
  feat_sum = None
  feat_sqrd_sum = None

  batches = [batch_size] * (n_samples // batch_size)
  if n_samples % batch_size > 0:
    batches.append(n_samples % batch_size)
  if n_classes is not None:
    balanced_labels_int = pt.repeat_interleave(pt.arange(0, n_classes, dtype=pt.int64,
                                                         device=device),
                                               n_samples // n_classes)
    random_labels_int = pt.randint(n_classes, (n_samples % n_classes,),
                                   dtype=pt.int64, device=device)
    labels_int = pt.cat([balanced_labels_int, random_labels_int])
    labels_list = pt.split(pt.eye(n_classes, device=device)[labels_int], batches)
  else:
    labels_list = [None] * len(batches)

  with pt.no_grad():
    for labels, bs in zip(labels_list, batches):
      z_in = noise_maker.noise_fun(labels=labels, batch_size=bs)
      if isinstance(gen, pt.nn.parallel.DistributedDataParallel):
        syn_batch = gen.module(z_in)  # don't sync, since this only runs on one gpu
      else:
        syn_batch = gen(z_in)

      # get embedding
      ef_res = extract_and_bound_features(syn_batch, encoders, n_matching_layers,
                                          match_with_top_layers, dp_params,
                                          channel_ids_by_enc, do_second_moment)
      batch_feats, batch_feats_sqrd, l2_batch, _ = ef_res
      if n_classes is not None:
        batch_sum, batch_sqrd_sum, _ = sum_feats_by_label(batch_feats, batch_feats_sqrd, l2_batch,
                                                          labels)
      else:
        batch_sum = pt.sum(batch_feats, dim=0).detach()
        batch_sqrd_sum = pt.sum(batch_feats_sqrd, dim=0).detach() if do_second_moment else None

      feat_sum, feat_sqrd_sum = acc_batch_feats(batch_sum, batch_sqrd_sum, feat_sum, feat_sqrd_sum)

  # now compute simple loss between feat_sum and means_valid and the sqrd version
  mean_diff = pt.sum((means_valid - (feat_sum / n_samples)) ** 2)
  vars_diff = pt.sum((vars_valid - (feat_sqrd_sum / n_samples)) ** 2)

  # then write loss to tensorflow
  writer.add_scalar('val/mean_loss', mean_diff, global_step=step)
  writer.add_scalar('val/vars_loss', vars_diff, global_step=step)
  LOG.info(f'Validation Losses: m1 {mean_diff}, m2 {vars_diff}')

  return mean_diff, vars_diff


def get_staticdset_static_losses_valid(means_valid, vars_valid, gen: StaticDataset, encoders,
                                       n_matching_layers, match_with_top_layers, dp_params,
                                       channel_ids_by_enc, do_second_moment, writer, step):
  # make a small synthetic dataset and collect embeddings
  labels = gen.labels
  syn_batch = gen(None)

  with pt.no_grad():
    # get embedding
    ef_res = extract_and_bound_features(syn_batch, encoders, n_matching_layers,
                                        match_with_top_layers, dp_params,
                                        channel_ids_by_enc, do_second_moment)
    batch_feats, batch_feats_sqrd, l2_batch, _ = ef_res
    feat_sum, feat_sqrd_sum, _ = sum_feats_by_label(batch_feats, batch_feats_sqrd, l2_batch,
                                                    labels)

  mean_diff = pt.sum((means_valid - (feat_sum / labels.shape[0])) ** 2)
  vars_diff = pt.sum((vars_valid - (feat_sqrd_sum / labels.shape[0])) ** 2)

  # then write loss to tensorflow
  writer.add_scalar('val/mean_loss', mean_diff, global_step=step)
  writer.add_scalar('val/vars_loss', vars_diff, global_step=step)
  LOG.info(f'Validation Losses: m1 {mean_diff}, m2 {vars_diff}')
  return mean_diff, vars_diff


def ma_update(feat_emb, m_avg, optimizers, train_acc_losses,
              m_avg_valid, optimizers_valid, acc_losses_valid,
              update_type, ma_validation, matched_moments, m_avg_alpha, gen):
  if update_type == 'regular_ma':
    regular_moving_average_update(feat_emb, m_avg, m_avg_alpha, train_acc_losses,
                                  optimizers.gen, matched_moments)

    if ma_validation:
      regular_moving_average_valid(feat_emb, m_avg_valid, m_avg_alpha, acc_losses_valid,
                                   matched_moments)

  elif update_type == 'adam_ma':
    adam_moving_average_update(feat_emb, train_acc_losses, m_avg, optimizers, matched_moments)

    if ma_validation:
      adam_moving_average_valid(feat_emb, acc_losses_valid, m_avg_valid, optimizers_valid,
                                matched_moments)
  else:  # no_ma
    no_mavg_update(feat_emb, optimizers.gen, matched_moments, gen)
    # fake_data_feat_mean = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
    #                                                   feat_emb.fake_feats)
    # res_mean = adam_match_loss(m_avg.mean, fake_data_feat_mean, feat_emb.real_means,
    #                            acc_losses.mean, optimizers.mean, acc_losses.gen_mean)
    # acc_losses.mean, acc_losses.gen_mean, loss_net_g_mean = res_mean
    #
    # if matched_moments == 'mean':
    #   loss_net_g_var = 0.
    # elif matched_moments == 'mean_and_var':
    #   fake_data_feat_var = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    #   # fake_data_feat_var = pt.var(fake_data_feats, 0)
    #   res_var = adam_match_loss(m_avg.net_var, fake_data_feat_var, feat_emb.real_vars,
    #                             acc_losses.var, optimizers.var, acc_losses.gen_var)
    #   acc_losses.var, acc_losses.gen_var, loss_net_g_var = res_var
    # else:
    #   fake_data_feat_var = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0),
    #                                                    feat_emb.fake_feats_sqrd)
    #   res_var = adam_match_loss(m_avg.var, fake_data_feat_var, feat_emb.real_vars,
    #                             acc_losses.var, optimizers.var, acc_losses.gen_var)
    #   acc_losses.var, acc_losses.gen_var, loss_net_g_var = res_var
    #
    # loss_net_g = loss_net_g_mean + loss_net_g_var
    # loss_net_g.backward()
    # optimizers.gen.step()


def no_mavg_update(feat_emb: Embeddings, opt_gen, matched_moments, gen):
  fake_means = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0), feat_emb.fake_feats)
  mean_loss = mse_loss(feat_emb.real_means, fake_means)
  if matched_moments == 'mean':
    var_loss = 0.
  elif matched_moments == 'mean_and_var':
    fake_vars = reduce_feats_list_or_tensor(var_reduce_op, feat_emb.fake_feats_sqrd)
    var_loss = mse_loss(feat_emb.real_vars, fake_vars)
  else:
    fake_vars = reduce_feats_list_or_tensor(lambda x: pt.mean(x, dim=0), feat_emb.fake_feats_sqrd)
    var_loss = mse_loss(feat_emb.real_vars, fake_vars)

  total_loss = mean_loss + var_loss
  total_loss.backward()
  # for param in gen.parameters():
  #   LOG.warning(f'{param.shape}')
  #   LOG.warning(f'{pt.norm(param.grad)}')
  #   LOG.warning(f'{pt.max(param.grad)}')
  #   LOG.warning(f'{pt.min(param.grad)}')
  opt_gen.step()
