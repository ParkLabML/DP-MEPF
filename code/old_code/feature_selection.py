import math
from collections import defaultdict
import torch as pt
from models.encoders_class import Encoders
from util_logging import LOG
# from feature_selection_hsic import get_channel_subsets_hsic_low_memory, \
#   get_weight_subsets_hsic_low_memory, get_layer_names
# from feature_selection_pca import get_pca_reduction, PCAMapping


def get_number_of_matching_layers_pytorch(encoders: Encoders, device, data_loader):
  assert data_loader is not None

  x, _ = next(data_loader.__iter__())
  encoders.load_features(x.to(device))
  n_matching_layers = len(encoders.layer_feats)
  LOG.info(f"# actual n_matching_layers: {n_matching_layers}")

  n_feat_total = 0
  for layer_name, layer_feat in encoders.layer_feats.items():
    n_feat = math.prod(layer_feat.shape[1:])
    encoders.n_feats_by_layer[layer_name] = n_feat
    n_feat_total += n_feat

  LOG.info(f"# of features total (before selection): {n_feat_total}")


def select_pruned_channels(encoders: Encoders, device, writer, channel_filter_rate, data_loader,
                           channel_filter_mode, n_select_batches, ckpt, hsic_block_size, hsic_reps,
                           hsic_max_samples, n_pca_iter):
  if ckpt is not None:
    encoders.n_feats_by_layer = ckpt['encoder_nfeats']
    encoders.channel_ids = ckpt['encoder_channels']
    encoders.weight_ids = ckpt['encoder_weights']
    # pca_state_dicts = {name: pca.to_param_tuple() for (name, pca) in encoders.pca_maps.items()}
    # encoders.pca_maps = {name: PCAMapping.from_param_tuple(p) for (name, p) in
    #                      ckpt['encoder_pca'].items()}
  elif channel_filter_rate < 1.:
    if channel_filter_mode == 'maxnorm':
      get_channel_subsets_maxnorm_pytorch(encoders, channel_filter_rate, data_loader, device,
                                          writer)
    elif channel_filter_mode == 'random':
      get_channel_subsets_random_pytorch(encoders, channel_filter_rate, data_loader, device)
    # elif channel_filter_mode == 'hsic_channels':
    #
    #   get_channel_subsets_hsic_low_memory(encoders, channel_filter_rate, data_loader, device,
    #                                       n_select_batches, hsic_max_samples, hsic_block_size,
    #                                       hsic_reps)
    # elif channel_filter_mode == 'hsic_weights':
    #   get_weight_subsets_hsic_low_memory(encoders, channel_filter_rate, data_loader, device,
    #                                      n_select_batches, hsic_block_size, hsic_reps)
    elif channel_filter_mode in {'inv_maxnorm', 'minnorm'}:
      get_channel_subsets_maxnorm_pytorch(encoders, channel_filter_rate, data_loader, device,
                                          writer, inverse_sort=True)
    elif channel_filter_mode == 'maxnorm_95':
      get_channel_subsets_maxnorm_pytorch(encoders, channel_filter_rate, data_loader, device,
                                          writer, cut_off_five_percent=True)
    elif channel_filter_mode == 'maxnorm_w':
      get_weight_subsets_maxnorm(encoders, channel_filter_rate, data_loader, device, writer,
                                 inverse_sort=False)
    elif channel_filter_mode == 'minnorm_w':
      get_weight_subsets_maxnorm(encoders, channel_filter_rate, data_loader, device, writer,
                                 inverse_sort=True)
    elif channel_filter_mode == 'maxvar_w':
      get_weight_subsets_maxvar(encoders, channel_filter_rate, data_loader, device,
                                writer, inverse_sort=False)
    elif channel_filter_mode == 'minvar_w':
      get_weight_subsets_maxvar(encoders, channel_filter_rate, data_loader, device,
                                writer, inverse_sort=True)
    elif channel_filter_mode == 'every_other_layer':
      get_channel_subsets_every_other_layer(encoders, channel_filter_rate)
    # elif channel_filter_mode == 'pca':
    #   get_pca_reduction(encoders, channel_filter_rate, data_loader, device, n_select_batches,
    #                     n_pca_iter)
    elif channel_filter_mode == 'maxvar':
      get_channel_subsets_maxvar(encoders, channel_filter_rate, data_loader, device, writer)
    elif channel_filter_mode == 'minvar':
      get_channel_subsets_maxvar(encoders, channel_filter_rate, data_loader, device, writer,
                                 inverse_sort=True)
    else:
      raise ValueError
  else:
    for layer_name, layer_act in encoders.layer_feats.items():
      encoders.channel_ids[layer_name] = None
      encoders.n_feats_by_layer[layer_name] = math.prod(layer_act.shape[1:])

  LOG.info(f"# features per layer: {encoders.n_feats_by_layer.items()}")

  LOG.info(f"# of features to be used (per class if labeled): {encoders.n_feats_total}")


def get_number_of_matching_features(net_enc, default_n_matching_layers, match_with_top_layers,
                                    device, writer, channel_filter_rate=1., data_loader=None,
                                    channel_filter_mode='maxnorm'):
  n_features_in_enc = 0
  n_matching_layers = 0
  # computes the total number of features
  channel_ids_by_enc = []
  for enc in net_enc:

    # number of features output for each layer, from the last to the first layer
    if data_loader is None or channel_filter_rate == 1.:
      num_features_for_each_enc_layer = enc.n_features_per_layer
      channel_ids_by_enc.append(None)
    else:
      grayscale_setting_saved = False
      if hasattr(enc, 'grayscale_input'):  # all encoders are pretrained on colored images
        grayscale_setting_saved = enc.grayscale_input
        enc.grayscale_input = False
      if channel_filter_mode == 'maxnorm':
        channel_ids, n_feats = get_channel_subsets_maxnorm(enc, channel_filter_rate, data_loader,
                                                           device, writer)
      elif channel_filter_mode == 'random':
        channel_ids, n_feats = get_channel_subsets_random(enc, channel_filter_rate, data_loader,
                                                          device)
      elif channel_filter_mode == 'hsic':
        raise NotImplementedError
      else:
        raise ValueError
      if hasattr(enc, 'grayscale_input'):
        enc.grayscale_input = grayscale_setting_saved
      num_features_for_each_enc_layer = n_feats
      channel_ids_by_enc.append(channel_ids)

    LOG.info(f"# numFeaturesForEachEncLayer (top to bottom): {num_features_for_each_enc_layer}")

    n_matching_layers = min(default_n_matching_layers, len(num_features_for_each_enc_layer))
    LOG.info("@ arg.match_with_top_layers: {}".format(match_with_top_layers))
    LOG.info("@ actual n_matching_layers: {}".format(n_matching_layers))

    if match_with_top_layers:
      n_features_in_enc += sum(num_features_for_each_enc_layer[:n_matching_layers])
    else:
      n_features_in_enc += sum(num_features_for_each_enc_layer[-n_matching_layers:])

  LOG.info("# of features to be used (per class if labeled): {}".format(n_features_in_enc))

  return n_features_in_enc, n_matching_layers, channel_ids_by_enc


def get_channel_subsets_maxnorm_pytorch(encoders, channel_filter_rate, data_loader, device,
                                        writer, n_batches=10, inverse_sort=False,
                                        cut_off_five_percent=False):
  layer_norms_dict = defaultdict(float)
  for idx, batch in enumerate(data_loader):
    x, y = batch
    encoders.load_features(x.to(device))

    for layer_name, layer_act in encoders.layer_feats.items():
      layer_feats_per_channel = layer_act.view(layer_act.shape[0], layer_act.shape[1], -1)
      layer_norms_sum = pt.sum(pt.norm(layer_feats_per_channel, dim=2), dim=0)

      layer_norms_dict[layer_name] += layer_norms_sum

    if idx >= n_batches:
      break
  descending = not inverse_sort
  for layer_name, norms in layer_norms_dict.items():
    if writer is not None:
      writer.add_histogram(f'real_data_embedding_channel_norms/{layer_name}', norms)
    _, idcs = pt.sort(norms, descending=descending)
    if cut_off_five_percent:  # drop the 5% samples with the highest values
      assert channel_filter_rate < 0.95
      n_to_cut = int(math.floor(len(idcs) * 0.05))
      idcs = idcs[n_to_cut:]
    selected_ids = idcs[:int(math.floor(len(idcs) * channel_filter_rate))]
    selected_ids, _ = pt.sort(selected_ids)
    encoders.channel_ids[layer_name] = selected_ids
    n_layer_feats = math.prod(encoders.layer_feats[layer_name][:, selected_ids, :, :].shape[1:])
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_weight_subsets_maxnorm(encoders, channel_filter_rate, data_loader, device,
                               writer, n_batches=10, inverse_sort=False):
  weight_norms_dict = defaultdict(float)
  for idx, batch in enumerate(data_loader):
    x, y = batch
    encoders.load_features(x.to(device))

    for layer_name, layer_act in encoders.layer_feats.items():
      layer_feats_per_channel = layer_act.view(layer_act.shape[0], layer_act.shape[1], -1)
      weight_sums_sqrd = pt.sum(layer_feats_per_channel**2, dim=0)
      weight_norms_dict[layer_name] += weight_sums_sqrd

    if idx >= n_batches:
      break
  descending = not inverse_sort
  for layer_name, sums_sqrd in weight_norms_dict.items():
    # n_channels, height, width = sums_sqrd.shape
    sums_sqrd = pt.flatten(sums_sqrd)
    if writer is not None:
      writer.add_histogram(f'real_data_embedding_weight_sums_sqrd/{layer_name}', sums_sqrd)
    _, idcs = pt.sort(sums_sqrd, descending=descending)
    selected_ids = idcs[:int(math.floor(len(idcs) * channel_filter_rate))]
    selected_ids, _ = pt.sort(selected_ids)
    encoders.weight_ids[layer_name] = selected_ids
    encoders.n_feats_by_layer[layer_name] = len(selected_ids)


def get_weight_subsets_maxvar(encoders, channel_filter_rate, data_loader, device,
                              writer, n_batches=10, inverse_sort=False):
  weight_sums_dict = defaultdict(float)
  weight_sqrd_dict = defaultdict(float)
  n_samples = 0
  for idx, batch in enumerate(data_loader):
    x, y = batch
    encoders.load_features(x.to(device))

    for layer_name, layer_act in encoders.layer_feats.items():
      layer_feats_per_channel = layer_act.view(layer_act.shape[0], layer_act.shape[1], -1)
      weight_sums_dict[layer_name] += pt.sum(layer_feats_per_channel, dim=0)
      weight_sqrd_dict[layer_name] += pt.sum(layer_feats_per_channel ** 2, dim=0)
      n_samples += x.shape[0]
    if idx >= n_batches:
      break
  descending = not inverse_sort
  for layer_name in weight_sums_dict:
    weight_sums = weight_sums_dict[layer_name]
    weight_sqrd = weight_sqrd_dict[layer_name]
    weight_var = (weight_sqrd - weight_sums) / n_samples
    weight_var = pt.flatten(weight_var)
    if writer is not None:
      writer.add_histogram(f'real_data_embedding_weight_variance/{layer_name}', weight_var)
    _, idcs = pt.sort(weight_var, descending=descending)
    selected_ids = idcs[:int(math.floor(len(idcs) * channel_filter_rate))]
    selected_ids, _ = pt.sort(selected_ids)
    encoders.weight_ids[layer_name] = selected_ids
    encoders.n_feats_by_layer[layer_name] = len(selected_ids)

# def get_weight_subsets_maxnorm(encoders: Encoders, channel_filter_rate, data_loader, device,
#                                n_batches, hsic_block_size, hsic_reps):
#   layer_names = get_layer_names(encoders, data_loader, device)
#   # layer_feats_dict, labels = collect_feats_and_labels(encoders, data_loader, device, n_batches)
#
#   for layer_name in layer_names:
#     l_feats, labels = collect_single_layer_feats_and_labels(encoders, data_loader, device,
#                                                             n_batches, layer_name)
#     encoders.flush_features()
#     weights_list = hsic_weight_selection(l_feats, labels, channel_filter_rate, hsic_block_size,
#                                          hsic_reps)
#     del l_feats
#     LOG.warning(f'weights_list: {weights_list}')
#     encoders.weight_ids[layer_name] = weights_list
#     n_layer_feats = len(weights_list)
#     encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_channel_subsets_maxvar(encoders, channel_filter_rate, data_loader, device,
                               writer, n_batches=10, inverse_sort=False):
  layer_norms_sqrd_dict = defaultdict(float)
  for idx, batch in enumerate(data_loader):
    x, y = batch
    encoders.load_features(x.to(device))

    for layer_name, layer_act in encoders.layer_feats.items():
      layer_feats_per_channel = layer_act.view(layer_act.shape[0], layer_act.shape[1], -1)
      layer_norms_sqrd_sum = pt.sum(pt.norm(layer_feats_per_channel, dim=2)**2, dim=0)

      layer_norms_sqrd_dict[layer_name] += layer_norms_sqrd_sum

    if idx >= n_batches:
      break
  descending = not inverse_sort
  for layer_name, norms_sqrd in layer_norms_sqrd_dict.items():
    if writer is not None:
      writer.add_histogram(f'real_data_embedding_channel_norms/{layer_name}', norms_sqrd)
    _, idcs = pt.sort(norms_sqrd, descending=descending)
    selected_ids = idcs[:int(math.floor(len(idcs) * channel_filter_rate))]
    selected_ids, _ = pt.sort(selected_ids)
    encoders.channel_ids[layer_name] = selected_ids
    n_layer_feats = math.prod(encoders.layer_feats[layer_name][:, selected_ids, :, :].shape[1:])
    encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_channel_subsets_random_pytorch(encoders, channel_filter_rate, data_loader, device):
  for idx, batch in enumerate(data_loader):
    x, y = batch
    for enc_name, enc in encoders.models.items():
      enc(x.to(device))
    break

  for layer_name, layer_act in encoders.layer_feats.items():
    # if len(layer_feats.shape) == 4:
    nc = layer_act.shape[1]
    selected_channels = pt.randperm(nc)[:int(math.floor(nc * channel_filter_rate))]
    selected_ids, _ = pt.sort(selected_channels)
    encoders.channel_ids[layer_name] = selected_ids
    # n_feats.append(math.prod(layer_act[:, selected_ids, :, :].shape[1:]))
    n_layer_feats = math.prod(layer_act[:, selected_ids, :, :].shape[1:])
    encoders.n_feats_by_layer[layer_name] = n_layer_feats
    # else:
    #   channel_ids_list.append(pt.arange(0, layer_feats.shape[1]))
    #   n_feats.append(math.prod(layer_feats.shape[1:]))


def get_channel_subsets_every_other_layer(encoders, channel_filter_rate):
  assert channel_filter_rate == 0.5
  for layer_idx, (layer_name, layer_act) in enumerate(encoders.layer_feats.items()):
    if layer_idx % 2 == 1:
      encoders.channel_ids[layer_name] = []
      encoders.n_feats_by_layer[layer_name] = 0
    else:
      encoders.channel_ids[layer_name] = list(range(layer_act.shape[1]))
      n_layer_feats = math.prod(encoders.layer_feats[layer_name].shape[1:])
      encoders.n_feats_by_layer[layer_name] = n_layer_feats


def get_channel_subsets_maxnorm(enc, channel_filter_rate, data_loader, device,
                                writer, n_batches=10):
  layer_norms_list = []
  layer_feats_list = None
  for idx, batch in enumerate(data_loader):
    x, y = batch
    _, layer_feats_list = enc(x.to(device))
    for jdx, layer_feats in enumerate(layer_feats_list):
      if len(layer_feats.shape) == 4:
        layer_feats_per_channel = layer_feats.view(layer_feats.shape[0], layer_feats.shape[1], -1)
        layer_norms_sum = pt.sum(pt.norm(layer_feats_per_channel, dim=2), dim=0)

        if idx == 0:
          layer_norms_list.append(layer_norms_sum)
        else:
          layer_norms_list[jdx] = layer_norms_list[jdx] + layer_norms_sum
      else:  # fully connnected layers
        if idx == 0:
          layer_norms_list.append(None)

    if idx >= n_batches:
      break

  channel_ids_list = []
  n_feats = []
  for layer_id, layer_norms, layer_feats in zip(range(len(layer_norms_list)), layer_norms_list,
                                                layer_feats_list):
    if len(layer_feats.shape) == 4:
      if writer is not None:
        writer.add_histogram(f'real_data_embedding_channel_norms/layer_{layer_id}', layer_norms)
      _, idcs = pt.sort(layer_norms, descending=True)
      selected_ids = idcs[:int(math.floor(len(idcs) * channel_filter_rate))]
      selected_ids, _ = pt.sort(selected_ids)
      channel_ids_list.append(selected_ids)
      n_feats.append(math.prod(layer_feats[:, selected_ids, :, :].shape[1:]))
    else:
      channel_ids_list.append(pt.arange(0, layer_feats.shape[1]))
      n_feats.append(math.prod(layer_feats.shape[1:]))
  return channel_ids_list, n_feats[::-1]


def get_channel_subsets_random(enc, channel_filter_rate, data_loader, device):
  layer_feats_list = None
  for x, y in data_loader:
    _, layer_feats_list = enc(x.to(device))
    break

  channel_ids_list = []
  n_feats = []
  for layer_id, layer_feats in enumerate(layer_feats_list):
    if len(layer_feats.shape) == 4:
      nc = layer_feats.shape[1]
      selected_channels = pt.randperm(nc)[:int(math.floor(nc * channel_filter_rate))]
      selected_ids, _ = pt.sort(selected_channels)
      channel_ids_list.append(selected_ids)
      n_feats.append(math.prod(layer_feats[:, selected_ids, :, :].shape[1:]))
    else:
      channel_ids_list.append(pt.arange(0, layer_feats.shape[1]))
      n_feats.append(math.prod(layer_feats.shape[1:]))
  return channel_ids_list, n_feats[::-1]
