import torch as pt
from collections import OrderedDict
from data_loading import IMAGENET_MEAN, IMAGENET_SDEV, CIFAR10_MEAN, CIFAR10_SDEV, CELEBA32_MEAN, CELEBA32_SDEV, \
  CELEBA64_MEAN, CELEBA64_SDEV


class Encoders:
  def __init__(self, models: dict, layer_acts: dict, n_split_layers: int, device, n_classes: int = None,
               input_scalings: dict = None):
    self.models = models
    self.layer_feats = layer_acts
    self.n_split_layers = n_split_layers  # enables labeled training with partial shared embedding
    self.n_classes = n_classes
    self.n_feats_by_layer = OrderedDict()
    self._n_feats_total = None
    self._n_split_features = None
    self.device = device
    self.input_scaling_names = input_scalings
    self.input_scaling_tensors = None
    self.initialize_input_scalings()


  def load_features(self, data_batch):
    for enc in self.models.values():
      enc(data_batch)

  def flush_features(self):  # might be useful for reducing memory in some places
    for key in self.layer_feats:
      self.layer_feats[key] = None

  def initialize_input_scalings(self):
    if self.input_scaling_names is None:
      return
    scaling_names = {'0_1_to_IMGNet_Norm': (IMAGENET_MEAN, IMAGENET_SDEV),
                     '0_1_to_Cifar10_Norm': (CIFAR10_MEAN, CIFAR10_SDEV),
                     '0_1_to_Celeba32_Norm': (CELEBA32_MEAN, CELEBA32_SDEV),
                     '0_1_to_Celeba64_Norm': (CELEBA64_MEAN, CELEBA64_SDEV)}
    scale_by_model_dict = dict()

    for model_name in self.models:
      mean, sdev = scaling_names[self.input_scaling_names[model_name]]
      mean_tsr = pt.tensor(mean, device=self.device)[None, :, None, None]
      sdev_tsr = pt.tensor(sdev, device=self.device)[None, :, None, None]
      scale_by_model_dict[model_name] = mean_tsr, sdev_tsr
    self.input_scaling_tensors = scale_by_model_dict

  def rescale_batch_input(self, x_in, model_name):
    if self.input_scaling_tensors is None:
      return x_in
    else:
      mean_tsr, sdev_tsr = self.input_scaling_tensors[model_name]
      return (x_in - mean_tsr) / sdev_tsr

  @property
  def n_feats_total(self):
    if self._n_feats_total is not None:
      return self._n_feats_total
    elif len(self.n_feats_by_layer) == 0:
      return None
    else:
      if self.n_split_layers is not None:
        n_feats_shared = sum(list(self.n_feats_by_layer.values())[:-self.n_split_layers])
        n_feats_split = sum(list(self.n_feats_by_layer.values())[-self.n_split_layers:])
        self._n_feats_total = n_feats_shared + self.n_classes * n_feats_split
      else:
        self._n_feats_total = sum(self.n_feats_by_layer.values())
      return self._n_feats_total

  @property
  def n_split_features(self):
    assert self.n_split_layers is not None
    if self._n_split_features is None:
      self._n_split_features = 0
      # LOG.info(f'counting features of {self.n_split_layers} last layers to use per class')
      for idx, (layer, n_feats) in enumerate(reversed(self.n_feats_by_layer.items())):
        if idx == self.n_split_layers:
          break
        else:
          self._n_split_features += n_feats

    return self._n_split_features
