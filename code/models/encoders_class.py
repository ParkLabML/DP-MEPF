from collections import OrderedDict


class Encoders:
  def __init__(self, models: dict, layer_acts: dict, n_split_layers: int, n_classes: int = None):
    self.models = models
    self.layer_feats = layer_acts
    self.n_split_layers = n_split_layers  # enables labeled training with partial shared embedding
    self.n_classes = n_classes
    self.channel_ids = OrderedDict()
    self.weight_ids = OrderedDict()
    self.pca_maps = OrderedDict()
    self.n_feats_by_layer = OrderedDict()
    self._n_feats_total = None
    self._n_split_features = None
    self._prune_mode = 'not_init'

  def load_features(self, data_batch):
    for enc in self.models.values():
      enc(data_batch)

  def flush_features(self):  # might be useful for reducing memory in some places
    for key in self.layer_feats:
      self.layer_feats[key] = None

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
  def do_pruning(self):
    return (len(self.pca_maps) + len(self.channel_ids) + len(self.weight_ids)) > 0

  @property
  def prune_mode(self):
    if self._prune_mode == 'not_init':
      if len(self.channel_ids) > 0:
        assert (len(self.weight_ids) == 0) and (len(self.pca_maps) == 0)
        self._prune_mode = 'channels'
      elif len(self.weight_ids) > 0:
        assert len(self.pca_maps) == 0
        self._prune_mode = 'weights'
      elif len(self.pca_maps) > 0:
        self._prune_mode = 'pca'
      else:
        self._prune_mode = None

    return self._prune_mode

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
