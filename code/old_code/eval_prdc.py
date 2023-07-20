import numpy as np
from prdc.prdc import compute_prdc, compute_nearest_neighbour_distances, compute_pairwise_distance
from torchvision.models import vgg16
import torch as pt
import torch.nn as nn
from data_loading import load_synth_dataset, load_dataset


def compute_dc(real_features, fake_features, nearest_k):
  """
  Computes only density, and coverage given two manifolds.
  Args:
      real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
      fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
      nearest_k: int.
  Returns:
      dict of precision, recall, density, and coverage.
  """

  print('Num real: {} Num fake: {}'
        .format(real_features.shape[0], fake_features.shape[0]))

  real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
    real_features, nearest_k)
  distance_real_fake = compute_pairwise_distance(
    real_features, fake_features)

  density = (1. / float(nearest_k)) * (
      distance_real_fake <
      np.expand_dims(real_nearest_neighbour_distances, axis=1)).sum(axis=0).mean()

  coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

  return dict(density=density, coverage=coverage)


class PRDCEval:

  def __init__(self, pretrained=False, reduced_embedding=False, device='cpu'):
    assert not (pretrained and reduced_embedding)
    self.device = device
    self.vgg = vgg16(pretrained=False)
    self.vgg.to(device)
    self.vgg.eval()
    self.reduced = reduced_embedding
    self.reduction = None if not self.reduced else nn.Linear(4096, 64, device=device)
    self.embeddings_list = []

    if self.reduced:
      def hook(_model, _input, output):
        self.embeddings_list.append(self.reduction(output).detach().cpu())

      self.vgg.classifier[2].register_forward_hook(hook)
    else:
      def hook(_model, _input, output):
        self.embeddings_list.append(output.detach().cpu())

      self.vgg.classifier[3].register_forward_hook(hook)

  def collect_batch_embeddings(self, data_loader, max_n_samples=None):
    self.embeddings_list.clear()
    n_samples = 0
    for batch in data_loader:
      x = batch[0] if isinstance(batch, (tuple, list)) else batch
      x = x.to(self.device)
      if max_n_samples is not None and (n_samples + x.shape[0] > max_n_samples):
        diff = max_n_samples - n_samples
        x = x[:diff]
        self.vgg(x)
        break
      else:
        n_samples += x.shape[0]
        self.vgg(x)

    embeddings = pt.cat(self.embeddings_list)
    self.embeddings_list.clear()
    return embeddings

  def eval(self, real_data_loader, fake_data_loader, nearest_k, max_n_samples, skip_pr=False):
    real_emb = self.collect_batch_embeddings(real_data_loader, max_n_samples)
    fake_emb = self.collect_batch_embeddings(fake_data_loader, max_n_samples)
    if skip_pr:
      return compute_dc(real_emb, fake_emb, nearest_k)
    else:
      return compute_prdc(real_emb, fake_emb, nearest_k)


def get_prdc(synth_data_file, batch_size, n_samples, dataset_name, image_size,
             center_crop_size, data_scale, dataroot='../data',
             nearest_k=5, max_n_samples=10_000, skip_pr=False, pretrained=False,
             reduced_embedding=False, device='cpu'):
  fake_data_loader = load_synth_dataset(synth_data_file, batch_size, n_samples)
  real_data_loader, _ = load_dataset(dataset_name, image_size, center_crop_size, dataroot,
                                     batch_size, n_workers=1, data_scale=data_scale, labeled=False,
                                     test_set=False)
  prdc_evaluator = PRDCEval(pretrained, reduced_embedding, device)
  return prdc_evaluator.eval(real_data_loader, fake_data_loader, nearest_k, max_n_samples, skip_pr)


if __name__ == '__main__':
  # pass
  def get_randn_batches(bs, n):
    return [(pt.randn(bs, 3, 32, 32), None) for _ in range(n)]

  real_loader = get_randn_batches(10, 2)
  fake_loader = get_randn_batches(10, 2)
  prdc_evaluator = PRDCEval(pretrained=False, reduced_embedding=True, device=0)
  res = prdc_evaluator.eval(real_loader, fake_loader, nearest_k=5, skip_pr=False)

  print(res)
