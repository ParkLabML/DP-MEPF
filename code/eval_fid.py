import os
import numpy as np
import torch as pt
from torchvision import transforms
import torchvision.datasets as dset
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from data_loading import load_synth_dataset, load_dataset, load_imagenet_numpy, IMAGENET_MEAN, IMAGENET_SDEV


def cifar10_stats(model, device, batch_size, workers, image_size=32, dataroot='../data'):
  transformations = [transforms.Resize(image_size), transforms.ToTensor(),
                     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_SDEV)]
  dataset = dset.CIFAR10(root=dataroot, download=True,
                         transform=transforms.Compose(transformations))
  assert dataset
  # noinspection PyUnresolvedReferences
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=int(workers))
  return stats_from_dataloader(dataloader, model, device)


def store_imagenet32_stats():
  log_dir = '../data'
  device = pt.device("cuda")
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir, 'imagenet32.npz')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)
  # mu_real, sig_real = cifar10_stats(model, device, batch_size, workers,
  #                                   image_size=32, dataroot='../data')
  dataloader = load_imagenet_numpy('../data', batch_size=50, workers=1, img_hw=32)
  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def store_data_stats(dataset_name, image_size, center_crop_size, dataroot, data_scale):
  log_dir = '../data'
  device = pt.device("cuda")
  batch_size = 50
  workers = 1
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir,
                                      embedding_file(dataset_name, image_size, data_scale))
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)
  # mu_real, sig_real = cifar10_stats(model, device, batch_size, workers,
  #                                   image_size=32, dataroot='../data')
  dataloader, _ = load_dataset(dataset_name, image_size, center_crop_size, dataroot,
                               batch_size, workers, data_scale, labeled=False,
                               test_set=False)
  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def embedding_file(dataset_name, image_size, data_scale):
  scale_str = '' if data_scale is None else f'_{data_scale}'
  return f'{dataset_name}_{image_size}{scale_str}.npz'


def get_fid_scores_err(synth_data_file, dataset_name, device, n_samples,
                   image_size, center_crop_size, data_scale,
                   base_data_dir='../data', batch_size=50):
  real_data_stats_dir = os.path.join(base_data_dir, 'fid_stats')
  real_data_stats_file = os.path.join(real_data_stats_dir,
                                      embedding_file(dataset_name, image_size, data_scale))
  dims = 2048

  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

  model = InceptionV3([block_idx]).to(device)

  if not os.path.exists(real_data_stats_file):
    print(f'fid stats not found at {real_data_stats_file}. computing new stats')
    store_data_stats(dataset_name, image_size, center_crop_size, base_data_dir, data_scale)

  stats = np.load(real_data_stats_file)
  mu_real, sig_real = stats['mu'], stats['sig']

  print('computing synth data stats')
  synth_data_loader = load_synth_dataset(synth_data_file, batch_size, n_samples)
  mu_syn, sig_syn = stats_from_dataloader(synth_data_loader, model, device)

  fid = calculate_frechet_distance(mu_real, sig_real, mu_syn, sig_syn)
  return fid


def get_fid_scores_fixed(synth_data_file, dataset_name, device, n_samples,
                   image_size, center_crop_size, data_scale,
                   base_data_dir='../data', batch_size=50):
  real_data_stats_dir = os.path.join(base_data_dir, 'fid_stats')
  target_data_scale = '0_1'
  real_data_stats_file = os.path.join(real_data_stats_dir,
                                      embedding_file(dataset_name, image_size, target_data_scale))
  dims = 2048

  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

  model = InceptionV3([block_idx], normalize_input=True).to(device)

  if not os.path.exists(real_data_stats_file):
    store_data_stats(dataset_name, image_size, center_crop_size, base_data_dir, target_data_scale)

  stats = np.load(real_data_stats_file)
  mu_real, sig_real = stats['mu'], stats['sig']

  synth_data_loader = load_synth_dataset(synth_data_file, batch_size, n_samples,
                                         source_data_scale=data_scale,
                                         target_data_scale=target_data_scale)
  mu_syn, sig_syn = stats_from_dataloader(synth_data_loader, model, device)

  fid = calculate_frechet_distance(mu_real, sig_real, mu_syn, sig_syn)
  return fid


def stats_from_dataloader(dataloader, model, device='cpu'):
  """
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the inception model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the inception model.
  """
  model.eval()

  pred_list = []

  start_idx = 0
  n_prints = 2
  for batch in tqdm(dataloader):
    x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
    x = x.to(device)
    if n_prints > 0:
      print(f'stats_from_dataloader data scale. batch max={pt.max(x)}, min={pt.min(x)}')
      print(f'stats_from_dataloader data scale. batch q0.9={pt.quantile(x, q=0.9)}, q0.1={pt.quantile(x, q=0.1)}')
      n_prints -= 1
    with pt.no_grad():
      pred = model(x)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
      pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    # pred_arr[start_idx:start_idx + pred.shape[0]] = pred
    pred_list.append(pred)

    start_idx = start_idx + pred.shape[0]

  pred_arr = np.concatenate(pred_list, axis=0)
  # return pred_arr
  mu = np.mean(pred_arr, axis=0)
  sigma = np.cov(pred_arr, rowvar=False)
  return mu, sigma


if __name__ == '__main__':
  # store_data_stats('cifar10')
  # store_data_stats('cifar10', 32, 32, '../data', True)
  store_imagenet32_stats()
