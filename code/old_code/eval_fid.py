import os
import argparse
import numpy as np
import torch as pt
from torchvision import transforms
import torchvision.datasets as dset
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from data_loading import load_synth_dataset, load_dataset, load_imagenet_numpy, ImagenetNumpyDataset


def cifar10_stats(model, device, batch_size, workers, image_size=32, dataroot='../data'):
  transformations = [transforms.Resize(image_size), transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
  dataset = dset.CIFAR10(root=dataroot, download=True,
                         transform=transforms.Compose(transformations))
  assert dataset
  # noinspection PyUnresolvedReferences
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=int(workers))
  return stats_from_dataloader(dataloader, model, device)


def store_celeba_05_stats():
  log_dir = '../data'
  device = pt.device("cuda")
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir, 'celeba_32_normed05.npz')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)
  image_size = 32
  center_crop_size = 32

  transformations = [transforms.Resize(image_size),
                     transforms.CenterCrop(center_crop_size),
                     transforms.ToTensor()]
  # folder dataset
  dataset = dset.ImageFolder(root=os.path.join(log_dir, 'img_align_celeba'),
                             transform=transforms.Compose(transformations))
  assert dataset
  # noinspection PyUnresolvedReferences
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=50,
                                        shuffle=True, num_workers=1)

  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def store_celeba_0_1_stats():
  log_dir = '../data'
  device = pt.device("cuda")
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir, 'celeba_32_0_1.npz')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)
  image_size = 32
  center_crop_size = 32

  transformations = [transforms.Resize(image_size),
                     transforms.CenterCrop(center_crop_size),
                     transforms.ToTensor()]
  # folder dataset
  dataset = dset.ImageFolder(root=os.path.join(log_dir, 'img_align_celeba'),
                             transform=transforms.Compose(transformations))

  assert dataset
  # noinspection PyUnresolvedReferences
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=50,
                                        shuffle=True, num_workers=1)

  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def store_imagenet32_0_1_stats():
  log_dir = '../data'
  device = pt.device("cuda")
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir, 'imagenet32_0_1.npz')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)

  dataset = ImagenetNumpyDataset('../data', transform=None, test_set=False)
  # Create the dataloader
  dataloader = pt.utils.data.DataLoader(dataset, batch_size=50,
                                        shuffle=True, num_workers=1)
  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


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
  dataloader = load_imagenet_numpy('../data', batch_size=50, workers=1)
  mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
  np.savez(real_data_stats_file, mu=mu_real, sig=sig_real)


def store_imagenet64_stats():
  log_dir = '../data'
  device = pt.device("cuda")
  real_data_stats_dir = os.path.join(log_dir, 'fid_stats')
  os.makedirs(real_data_stats_dir, exist_ok=True)

  real_data_stats_file = os.path.join(real_data_stats_dir, 'imagenet64_64.npz')
  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
  model = InceptionV3([block_idx]).to(device)
  # mu_real, sig_real = cifar10_stats(model, device, batch_size, workers,
  #                                   image_size=32, dataroot='../data')
  dataloader = load_imagenet_numpy('../data', batch_size=300, workers=1, img_hw=64)
  mu_real, sig_real = stats_from_dataloader(dataloader, model, device, save_memory=True)
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
  if data_scale == 'bounded':
    bounded_str = '_bounded'
  elif data_scale == 'normed05':
    bounded_str = '_normed05'
  else:
    bounded_str = ''
  return f'{dataset_name}_{image_size}{bounded_str}.npz'


def get_fid_scores(synth_data_file, dataset_name, device, n_samples,
                   image_size, center_crop_size, data_scale,
                   base_data_dir='../data', batch_size=50):
  raise NotImplementedError('this FID score computation contains errors. Use the updated version instead.')
  real_data_stats_dir = os.path.join(base_data_dir, 'fid_stats')
  real_data_stats_file = os.path.join(real_data_stats_dir,
                                      embedding_file(dataset_name, image_size, data_scale))
  dims = 2048

  block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

  model = InceptionV3([block_idx]).to(device)

  if not os.path.exists(real_data_stats_file):
    store_data_stats(dataset_name, image_size, center_crop_size, base_data_dir, data_scale)

  stats = np.load(real_data_stats_file)
  mu_real, sig_real = stats['mu'], stats['sig']

  synth_data_loader = load_synth_dataset(synth_data_file, batch_size, n_samples)
  mu_syn, sig_syn = stats_from_dataloader(synth_data_loader, model, device)

  fid = calculate_frechet_distance(mu_real, sig_real, mu_syn, sig_syn)
  return fid


def stats_from_dataloader(dataloader, model, device='cpu', save_memory=False):
  """
  Returns:
  -- mu    : The mean over samples of the activations of the pool_3 layer of
             the inception model.
  -- sigma : The covariance matrix of the activations of the pool_3 layer of
             the inception model.
  """
  model.eval()

  pred_list = []

  if not save_memory:  # compute in single pass, store all embeddings
    for batch in tqdm(dataloader):
      x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
      x = x.to(device)

      with pt.no_grad():
        pred = model(x)[0]

      # If model output is not scalar, apply global spatial average pooling.
      # This happens if you choose a dimensionality not equal 2048.
      if pred.size(2) != 1 or pred.size(3) != 1:
        pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

      pred = pred.squeeze(3).squeeze(2).cpu().numpy()
      pred_list.append(pred)

    pred_arr = np.concatenate(pred_list, axis=0)
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma
  else:  # compute in two passes, no need to store all embeddings
    # first pass: calculate mean
    mu_acc = None
    n_samples = 0
    for batch in tqdm(dataloader):
      x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
      x = x.to(device)

      with pt.no_grad():
        pred = model(x)[0]
      if pred.size(2) != 1 or pred.size(3) != 1:
        pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

      n_samples += pred.shape[0]
      pred = pt.sum(pred.squeeze(3).squeeze(2), dim=0)
      mu_acc = mu_acc + pred if mu_acc is not None else pred

    mu = mu_acc / n_samples
    sigma_acc = None
    for batch in tqdm(dataloader):
      x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
      x = x.to(device)
      with pt.no_grad():
        pred = model(x)[0]
      if pred.size(2) != 1 or pred.size(3) != 1:
        pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

      pred = pred.squeeze(3).squeeze(2)
      pred_cent = pred - mu
      sigma_batch = pt.matmul(pt.t(pred_cent), pred_cent)
      sigma_acc = sigma_acc + sigma_batch if sigma_acc is not None else sigma_batch

    sigma = sigma_acc / (n_samples - 1)
    return mu.cpu().numpy(), sigma.cpu().numpy()
  # return pred_arr


def stats_from_dataloader_low_mem_test():
  # a short test to ensure the low memory version of stats_from_dataloader actually
  # gets the same results
  pred_list = []

  dataloader = [np.random.normal(size=(100, 500)) for _ in range(1000)]
  for pred in tqdm(dataloader):
    pred_list.append(pred)

  pred_arr = np.concatenate(pred_list, axis=0)
  mu1 = np.mean(pred_arr, axis=0)
  sigma1 = np.cov(pred_arr, rowvar=False)

  # first pass: calculate mean
  mu_acc = None
  n_samples = 0
  for pred in tqdm(dataloader):
    n_samples += pred.shape[0]
    pred = np.sum(pred, axis=0)
    mu_acc = mu_acc + pred if mu_acc is not None else pred

  mu2 = mu_acc / n_samples
  sigma_acc = None
  for pred in tqdm(dataloader):
    pred_cent = pred - mu2
    sigma_batch = np.matmul(pred_cent.T, pred_cent)
    sigma_acc = sigma_acc + sigma_batch if sigma_acc is not None else sigma_batch

  sigma2 = sigma_acc / (n_samples -1)
  print(np.linalg.norm(mu1 - mu2))
  print(np.linalg.norm(sigma1 - sigma2))


def standalone_fid_eval():
  parser = argparse.ArgumentParser()

  # PARAMS YOU LIKELY WANT TO SET
  parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'lsun', 'celeba',
                                                               'dmnist', 'fmnist'])
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--synth_data_file', type=str)
  parser.add_argument('--n_samples', type=int, default=5000)
  parser.add_argument('--image_size', type=int, default=32)
  parser.add_argument('--crop_size', type=int, default=32)
  parser.add_argument('--data_scale', default='normed', choices=['normed', 'bounded', 'normed05'])

  arg = parser.parse_args()
  get_fid_scores(arg.synth_data_file, arg.dataset, arg.device, arg.n_samples,
                 arg.image_size, arg.crop_size, arg.data_scale,
                 base_data_dir='../data', batch_size=50)

if __name__ == '__main__':
  # pass
  # stats_from_dataloader_low_mem_test()
  # store_data_stats('cifar10')
  # store_data_stats('cifar10', 32, 32, '../data', True)
  # store_imagenet32_stats()
  # store_celeba_05_stats()
  # store_celeba_0_1_stats()
  # store_imagenet32_0_1_stats()
  store_imagenet64_stats()
