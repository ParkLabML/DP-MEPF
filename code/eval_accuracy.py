"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.
First, from the same directory, run:
    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`
to generate the FFCV-formatted versions of CIFAR.
Then, simply run this to train models with default hyperparameters:
    `python train_cifar.py --config-file default_config.yaml`
You can override arguments as follows:
    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`
or by using a different config file.
"""
import numpy as np
from tqdm import tqdm

import torch as pt
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from models.downstream_models import get_ffcv_model
from data_loading import load_cifar10, load_synth_dataset


def train(model, loaders, device, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
  opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
  iters_per_epoch = len(loaders['train'])
  # Cyclic LR with single triangle
  lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                          [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                          [0, 1, 0])
  scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
  scaler = GradScaler()
  loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

  for _ in range(epochs):
    for ims, labs in tqdm(loaders['train']):
      ims, labs = ims.to(device), labs.to(device)
      opt.zero_grad(set_to_none=True)
      with autocast():
        out = model(ims)
        loss = loss_fn(out, labs)

      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      scheduler.step()


def evaluate(model, loaders, device, lr_tta=False):
  model.eval()
  accs = dict()
  with pt.no_grad():
    for name in ['train', 'test']:
      total_correct, total_num = 0., 0.
      for ims, labs in tqdm(loaders[name]):
        ims, labs = ims.to(device), labs.to(device)
        with autocast():
          out = model(ims)
          if lr_tta:
            out += model(pt.fliplr(ims))
          total_correct += out.argmax(1).eq(labs).sum().cpu().item()
          total_num += ims.shape[0]
          accs[name] = total_correct / total_num * 100

  return accs['test'], accs['train']


def synth_to_real_test(device, synth_data_file, data_scale):
  batch_size = 512
  epochs = 10  # 24
  lr = 0.5
  momentum = 0.9
  lr_peak_epoch = 5
  weight_decay = 5e-4
  label_smoothing = 0.1
  lr_tta = True
  synth_loader = load_synth_dataset(synth_data_file, batch_size, to_tensor=True)
  test_loader, _ = load_cifar10(32, '../data', batch_size, 2, data_scale, labeled=True, test_set=True)
  loaders = {'train': synth_loader, 'test': test_loader}
  # loaders = {'test': synth_loader, 'train': test_loader}
  model = get_ffcv_model(device)
  train(model, loaders, device, lr, epochs, label_smoothing, momentum, weight_decay, lr_peak_epoch)
  test_acc, train_acc = evaluate(model, loaders, device, lr_tta)

  return test_acc, train_acc


def main():
  # parser = ArgumentParser(description='Fast CIFAR-10 training')
  batch_size = 512
  epochs = 10  # 24
  lr = 0.5
  momentum = 0.9
  lr_peak_epoch = 5
  weight_decay = 5e-4
  label_smoothing = 0.1
  lr_tta = True
  device = 0 if pt.cuda.is_available() else 'cpu'
  # loaders, start_time = make_dataloaders(batch_size)
  tanh_out = False
  train_loader, _ = load_cifar10(32, '../data', batch_size, 2, tanh_out, labeled=True,
                                 test_set=False)
  test_loader, _ = load_cifar10(32, '../data', batch_size, 2, tanh_out, labeled=True, test_set=True)
  loaders = {'train': train_loader, 'test': test_loader}
  model = get_ffcv_model(device)
  train(model, loaders, device, lr, epochs, label_smoothing, momentum, weight_decay, lr_peak_epoch)
  test_acc, train_acc = evaluate(model, loaders, device, lr_tta)
  print(f'train accuracy: {train_acc:.1f}%')
  print(f'test accuracy: {test_acc:.1f}%')


if __name__ == "__main__":
  main()
  # synth_to_real_test(device=0, synth_data_file='../logs/may12_cifar_nondp_labeled_gridsearch/run_1/synth_data.npz')
