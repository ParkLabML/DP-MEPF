"""
ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from collections import OrderedDict

import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf

from util import get_image_size_n_feats
from util_logging import LOG

CUSTOM_RESNET_WEIGHTS_PATH = 'models/resnet18.pt'

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
    self.relu1 = nn.ReLU(True)
    self.relu2 = nn.ReLU(True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes, track_running_stats=True)
      )

  def forward(self, x):
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.relu2(out)
    return out


class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10, image_size=32, get_perceptual_feats=False,
               grayscale_input=False):
    super(ResNet, self).__init__()
    self.in_planes = 64
    self.grayscale_input = grayscale_input

    # assert image_size % 32 == 0, f'image size {image_size} not supported'
    # n_max_pools = image_size // 32
    n_max_pools = 1

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512*block.expansion*n_max_pools*n_max_pools, num_classes)

    self._initialize_weights()
    self.get_perceptual_feats = get_perceptual_feats
    self.layer_feats = OrderedDict()

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    if self.grayscale_input:
      x = x.expand(-1, 3, -1, -1)
    feature_list = []
    x = nnf.relu(self.bn1(self.conv1(x)))
    # first layer was not hooked, therefore we have to add its result manually
    feature_list.append(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = nnf.avg_pool2d(x, 4)
    x = x.view(x.size(0), -1)
    out = self.linear(x)
    if self.get_perceptual_feats:
      for k, v in self.layer_feats.items():
        # LOG.warning(f'{k}: {v.shape}')  # first time, layer feats is empty, then adds all layers
        feature_list.append(v)
      feature_list.append(out)
      return out, feature_list
    else:
      return out

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _get_hook(self, layer_num, layer):
    def myhook(_module, _input, _out):
      self.layer_feats[layer_num] = _out
    layer.register_forward_hook(myhook)


def resnet_18(get_perceptual_feats=False, num_classes=10, image_size=32, grayscale_input=False):
  net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, image_size=image_size,
               get_perceptual_feats=get_perceptual_feats, grayscale_input=grayscale_input)
  if get_perceptual_feats:
    img_size_l, n_feats_l = get_image_size_n_feats(net, image_size=image_size)
    net.image_size_per_layer = np.array(img_size_l)
    net.n_features_per_layer = np.array(n_feats_l)

    # registers a hook for each RELU layer
    layer_num = 0
    for feature_layers in [net.layer1, net.layer2, net.layer3, net.layer4]:
      for res_block in feature_layers:
        for modl in res_block.modules():
          if str(modl)[0:4] == 'ReLU':
            LOG.debug("# registering hook module {} ".format(str(modl)))
            net._get_hook(layer_num, modl)
            layer_num += 1

    img_size_l, n_feats_l = get_image_size_n_feats(net, image_size=image_size)
    net.image_size_per_layer = np.array(img_size_l)
    net.n_features_per_layer = np.array(n_feats_l)
  return net
