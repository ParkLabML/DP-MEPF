import os.path
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math


# The ResNet models for CIFAR in https://arxiv.org/abs/1512.03385.


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


gn_groups = 4


class LambdaLayer(nn.Module):
  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x): return self.func(x)


# it's the same class as the original one but with group norm instead of batch norm
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.gn1 = nn.GroupNorm(gn_groups, planes, affine=False)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.gn2 = nn.GroupNorm(gn_groups, planes, affine=False)

    self.relu = nn.ReLU(inplace=True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      # self.shortcut = nn.Sequential(
      #   nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
      self.shortcut = LambdaLayer(lambda x:
                                  nnf.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4),
                                          "constant", 0))


  def forward(self, x):

    out = self.conv1(x)
    out = self.gn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.gn2(out)
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, num_classes=10):
    super(ResNet, self).__init__()

    self.num_layers = sum(layers)
    self.in_planes = 16  # inplanes
    # self.conv1 = conv3x3(3, 16)
    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.gn1 = nn.GroupNorm(gn_groups, 16, affine=False)
    self.relu = nn.ReLU(inplace=True)
    self.layer1 = self._make_layer(block, 16, layers[0])
    self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(64, num_classes)

    # standard initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.GroupNorm):
        try:
          m.weight.data.fill_(1)
          m.bias.data.zero_()
        except:
          pass

  def _make_layer(self, block, planes, num_blocks, stride=1):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.gn1(x)
    x = self.relu(x)

    x = self.layer1(x)
    x = self.layer2(x)

    x = self.layer3(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet20(classes):
  """Constructs a ResNet-20 model.
  """
  model = ResNet(BasicBlock, [3, 3, 3], classes)
  return model


def get_dp_resnet20(feats_dict, epsilon):
  base_model_dir = 'models/'
  pretrained_weights = {2: 'cifar_resnet_dp_1_epo:4_acc:25.93_eps2',
                        5: 'cifar_resnet_dp_1_epo:49_acc:49.51_eps5',
                        8: 'cifar_resnet_dp_1_epo:98_acc:54.03_eps8',
                        0: None}
  assert epsilon in pretrained_weights
  enc = resnet20(10)

  if epsilon != 0:
    model_weights_path = os.path.join(base_model_dir, pretrained_weights[epsilon])
    enc.load_state_dict(torch.load(model_weights_path)['model_state_dict'])

  blocks_per_layer = [3, 3, 3]
  conv_ids = [1, 2]
  layer_ids = [1, 2, 3]
  conv_layers = [enc.conv1]
  conv_names = ['c1']

  for layer_number_zero_index, layer_id in enumerate(layer_ids):
    for block_id in range(blocks_per_layer[layer_number_zero_index]):
      block = getattr(enc, f'layer{layer_id}')[block_id]
      for conv_id in conv_ids:
        conv_layers.append(getattr(block, f'conv{conv_id}'))
        conv_names.append(f'l{layer_id}_b{block_id}_c{conv_id}')

  for layer, layer_name in zip(conv_layers, conv_names):
    set_layer_hook('resnet18', layer, layer_name, feats_dict)
  return enc


def set_layer_hook(model_name, layer, layer_name, feats_dict):
  def hook(_model, _input, output):
    feats_dict[f'{model_name}_{layer_name}'] = output
  layer.register_forward_hook(hook)


if __name__ == '__main__':
  net = resnet20(10)
  print(net)
