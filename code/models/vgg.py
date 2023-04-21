from collections import OrderedDict
import numpy as np
import torch.nn as nn
from util import get_image_size_n_feats
from util_logging import LOG

CUSTOM_VGG_WEIGHTS_PATH = 'models/vgg19.pt'
# logger = colorlog.getLogger(__name__)


def get_model_config(model_name):
  configs = {'VGG19': [64, 64, 'M',
                       128, 128, 'M',
                       256, 256, 256, 256, 'M',
                       512, 512, 512, 512, 'M',
                       512, 512, 512, 512, 'M'],
             }
  return configs[model_name]


class VGG(nn.Module):

  def __init__(self, vgg_name, get_perceptual_feats=False, num_classes=10, image_size=32,
               classifier_depth=3):
    super(VGG, self).__init__()
    vgg_modules = get_model_config(vgg_name)

    n_max_pools = 0
    for m in vgg_modules:
      if m == 'M':
        n_max_pools += 1

    self.features = self._make_layers(vgg_modules)

    if image_size % (2**n_max_pools) == 0:
      # 32 -> 512 * 1 * 1, 64 -> 512 * 2 * 2, ..., 256 -> 512 * 7 * 7
      out_hw = image_size // 2**n_max_pools  # height and width of encoder output
      LOG.info("_make_classifier: {},{},{},{}".format(out_hw, 512 * out_hw * out_hw, num_classes,
                                                      classifier_depth))
      self.classifier = self._make_classifier(512 * out_hw * out_hw, num_classes, classifier_depth)
    else:
      assert 0, f'image size {image_size} not supported'

    self._initialize_weights()
    self.name = vgg_name
    self.layer_feats = OrderedDict()
    self.get_perceptual_feats = get_perceptual_feats

  def forward(self, x):
    feats = self.features(x)
    feats = feats.view(feats.size(0), -1)
    out = self.classifier(feats)
    if self.get_perceptual_feats:
      feature_list = []
      for k, v in self.layer_feats.items():
        feature_list.append(v)
      feature_list.append(out)
      return out, feature_list
    else:
      return out

  @staticmethod
  def _make_classifier(input_size, num_classes, depth=3):
    assert 0 < depth <= 3, f'depth must be in [1,3] range not {depth}'

    if depth == 1:
      classify = nn.Sequential(
        nn.Linear(input_size, num_classes),
      )
    elif depth == 2:
      classify = nn.Sequential(
        nn.Linear(input_size, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
      )
    elif depth == 3:
      classify = nn.Sequential(
        nn.Linear(input_size, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
      )
    else:
      assert 0, 'classifier depth %d is not supported' % depth

    return classify

  @staticmethod
  def _make_layers(cfg):
    layers = []
    in_channels = 3
    for x in cfg:
      if x == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                   nn.BatchNorm2d(x, track_running_stats=True),
                   nn.ReLU(inplace=True)]
        in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # for torchvision version > v0.2.0
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # for torchvision version < v0.2.0
        # nn.init.kaiming_normal(m.weight, mode='fan_out')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

  def _get_hook(self, layer_num):
    def myhook(_module, _input, _out):
      self.layer_feats[layer_num] = _out
    self.features[layer_num].register_forward_hook(myhook)


def build_vgg_net(net_type='VGG19', get_perceptual_feats=False, num_classes=10, image_size=32,
                  classifier_depth=1):
  LOG.info("buildVGGNet: {} {} {} {} {}".format(net_type, get_perceptual_feats, num_classes,
                                                image_size, classifier_depth))
  net = VGG(net_type, get_perceptual_feats, num_classes, image_size=image_size,
            classifier_depth=classifier_depth)
  LOG.debug("# net : {} {}".format(len(net.features), net))
  if get_perceptual_feats:
    for idx in range(len(net.features)):
      if str(net.features[idx])[0:4] == 'ReLU':
        LOG.debug("# registering hook module ({}), {}".format(idx, str(net.features[idx])))
        net._get_hook(idx)
    img_size_l, num_feats_l = get_image_size_n_feats(net, image_size=image_size)
    net.image_size_per_layer = np.array(img_size_l)
    net.n_features_per_layer = np.array(num_feats_l)
  return net


def vgg_19(get_perceptual_feats=False, num_classes=10, image_size=32, classifier_depth=1):
  LOG.info("VGG19: {},{},{},{}".format(get_perceptual_feats,num_classes,image_size,
                                          classifier_depth))
  return build_vgg_net('VGG19', get_perceptual_feats, num_classes, image_size=image_size,
                       classifier_depth=classifier_depth)
