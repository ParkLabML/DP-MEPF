from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
from util_logging import LOG


class ConvEncoderSkipConnectionBlock(nn.Module):
  def __init__(self, skip_connections_data, submodule, block_id, input_nc, output_nc, batch_norm,
               act_fun='', filter_size=4, stride=2, padding=1, dropout_rate=0, use_fc_layer=False):
    super(ConvEncoderSkipConnectionBlock, self).__init__()

    self._use_fc_layer = use_fc_layer
    self._skip_connections_data = skip_connections_data

    nn_layers = OrderedDict()

    if submodule is not None:
      nn_layers["submodel"] = submodule

    if self._use_fc_layer:
      nn_layers["fc_{block_id}"] = nn.Linear(input_nc, output_nc, bias=True)
    else:
      nn_layers[f"conv_{block_id}"] = nn.Conv2d(input_nc, output_nc, filter_size, stride, padding,
                                                bias=False)
    if batch_norm:
      nn_layers[f"btnConv_{block_id}"] = nn.BatchNorm2d(output_nc)
    if act_fun == 'tanh':
      nn_layers[f"reluConv_{block_id}"] = nn.Tanh()
    elif act_fun == 'relu':
      nn_layers[f"reluConv_{block_id}"] = nn.ReLU(True)
    elif act_fun == 'lrelu':
      nn_layers[f"lreluConv_{block_id}"] = nn.LeakyReLU(0.2, inplace=True)

    if dropout_rate > 0:
      nn_layers[f"dropout_{block_id}"] = nn.Dropout(dropout_rate)

    self.net = nn.Sequential(nn_layers)

  def forward(self, x):
    if self._use_fc_layer:
      output = self.net(x.view(x.size()[0], -1))
      output = output.view(output.size()[0], output.size()[1], 1, 1)
    else:
      output = self.net(x)
    self._skip_connections_data.append(output)
    return output


class ConvEncoderSkipConnections(nn.Module):

  def __init__(self, enc_size, ndf=64, n_channels=3,
               batch_norm=True, act_fun='relu', n_convs=4,
               dropout_rate=0, out_layer_act_fun='tanh', use_fc_for_last_layer=False,
               n_extra_layers=0):
    super(ConvEncoderSkipConnections, self).__init__()
    self.enc_size = enc_size
    self._skip_connections_data = []

    self.image_size_per_layer = [1, 4, 8, 16, 32, 64][:n_convs + 1]
    self.n_features_per_layer = None
    # image size is >= 64
    if n_convs > 3:
      self.n_features_per_layer = [self.enc_size, 8192, 16384, 32768, 65536, 131072][:n_convs + 1]
    # image size is 32
    else:
      self.n_features_per_layer = [self.enc_size, 4096, 8192, 16384]

    block_id = 0

    net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, None, block_id,
                                         n_channels, ndf, batch_norm, act_fun=act_fun,
                                         dropout_rate=dropout_rate)
    block_id += 1

    for i in range(n_extra_layers):
      net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, block_id,
                                           ndf, ndf, batch_norm, act_fun=act_fun,
                                           dropout_rate=dropout_rate,
                                           filter_size=3, stride=1, padding=1)
      block_id += 1

      # the number of features output in the extra layer is the same as the one in the first layer
      self.n_features_per_layer.append(self.n_features_per_layer[-1])
      self.image_size_per_layer.append(self.image_size_per_layer[-1])

    self.n_features_per_layer = np.asarray(self.n_features_per_layer)
    self.image_size_per_layer = np.asarray(self.image_size_per_layer)

    net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, block_id, ndf, ndf * 2,
                                         batch_norm, act_fun=act_fun, dropout_rate=dropout_rate)
    block_id += 1

    net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, block_id,
                                         ndf * 2, ndf * 4, batch_norm,
                                         act_fun=act_fun, dropout_rate=dropout_rate)
    block_id += 1

    if n_convs >= 4:

      net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, 3, ndf * 4, ndf * 8,
                                           batch_norm, act_fun=act_fun, dropout_rate=dropout_rate)
      block_id += 1
      n_last_layer_channels = ndf * 8
    else:
      n_last_layer_channels = ndf * 4

    for i in range(n_convs - 4):

      net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, block_id,
                                           ndf * 8, ndf * 8, batch_norm,
                                           act_fun=act_fun, dropout_rate=dropout_rate)
      block_id += 1

    # last conv output is the encoding size
    # nnLayers["conv_%d"%(len(nnLayers))]      = nn.Conv2d(ndf * 8, encSize, 4, 1, 0, bias=False)
    net = ConvEncoderSkipConnectionBlock(self._skip_connections_data, net, block_id,
                                         n_last_layer_channels, enc_size, batch_norm=False,
                                         act_fun=out_layer_act_fun, filter_size=4, stride=1,
                                         padding=0, use_fc_layer=use_fc_for_last_layer)

    self.net = net

  def forward(self, x):
    # removes data of skip connections from previous execution
    del self._skip_connections_data[:]
    output = self.net(x)
    return [output, self._skip_connections_data[:]]

  def get_n_features_per_layer(self):
    return self.n_features_per_layer

  def get_image_size_per_layer(self):
    return self.image_size_per_layer
