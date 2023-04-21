from collections import OrderedDict
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from util_logging import LOG


class DeconvDecoder(nn.Module):

  def __init__(self, enc_size, ngf, n_channels=3, batch_norm=True, use_relu=True,
               n_convs=4, n_extra_layers=0, first_deconv_kernel_size=4, gen_output='tanh'):
    super(DeconvDecoder, self).__init__()
    self.enc_size = enc_size
    self.gen_output = gen_output
    nn_layers = OrderedDict()

    num_out_channels_first_conv = ngf * 8
    if n_convs < 4:
      num_out_channels_first_conv = ngf * 4

    # first deconv goes from the encoding size
    nn_layers[f"deconv_{len(nn_layers)}"] = nn.ConvTranspose2d(enc_size,
                                                               num_out_channels_first_conv,
                                                               first_deconv_kernel_size,
                                                               1, 0, bias=False)
    if batch_norm:
      nn_layers[f"btn_{len(nn_layers)}"] = nn.BatchNorm2d(num_out_channels_first_conv)
    if use_relu:
      nn_layers[f"relu_{len(nn_layers)}"] = nn.ReLU(True)

    for i in range(n_convs - 4):
      self.create_deconv_block(nn_layers, ngf * 8, ngf * 8, batch_norm, use_relu)

    if n_convs >= 4:
      self.create_deconv_block(nn_layers, ngf * 8, ngf * 4, batch_norm, use_relu)

    self.create_deconv_block(nn_layers, ngf * 4, ngf * 2, batch_norm, use_relu)
    self.create_deconv_block(nn_layers, ngf * 2, ngf, batch_norm, use_relu)

    for i in range(n_extra_layers):
      self.create_conv_block(nn_layers, ngf, ngf, batch_norm, use_relu)

    self.create_deconv_block(nn_layers, ngf, n_channels, False, False)
    if self.gen_output == 'tanh':
      nn_layers[f"tanh_{len(nn_layers)}"] = nn.Tanh()

    self.net = nn.Sequential(nn_layers)

  @staticmethod
  def create_deconv_block(layers_dict, input_nc, output_nc, batch_norm=True, use_relu=True):
    layers_dict[f"deconv_{len(layers_dict)}"] = nn.ConvTranspose2d(input_nc, output_nc, 4, 2, 1,
                                                                   bias=False)
    if batch_norm:
      layers_dict[f"btnDeconv_{len(layers_dict)}"] = nn.BatchNorm2d(output_nc)
    if use_relu:
      layers_dict[f"reluDeconv_{len(layers_dict)}"] = nn.ReLU(True)

  @staticmethod
  def create_conv_block(layers_dict, input_nc, output_nc, batch_norm=True, use_relu=True):
    layers_dict[f"conv_{len(layers_dict)}"] = nn.Conv2d(input_nc, output_nc, 3, 1, 1, bias=False)
    if batch_norm:
      layers_dict[f"btnConv_{len(layers_dict)}"] = nn.BatchNorm2d(output_nc)
    if use_relu:
      layers_dict[f"reluConv_{len(layers_dict)}"] = nn.ReLU(True)

  def forward(self, x):
    # noinspection PyUnresolvedReferences
    # if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
    #   output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
    # else:
    # output =
    return self.net(x)


class ResnetG(nn.Module):
  def __init__(self, enc_size, nc, ndf, image_size=32, adapt_filter_size=False,
               use_conv_at_skip_conn=False, gen_output='tanh'):
    super(ResnetG, self).__init__()
    self.enc_size = enc_size
    self.ndf = ndf
    self.gen_output = gen_output

    if adapt_filter_size is True and use_conv_at_skip_conn is False:
      use_conv_at_skip_conn = True
      LOG.warning("WARNING: In ResnetG, setting use_conv_at_skip_conn to True because "
                  "adapt_filter_size is True.")

    n_upsample_blocks = int(log(image_size, 2)) - 2

    n_layers = n_upsample_blocks + 1
    filter_size_per_layer = [ndf] * n_layers
    if adapt_filter_size:
      for i in range(n_layers - 1, -1, -1):
        if i == n_layers - 1:
          filter_size_per_layer[i] = ndf
        else:
          filter_size_per_layer[i] = filter_size_per_layer[i+1]*2

    first_layer = nn.ConvTranspose2d(enc_size, filter_size_per_layer[0], 4, 1, 0, bias=False)
    nn.init.xavier_uniform_(first_layer.weight.data, 1.)
    last_layer = nn.Conv2d(filter_size_per_layer[-1], nc, 3, stride=1, padding=1)
    nn.init.xavier_uniform_(last_layer.weight.data, 1.)

    nn_layers = OrderedDict()
    # first deconv goes from the z size
    nn_layers["firstConv"] = first_layer

    layer_number = 1
    for i in range(n_upsample_blocks):
      nn_layers[f"resblock_{i}"] = ResidualBlockG(filter_size_per_layer[layer_number-1],
                                                  filter_size_per_layer[layer_number], stride=2,
                                                  use_conv_at_skip_conn=use_conv_at_skip_conn)
      layer_number += 1
    nn_layers["batchNorm"] = nn.BatchNorm2d(filter_size_per_layer[-1])
    nn_layers["relu"] = nn.ReLU()
    nn_layers["lastConv"] = last_layer
    if self.gen_output == 'tanh':
      nn_layers["tanh"] = nn.Tanh()

    self.net = nn.Sequential(nn_layers)

  def forward(self, x):
    return self.net(x)


class Upsample(nn.Module):
  def __init__(self, scale_factor=2, size=None):
    super(Upsample, self).__init__()
    self.size = size
    self.scale_factor = scale_factor

  def forward(self, x):
    x = nnf.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode='nearest')
    return x

  def get_input_noise(self, batch_size):
    return torch.randn(batch_size, self.enc_size, 1, 1)


class ResidualBlockG(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1, use_conv_at_skip_conn=False):
    super(ResidualBlockG, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

    if use_conv_at_skip_conn:
      self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
      nn.init.xavier_uniform_(self.conv_bypass.weight.data, 1.)

    nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
    nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

    self.model = nn.Sequential(
      nn.BatchNorm2d(in_channels),
      nn.ReLU(),
      Upsample(scale_factor=2),
      self.conv1,
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),
      self.conv2
    )
    self.bypass = nn.Sequential()
    if stride != 1:
      if use_conv_at_skip_conn:
        self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
      else:
        self.bypass = Upsample(scale_factor=2)

  def forward(self, x):
    return self.model(x) + self.bypass(x)
