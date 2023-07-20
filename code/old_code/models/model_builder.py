from __future__ import print_function
import math
from collections import OrderedDict
import torch as pt
from torch import nn as nn
import torch.nn.functional as nnf
from models.encoders_class import Encoders
from models.generators import DeconvDecoder, ResnetG
from models.ae_encoder import ConvEncoderSkipConnections
from models.mnist_gen_model import ConvCondGen
from models.resnet20 import get_dp_resnet20
from models.resnet import resnet_18, CUSTOM_RESNET_WEIGHTS_PATH
from models.vgg import vgg_19, CUSTOM_VGG_WEIGHTS_PATH
from torch.nn.parallel import DistributedDataParallel as DDP
from util_logging import LOG
import torchvision.models as pt_models


def get_pt_resnet_with_conv_layers(feats_dict, n_layers):
  assert n_layers in {18, 34, 50, 101, 152}
  if n_layers == 18:
    enc = pt_models.resnet18(pretrained=True)  # 47k feats
    blocks_per_layer = [2, 2, 2, 2]
    conv_ids = [1, 2]
  elif n_layers == 34:
    enc = pt_models.resnet34(pretrained=True)  # 72k feats
    blocks_per_layer = [3, 4, 6, 3]
    conv_ids = [1, 2]
  elif n_layers == 50:
    enc = pt_models.resnet50(pretrained=True)  # 196k feats
    blocks_per_layer = [3, 4, 6, 3]
    conv_ids = [1, 2, 3]
  elif n_layers == 101:
    enc = pt_models.resnet101(pretrained=True)  # 300k feats
    blocks_per_layer = [3, 4, 23, 3]
    conv_ids = [1, 2, 3]
  elif n_layers == 152:
    enc = pt_models.resnet152(pretrained=True)  # 430k feats
    blocks_per_layer = [3, 8, 36, 3]
    conv_ids = [1, 2, 3]
  else:
    raise ValueError
  # resnet 34 blocks per layer
  # l1 3b l2 4b l3 6b l4 3b

  layer_ids = [1, 2, 3, 4]
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


def get_pt_vgg_with_conv_layers(feats_dict, n_layers, after_relu=False):
  assert n_layers in {11, 13, 16, 19}
  if n_layers == 11:
    enc = pt_models.vgg11(pretrained=True)
    conv_layer_ids = [0, 3, 6, 8, 11, 13, 16, 18]  # 151k feats
  elif n_layers == 13:
    enc = pt_models.vgg13(pretrained=True)
    conv_layer_ids = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]  # 250k feats
  elif n_layers == 16:
    enc = pt_models.vgg16(pretrained=True)
    conv_layer_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]  # 276k feats
  elif n_layers == 19:
    enc = pt_models.vgg19(pretrained=True)
    conv_layer_ids = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]  # 303k feats
  else:
    raise ValueError

  if after_relu:
    conv_layer_ids = [k + 1 for k in conv_layer_ids]  # 303k feats

  for count, conv_layer_id in enumerate(conv_layer_ids):
    set_layer_hook('vgg19', enc.features[conv_layer_id], f'conv{count + 1}', feats_dict)
  return enc


def get_pt_convnext(feats_dict, convnext_size, only_conv_layers=True):
  assert convnext_size in {'tiny', 'small', 'base', 'large'}

  n_blocks = 27
  if convnext_size == 'tiny':
    enc = pt_models.convnext_tiny()  # 53k feats (onlyconv)
    n_blocks = 9
  elif convnext_size == 'small':
    enc = pt_models.convnext_small()  # 80k feats (onlyconv)
  elif convnext_size == 'base':
    enc = pt_models.convnext_base()  # 108k feats (onlyconv)
  elif convnext_size == 'large':
    enc = pt_models.convnext_large()  # 161k feats (onlyconv)
  else:
    raise ValueError

  conv_layers = []
  conv_names = []

  cnb_block_ids = [(1, k) for k in range(3)] + [(3, k) for k in range(3)] \
                  + [(5, k) for k in range(n_blocks)]
  cnb_block_out_ids = [0] if only_conv_layers else [0, 3, 5]
  sequential_conv_ids = [(0, 0), (2, 1), (4, 1), (6, 1)]

  for idx, jdx in cnb_block_ids:
    for kdx in cnb_block_out_ids:
      conv_layers.append(enc.features[idx][jdx].block[kdx])
      conv_names.append(f'block{idx}_{jdx}_layer{kdx}')
  for idx, jdx in sequential_conv_ids:
    conv_layers.append(enc.features[idx][jdx])
    conv_names.append(f'block{idx}_layer{jdx}')

  for layer, layer_name in zip(conv_layers, conv_names):
    set_layer_hook(f'convnext_{convnext_size}', layer, layer_name, feats_dict)
  return enc


def get_pt_efficientnet_v2(feats_dict, net_size):
  assert net_size in {'s', 'm', 'l'}

  if net_size == 's':
    fused_blocks = [2, 4, 4]
    mbconv_blocks = [6, 9, 15]
    enc = pt_models.efficientnet_v2_s()  # 47.5k feats
  elif net_size == 'm':
    fused_blocks = [3, 5, 5]
    mbconv_blocks = [7, 14, 18, 5]
    enc = pt_models.efficientnet_v2_m()  # k 68.7k feats
  elif net_size == 'l':
    fused_blocks = [4, 7, 7]
    mbconv_blocks = [10, 19, 25, 7]
    enc = pt_models.efficientnet_v2_l()  # 119k feats
  else:
    raise ValueError

  conv_layers = [enc.features[0][0]]
  conv_names = ['layer0_conv']

  for layer_idx, n_f_blocks in enumerate(fused_blocks):
    corrected_layer_idx = layer_idx + 1
    for block_idx in range(n_f_blocks):
      conv_idx = 0 if layer_idx == 0 else 1
      print(f'layer{corrected_layer_idx}_fused_block{block_idx}_conv')
      conv_layers.append(enc.features[corrected_layer_idx][block_idx].block[conv_idx][0])
      conv_names.append(f'layer{corrected_layer_idx}_fused_block{block_idx}_conv')
  for layer_idx, n_mb_blocks in enumerate(mbconv_blocks):
    corrected_layer_idx = layer_idx + len(fused_blocks) + 1
    for block_idx in range(n_mb_blocks):
      print(f'layer{corrected_layer_idx}_mb_block{block_idx}_conv')
      conv_layers.append(enc.features[corrected_layer_idx][block_idx].block[3][0])
      conv_names.append(f'layer{corrected_layer_idx}_mb_block{block_idx}_conv')

  for layer, layer_name in zip(conv_layers, conv_names):
    set_layer_hook(f'efficient_net_v2_{net_size}', layer, layer_name, feats_dict)
  return enc


class ScaleInception(nn.Module):
  def __init__(self, feats_dict):
    super(ScaleInception, self).__init__()
    self.enc = pt_models.inception_v3(pretrained=True)
    set_layer_hook('fid_feats', self.enc.avgpool, 'fid_layer', feats_dict)

  def forward(self, x):
    x = nnf.interpolate(x,
                        size=(299, 299),
                        mode='bilinear',
                        align_corners=False)
    x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)
    return self.enc(x)


def get_pt_fid_features(feats_dict):
  return ScaleInception(feats_dict)


def get_pt_efficientnet_b(feats_dict, net_size):
  layers = [1, 2, 2, 3, 3, 4, 1]
  if net_size == 0:
    enc = pt_models.efficientnet_b0()
  elif net_size == 1:
    enc = pt_models.efficientnet_b1()
  elif net_size == 2:
    enc = pt_models.efficientnet_b2()
  elif net_size == 3:
    enc = pt_models.efficientnet_b3()
  elif net_size == 4:
    enc = pt_models.efficientnet_b4()
  elif net_size == 5:
    enc = pt_models.efficientnet_b5()
  elif net_size == 6:
    enc = pt_models.efficientnet_b6()
  elif net_size == 7:
    enc = pt_models.efficientnet_b7()
  else:
    raise ValueError

  conv_layers = [enc.features[0][0]]
  conv_names = ['layer0_conv']

  for layer_idx, n_f_blocks in enumerate(layers):
    corrected_layer_idx = layer_idx + 1
    for block_idx in range(n_f_blocks):
      conv_idx = 0 if layer_idx == 0 else 1
      print(f'layer{corrected_layer_idx}_fused_block{block_idx}_conv')
      conv_layers.append(enc.features[corrected_layer_idx][block_idx].block[conv_idx][0])
      conv_names.append(f'layer{corrected_layer_idx}_fused_block{block_idx}_conv')

  conv_layers.append(enc.features[8][0])
  conv_names.append('layer8_conv')

  for layer, layer_name in zip(conv_layers, conv_names):
    set_layer_hook(f'efficient_net_b{net_size}', layer, layer_name, feats_dict)
  return enc


def get_custom_vgg19_with_conv_layers(feats_dict, image_size, n_classes_in_enc):
  mc = model_constants(image_size)
  enc = vgg_19(get_perceptual_feats=False, num_classes=n_classes_in_enc,
               image_size=image_size, classifier_depth=mc['vgg_classifier_depth'])
  state = pt.load(CUSTOM_VGG_WEIGHTS_PATH)
  net = state['net']

  tensors_to_add = dict()
  for key in enc.state_dict().keys():
    if key not in net.state_dict():
      assert key.endswith('num_batches_tracked')
      new_t = enc.state_dict()[key]
      tensors_to_add[key] = 978 * pt.ones_like(new_t)  # dummy entry

  enc.load_state_dict({**net.state_dict(), **tensors_to_add})
  conv_layer_ids = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
  for count, conv_layer_id in enumerate(conv_layer_ids):
    set_layer_hook('vgg19', enc.features[conv_layer_id], f'conv{count + 1}', feats_dict)
  return enc


def get_custom_resnet18_with_conv_layers(feats_dict, image_size, n_classes_in_enc):
  enc = resnet_18(get_perceptual_feats=False, num_classes=n_classes_in_enc, image_size=image_size)
  state = pt.load(CUSTOM_RESNET_WEIGHTS_PATH)
  net = state['net']

  # add "num_batches_tracked" keys to loaded statedict. these shouldn't matter in eval mode
  tensors_to_add = dict()
  for key in enc.state_dict().keys():
    if key not in net.state_dict():
      assert key.endswith('num_batches_tracked')
      new_t = enc.state_dict()[key]
      tensors_to_add[key] = 987 * pt.ones_like(new_t)  # dummy entry

  enc.load_state_dict({**net.state_dict(), **tensors_to_add})
  conv_layers = [enc.conv1,
                 enc.layer1[0].conv1, enc.layer1[0].conv2, enc.layer1[1].conv1, enc.layer1[1].conv2,
                 enc.layer2[0].conv1, enc.layer2[0].conv2, enc.layer2[1].conv1, enc.layer2[1].conv2,
                 enc.layer3[0].conv1, enc.layer3[0].conv2, enc.layer3[1].conv1, enc.layer3[1].conv2,
                 enc.layer4[0].conv1, enc.layer4[0].conv2, enc.layer4[1].conv1, enc.layer4[1].conv2]

  conv_names = ['c1',
                'l1b1c1', 'l1b1c2', 'l1b2c1', 'l1b2c2', 'l2b1c1', 'l2b1c2', 'l2b2c1', 'l2b2c2',
                'l3b1c1', 'l3b1c2', 'l3b2c1', 'l3b2c2', 'l4b1c1', 'l4b1c2', 'l4b2c1', 'l4b2c2']
  for layer, layer_name in zip(conv_layers, conv_names):
    set_layer_hook('resnet18', layer, layer_name, feats_dict)
  return enc


def set_layer_hook(model_name, layer, layer_name, feats_dict):
  def hook(_model, _input, output):
    feats_dict[f'{model_name}_{layer_name}'] = output
  layer.register_forward_hook(hook)


def get_torchvision_encoders(encoder_names, image_size, device, pretrain_dataset, n_classes_in_enc,
                             n_split_layers, n_classes):
  if pretrain_dataset in {'svhn', 'cifar10_pretrain'}:
    assert len(encoder_names) == 1
    return small_data_model(pretrain_dataset, encoder_names[0], device, image_size)
  else:
    assert pretrain_dataset == 'imagenet'
  feats_dict = OrderedDict()
  models_dict = dict()

  # def set_layer_hook(model_name, layer, layer_name, feats_dict):
  #   def hook(_model, _input, output):
  #     feats_dict[f'{model_name}_{layer_name}'] = output
  #   layer.register_forward_hook(hook)

  for encoder_name in encoder_names:
    if encoder_name == 'resnet18':
      enc = get_pt_resnet_with_conv_layers(feats_dict, n_layers=18)
    elif encoder_name == 'vgg19':
      enc = get_pt_vgg_with_conv_layers(feats_dict, n_layers=19)
    elif encoder_name == 'vgg19_custom':
      enc = get_custom_vgg19_with_conv_layers(feats_dict, image_size, n_classes_in_enc)
    elif encoder_name == 'resnet18_custom':
      enc = get_custom_resnet18_with_conv_layers(feats_dict, image_size, n_classes_in_enc)
    elif encoder_name == 'vgg16':
      enc = get_pt_vgg_with_conv_layers(feats_dict, n_layers=16)
    elif encoder_name == 'vgg13':
      enc = get_pt_vgg_with_conv_layers(feats_dict, n_layers=13)
    elif encoder_name == 'vgg11':
      enc = get_pt_vgg_with_conv_layers(feats_dict, n_layers=11)
    elif encoder_name == 'vgg19relu':
      enc = get_pt_vgg_with_conv_layers(feats_dict, n_layers=19, after_relu=True)
    elif encoder_name == 'resnet34':
      enc = get_pt_resnet_with_conv_layers(feats_dict, n_layers=34)
    elif encoder_name == 'resnet50':
      enc = get_pt_resnet_with_conv_layers(feats_dict, n_layers=50)
    elif encoder_name == 'resnet101':
      enc = get_pt_resnet_with_conv_layers(feats_dict, n_layers=101)
    elif encoder_name == 'resnet152':
      enc = get_pt_resnet_with_conv_layers(feats_dict, n_layers=152)
    elif encoder_name == 'convnext_tiny':
      enc = get_pt_convnext(feats_dict, 'tiny')
    elif encoder_name == 'convnext_small':
      enc = get_pt_convnext(feats_dict, 'small')
    elif encoder_name == 'convnext_base':
      enc = get_pt_convnext(feats_dict, 'base')
    elif encoder_name == 'convnext_large':
      enc = get_pt_convnext(feats_dict, 'large')
    elif encoder_name == 'efficientnet_s':
      enc = get_pt_efficientnet_v2(feats_dict, 's')
    elif encoder_name == 'efficientnet_m':
      enc = get_pt_efficientnet_v2(feats_dict, 'm')
    elif encoder_name == 'efficientnet_l':
      enc = get_pt_efficientnet_v2(feats_dict, 'l')
    elif encoder_name == 'efficientnet_b0':
      enc = get_pt_efficientnet_b(feats_dict, 0)
    elif encoder_name == 'efficientnet_b1':
      enc = get_pt_efficientnet_b(feats_dict, 1)
    elif encoder_name == 'efficientnet_b2':
      enc = get_pt_efficientnet_b(feats_dict, 2)
    elif encoder_name == 'efficientnet_b3':
      enc = get_pt_efficientnet_b(feats_dict, 3)
    elif encoder_name == 'efficientnet_b4':
      enc = get_pt_efficientnet_b(feats_dict, 4)
    elif encoder_name == 'efficientnet_b5':
      enc = get_pt_efficientnet_b(feats_dict, 5)
    elif encoder_name == 'efficientnet_b6':
      enc = get_pt_efficientnet_b(feats_dict, 6)
    elif encoder_name == 'efficientnet_b7':
      enc = get_pt_efficientnet_b(feats_dict, 7)
    elif encoder_name == 'resnet20_eps0':
      enc = get_dp_resnet20(feats_dict, 0)
    elif encoder_name == 'resnet20_eps2':
      enc = get_dp_resnet20(feats_dict, 2)
    elif encoder_name == 'resnet20_eps5':
      enc = get_dp_resnet20(feats_dict, 5)
    elif encoder_name == 'resnet20_eps8':
      enc = get_dp_resnet20(feats_dict, 8)
    elif encoder_name == 'fid_features':
      enc = get_pt_fid_features(feats_dict)
    else:
      raise ValueError(f'name {encoder_name} not found')

    models_dict[encoder_name] = enc
    enc.to(device)
    enc.eval()
    LOG.info(f'# Encoder :{encoder_name}')
    for param in enc.parameters():
      param.requires_grad = False
  return Encoders(models_dict, feats_dict, n_split_layers, n_classes)


def get_encoders(net_enc_types, net_enc_files, image_size, z_dim, n_classes_in_enc, device,
                 pretrain_dataset):
  if pretrain_dataset in {'svhn', 'cifar10_pretrain'}:
    assert len(net_enc_types) == 1
    return small_data_model(pretrain_dataset, net_enc_types[0], device, image_size)
  else:
    assert pretrain_dataset == 'imagenet'

  mc = model_constants(image_size)
  net_enc = []
  net_enc_id = -1
  for net_enc_type in net_enc_types:
    net_enc_id += 1
    if net_enc_type == 'resnet18':  # pretrained resnet18
      net_enc.append(resnet_18(get_perceptual_feats=True, num_classes=n_classes_in_enc,
                               image_size=image_size))
      state = pt.load(net_enc_files[net_enc_id])
      net = state['net']

      # add "num_batches_tracked" keys to loaded statedict. these shouldn't matter in eval mode
      tensors_to_add = dict()
      for key in net_enc[-1].state_dict().keys():
        if key not in net.state_dict():
          assert key.endswith('num_batches_tracked')
          new_t = net_enc[-1].state_dict()[key]
          tensors_to_add[key] = 987 * pt.ones_like(new_t)  # dummy entry

      net_enc[-1].load_state_dict({**net.state_dict(), **tensors_to_add})
      print(net_enc[-1])
    elif net_enc_type == 'vgg19':  # CIFAR-10 pretrained vgg19
      net_enc.append(vgg_19(get_perceptual_feats=True, num_classes=n_classes_in_enc,
                            image_size=image_size, classifier_depth=mc['vgg_classifier_depth']))
      LOG.info("Reading Feat Extractor #{} from {}".format(net_enc_id, net_enc_files[net_enc_id]))
      state = pt.load(net_enc_files[net_enc_id])
      net = state['net']

      tensors_to_add = dict()
      for key in net_enc[-1].state_dict().keys():
        if key not in net.state_dict():
          assert key.endswith('num_batches_tracked')
          new_t = net_enc[-1].state_dict()[key]
          tensors_to_add[key] = 978 * pt.ones_like(new_t)  # dummy entry

      net_enc[-1].load_state_dict({**net.state_dict(), **tensors_to_add})
    else:
      net_enc.append(ConvEncoderSkipConnections(z_dim, ndf=mc['ndf'], n_channels=mc['nc'],
                                                batch_norm=True,
                                                act_fun='relu', n_convs=mc['n_convs'],
                                                dropout_rate=0.0,
                                                out_layer_act_fun='tanh',
                                                use_fc_for_last_layer=False,
                                                n_extra_layers=mc['n_enc_extra_layers']))

      net_enc[-1].apply(weights_init)
      if net_enc_files[net_enc_id] != '':
        net_enc[-1].load_state_dict(pt.load(net_enc_files[net_enc_id]))

    net_enc[-1].to(device)
    net_enc[-1].eval()
    LOG.info(f'# Encoder :{net_enc_type}')
    for param in net_enc[-1].parameters():
      param.requires_grad = False
  return net_enc


def model_constants(image_size):
  """
  offloading some args which we'll never change to constants here
  """
  constants = dict()
  constants['ndf'] = 64
  constants['ngf'] = 64
  constants['n_enc_extra_layers'] = 2
  constants['n_dec_extra_layers'] = 2
  constants['vgg_classifier_depth'] = 1
  constants['nc'] = 3
  constants['no_adapt_filter_size'] = False
  constants['use_conv_at_gen_skip_conn'] = False
  if image_size == 48:
    constants['first_deconv_kernel_size'] = 6
  else:
    constants['first_deconv_kernel_size'] = 4

  if image_size in [256, 224]:
    constants['n_convs'] = 6
  elif image_size == 128:
    constants['n_convs'] = 5
  elif image_size in [32, 48]:
    constants['n_convs'] = 3
  else:
    constants['n_convs'] = 4
  return constants


def get_generator(net_gen_type, image_size, z_dim, gen_output, device, ckpt, n_static_samples,
                  ddp_rank=None):
  mc = model_constants(image_size)
  # CREATES THE GENERATOR
  LOG.info('# Generator:')
  if net_gen_type == "dcgan":
    net_gen = DeconvDecoder(z_dim, mc['ngf'], n_channels=mc['nc'], batch_norm=True,
                            use_relu=True, n_convs=mc['n_convs'],
                            n_extra_layers=mc['n_dec_extra_layers'],
                            first_deconv_kernel_size=mc['first_deconv_kernel_size'],
                            gen_output=gen_output)
    net_gen.apply(weights_init)
  elif net_gen_type == "resnet":
    net_gen = ResnetG(z_dim, mc['nc'], mc['ngf'], image_size,
                      adapt_filter_size=not mc['no_adapt_filter_size'],
                      use_conv_at_skip_conn=mc['use_conv_at_gen_skip_conn'],
                      gen_output=gen_output)
  elif net_gen_type == 'condconvgen':

    net_gen = ConvCondGen(z_dim - 10, '500,500', 10, '16,8', '5,5', use_sigmoid=True,
                          batch_norm=True)
  elif net_gen_type == 'static':

    net_gen = StaticDataset(n_static_samples, image_size)
  else:
    raise ValueError(f'{net_gen_type} not in dcgan / resnet')

  net_gen.to(device)
  if ddp_rank is not None:
    net_gen = DDP(net_gen, device_ids=[ddp_rank])

  if ckpt is not None:
    net_gen.load_state_dict(ckpt['net_gen'])

  LOG.debug(net_gen)
  return net_gen


class StaticDataset(nn.Module):
  def __init__(self, n_samples, image_size, n_channels=3, gen_output='linear'):
    super(StaticDataset, self).__init__()
    self.weights = nn.Parameter(pt.randn(n_samples, n_channels, image_size, image_size,
                                         requires_grad=True))
    self.activation = nn.Tanh() if gen_output == 'tanh' else nn.Identity()
    self.labels = None

  def forward(self, _args):
    return self.activation(self.weights)


def weights_init(m):
  # custom weights initialization called on netG
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('ConvEncoder') == -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('Linear') != -1:
    m.weight.data.normal_(0.0, 0.02)
    m.bias.data.fill_(0)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


def get_mean_and_var_nets(n_feats_in_enc, device, ckpt, ddp_rank, n_classes,
                          mean_ckpt_name='net_mean', var_ckpt_name='net_var'):
  n_feats_out = 1 if n_classes is None else n_classes
  net_mean = ConstMat(n_feats_in_enc, n_feats_out, device)
  net_var = ConstMat(n_feats_in_enc, n_feats_out, device)

  if ddp_rank is not None:
    net_mean = DDP(net_mean, device_ids=[ddp_rank])
    net_var = DDP(net_var, device_ids=[ddp_rank])
  if ckpt is not None:
    net_mean.load_state_dict(ckpt[mean_ckpt_name])
    net_var.load_state_dict(ckpt[var_ckpt_name])
  LOG.debug(net_mean)
  LOG.debug(net_var)
  return net_mean, net_var


class ConstMat(nn.Module):  # hacky way to get DDP to work with these single parameters
  def __init__(self, in_features, out_features, device, dtype=None):
    super(ConstMat, self).__init__()
    self.out_features = out_features
    self.weight = nn.Parameter(pt.empty((out_features, in_features), device=device, dtype=dtype))
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, _):
    if self.out_features == 1:
      return pt.squeeze(self.weight)
    else:
      return self.weight

  def fwd(self):
    return self.forward(None)


def small_data_model(pretrain_data, model, device, image_size):
  if pretrain_data == 'cifar10_pretrain':
    model_path = 'models/Trained_ResNet_cifar10'
  else:
    model_path = f'models/Trained_ResNet_{pretrain_data}'
  if model != 'dummy':
    model = 'resnet18'
  # if model == 'VGG':
  #   net = VGG('VGG15')
  #   net = net.to(device)
  #   if device == 'cuda':
  #     net = pt.nn.DataParallel(net)
  #   checkpoint = pt.load(model_path)
  #   net.load_state_dict(checkpoint['net'], strict=False)

  if model == 'resnet18':
    net = resnet_18(get_perceptual_feats=True, num_classes=10, image_size=image_size,
                    grayscale_input=True).to(device)
    checkpoint = pt.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
  elif model == 'dummy':
    net = DummyEncoder(num_classes=10, image_size=image_size).to(device)
  else:
    raise ValueError

  for param in net.parameters():
    param.requires_grad = False

  return [net]


class DummyEncoder(nn.Module):
  def __init__(self, num_classes, image_size, return_features=True, channel_dim=1):
    super(DummyEncoder, self).__init__()
    self.lin = nn.Linear(image_size ** 2, num_classes)
    self.return_features = return_features
    self.channel_dim = channel_dim

  def forward(self, x):
    if x.shape[self.channel_dim] > 1:
      x = x[:, 0, :, :]
    x = pt.flatten(x, start_dim=1)
    x = self.lin(x)
    if self.return_features:
      return x, [x]
    else:
      return x
