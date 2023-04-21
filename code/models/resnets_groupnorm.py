from typing import Type, Any, Union, List, Optional
from collections import OrderedDict
from math import log

import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        affine_gn: bool = True

    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(planes, planes, affine=affine_gn)
        self.relu = nn.ReLU(inplace=affine_gn)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(planes, planes, affine=affine_gn)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
  pass
#     # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
#     # while original implementation places the stride at the first 1x1 convolution(self.conv1)
#     # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
#     # This variant is also known as ResNet V1.5 and improves accuracy according to
#     # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
#
#     expansion: int = 4
#
#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#
#     ) -> None:
#         super().__init__()
#
#         width = int(planes * (base_width / 64.0)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = nn.GroupNorm(width, width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = nn.GroupNorm(width, width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = nn.GroupNorm(planes * self.expansion, planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x: Tensor) -> Tensor:
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        planes: int = 64,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        affine_gn: bool = True,
        use_sigmoid: bool = True
    ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = planes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group


        self.net = nn.Sequential(
          nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
          nn.GroupNorm(planes, planes, affine=affine_gn),
          nn.ReLU(inplace=affine_gn),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          # self.layer1, self.layer2, self.layer3, self.layer4,
          self._make_layer(block, planes, layers[0], affine_gn=affine_gn),
          self._make_layer(block, planes * 2, layers[1],
                           dilate=replace_stride_with_dilation[0], affine_gn=affine_gn),
          self._make_layer(block, planes * 4, layers[2], stride=2,
                           dilate=replace_stride_with_dilation[1], affine_gn=affine_gn),
          self._make_layer(block, planes * 8, layers[3], stride=2,
                           dilate=replace_stride_with_dilation[2], affine_gn=affine_gn),
          nn.AdaptiveAvgPool2d((1, 1)),
          nn.Flatten(),
          nn.Linear(planes * 8 * block.expansion, 1),
          # nn.Linear(128, 1),
          nn.Sigmoid() if use_sigmoid else nn.Identity()
          # self.flatlayer, self.fc,
          # self.sigmoid

        )

    def make_layer_custom(self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        block = BasicBlock
        affine_gn = True
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
        # if stride != 1 or self.planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(planes * block.expansion, planes * block.expansion, affine=affine_gn),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                previous_dilation, affine_gn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    affine_gn=affine_gn
                )
            )

        return nn.Sequential(*layers)

    class BasicBlockSimple(nn.Module):
      expansion: int = 1

      def __init__(
          self,
          inplanes: int,
          planes: int,
          stride: int = 1,
      ) -> None:
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(planes, planes, affine=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(planes, planes, affine=False)
        self.stride = stride

      def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)

        return out

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        affine_gn: bool = True
    ) -> nn.Sequential:

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
        # if stride != 1 or self.planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(planes * block.expansion, planes * block.expansion, affine=affine_gn),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                previous_dilation, affine_gn
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    affine_gn=affine_gn
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.net(x)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # x = self.sigmoid(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # return self._forward_impl(x)
        return self.net(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    return model


def get_resnet9_discriminator(ndf, affine_gn=True, use_sigmoid=True):
  return _resnet(BasicBlock, [1, 1, 1, 1], planes=ndf, affine_gn=affine_gn, use_sigmoid=use_sigmoid)


# def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
#   """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.
#   return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


class ResnetGroupnormG(nn.Module):
  def __init__(self, enc_size, nc, ndf, image_size=32, adapt_filter_size=False,
               use_conv_at_skip_conn=False, gen_output='tanh', affine_gn=True):
    super(ResnetGroupnormG, self).__init__()
    self.enc_size = enc_size
    self.ndf = ndf
    self.gen_output = gen_output

    if adapt_filter_size is True and use_conv_at_skip_conn is False:
      use_conv_at_skip_conn = True

    n_upsample_blocks = int(log(image_size, 2)) - 2

    n_layers = n_upsample_blocks + 1
    filter_sizes = [ndf] * n_layers
    if adapt_filter_size:
      for i in range(n_layers - 1, -1, -1):
        if i == n_layers - 1:
          filter_sizes[i] = ndf
        else:
          filter_sizes[i] = filter_sizes[i+1]*2

    first_layer = nn.ConvTranspose2d(enc_size, filter_sizes[0], 4, 1, 0, bias=False)
    nn.init.xavier_uniform_(first_layer.weight.data, 1.)
    last_layer = nn.Conv2d(filter_sizes[-1], nc, 3, stride=1, padding=1)
    nn.init.xavier_uniform_(last_layer.weight.data, 1.)

    nn_layers = OrderedDict()
    # first deconv goes from the z size
    nn_layers["firstConv"] = first_layer

    layer_number = 1
    for i in range(n_upsample_blocks):
      nn_layers[f"resblock_{i}"] = ResidualBlockGroupnormG(filter_sizes[layer_number-1],
                                                           filter_sizes[layer_number], 2,
                                                           use_conv_at_skip_conn, affine_gn)
      layer_number += 1
    nn_layers["batchNorm"] = nn.GroupNorm(filter_sizes[-1], filter_sizes[-1], affine=affine_gn)
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
    return pt.randn(batch_size, self.enc_size, 1, 1)


class ResidualBlockGroupnormG(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1, use_conv_at_skip_conn=False, affine_gn=True):
    super(ResidualBlockGroupnormG, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

    if use_conv_at_skip_conn:
      self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
      nn.init.xavier_uniform_(self.conv_bypass.weight.data, 1.)

    nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
    nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

    self.model = nn.Sequential(
      nn.GroupNorm(in_channels, in_channels, affine=affine_gn),
      nn.ReLU(),
      Upsample(scale_factor=2),
      self.conv1,
      nn.GroupNorm(out_channels, out_channels, affine=affine_gn),
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


class ResNetFlat(nn.Module):
  def __init__(
      self,
      planes: int = 64,
      width_per_group: int = 64,
      n_classes: int = None,
      use_sigmoid: bool = True,
      image_size: int = 32
  ) -> None:
    super(ResNetFlat, self).__init__()
    self.labeled = n_classes is not None
    self.base_width = width_per_group
    assert image_size in {64, 32}
    self.image_size = image_size
    self.relu = nn.ReLU(inplace=False)

    self.l0_step1_conv = nn.Conv2d(3, planes, kernel_size=7, stride=2, padding=3, bias=False)
    self.l0_step2_gn = nn.GroupNorm(planes, planes, affine=False)
    self.l0_step4_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # self._make_layer_custom(planes, layers[0], affine_gn=affine_gn)
    self.b1_step1_conv1 = conv3x3(planes, planes)
    self.b1_step2_bn1 = nn.GroupNorm(planes, planes, affine=False)
    # self.b1_step3_relu = nn.ReLU(inplace=False)
    self.b1_step4_conv2 = conv3x3(planes, planes)
    self.b1_step5_bn2 = nn.GroupNorm(planes, planes, affine=False)

    # make_layer(block, planes * 2, layers[1], False, affine_gn=affine_gn)
    self.b2_step1_conv1 = conv3x3(planes, planes * 2)
    self.b2_step2_bn1 = nn.GroupNorm(planes * 2, planes * 2, affine=False)
    # self.b2_step3_relu = nn.ReLU(inplace=False)
    self.b2_step4_conv2 = conv3x3(planes * 2, planes * 2)
    self.b2_step5_bn2 = nn.GroupNorm(planes * 2, planes * 2, affine=False)
    self.b2_step6_downsample_conv = conv1x1(planes, planes * 2, stride=1)
    self.b2_step7_downsample_gn = nn.GroupNorm(planes * 2, planes * 2, affine=False)

    # make_layer(block, planes * 4, layers[2], stride=2, affine_gn=affine_gn)
    self.b3_step1_conv1 = conv3x3(planes * 2, planes * 4, stride=2)
    self.b3_step2_bn1 = nn.GroupNorm(planes * 4, planes * 4, affine=False)
    # self.b3_step3_relu = nn.ReLU(inplace=False)
    self.b3_step4_conv2 = conv3x3(planes * 4, planes * 4)
    self.b3_step5_bn2 = nn.GroupNorm(planes * 4, planes * 4, affine=False)
    self.b3_step6_downsample_conv = conv1x1(planes * 2, planes * 4, stride=2)
    self.b3_step7_downsample_gn = nn.GroupNorm(planes * 4, planes * 4, affine=False)

    # make_layer(block, planes * 8, layers[3], stride=2)
    self.b4_step1_conv1 = conv3x3(planes * 4, planes * 8, stride=2)
    self.b4_step2_bn1 = nn.GroupNorm(planes * 8, planes * 8, affine=False)
    # self.b4_step3_relu = nn.ReLU(inplace=False)
    self.b4_step4_conv2 = conv3x3(planes * 8, planes * 8)
    self.b4_step5_bn2 = nn.GroupNorm(planes * 8, planes * 8, affine=False)
    self.b4_step6_downsample_conv = conv1x1(planes * 4, planes * 8, stride=2)
    self.b4_step7_downsample_gn = nn.GroupNorm(planes * 8, planes * 8, affine=False)

    if self.image_size == 64:
      self.b45_step1_conv1 = conv3x3(planes * 8, planes * 16, stride=2)
      self.b45_step2_bn1 = nn.GroupNorm(planes * 16, planes * 16, affine=False)
      # self.b4_step3_relu = nn.ReLU(inplace=False)
      self.b45_step4_conv2 = conv3x3(planes * 16, planes * 16)
      self.b45_step5_bn2 = nn.GroupNorm(planes * 16, planes * 16, affine=False)
      self.b45_step6_downsample_conv = conv1x1(planes * 8, planes * 16, stride=2)
      self.b45_step7_downsample_gn = nn.GroupNorm(planes * 16, planes * 16, affine=False)
      planes_multiplier = 16
    else:
      self.b45_step1_conv1 = None
      self.b45_step2_bn1 = None
      self.b45_step4_conv2 = None
      self.b45_step5_bn2 = None
      self.b45_step6_downsample_conv = None
      self.b45_step7_downsample_gn = None
      planes_multiplier = 8

    self.l5_step1_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.l5_step2_flat = nn.Flatten()
    if n_classes is not None:
      self.l5_step3_fc = nn.Linear(planes * planes_multiplier + n_classes,
                                   planes * planes_multiplier // 2)
      self.l5_relu = nn.ReLU()
      self.l5_step3_fc2 = nn.Linear(planes * planes_multiplier // 2, 1)
    else:
      self.l5_step3_fc = nn.Linear(planes * planes_multiplier, 1)
      self.l5_relu = nn.Identity()
      self.l5_step3_fc2 = nn.Identity()
    self.l5_step4_sig = nn.Sigmoid() if use_sigmoid else nn.Identity()

  def forward(self, x, labels=None):
    # l0
    out = self.l0_step1_conv(x)
    out = self.l0_step2_gn(out)
    out = self.relu(out)
    out = self.l0_step4_pool(out)

    # b1
    identity = out

    out = self.b1_step1_conv1(out)
    out = self.b1_step2_bn1(out)
    out = self.relu(out)
    out = self.b1_step4_conv2(out)
    out = self.b1_step5_bn2(out)
    out = out + identity
    out = self.relu(out)

    # b2
    id_conv = self.b2_step6_downsample_conv(out)
    id_ds = self.b2_step7_downsample_gn(id_conv)

    out = self.b2_step1_conv1(out)
    out = self.b2_step2_bn1(out)
    out = self.relu(out)
    out = self.b2_step4_conv2(out)
    out = self.b2_step5_bn2(out)
    out = out + id_ds
    out = self.relu(out)

    # b3
    id_conv = self.b3_step6_downsample_conv(out)
    id_ds = self.b3_step7_downsample_gn(id_conv)

    out = self.b3_step1_conv1(out)
    out = self.b3_step2_bn1(out)
    out = self.relu(out)
    out = self.b3_step4_conv2(out)
    out = self.b3_step5_bn2(out)
    out = out + id_ds
    out = self.relu(out)

    # b4
    id_conv = self.b4_step6_downsample_conv(out)
    id_ds = self.b4_step7_downsample_gn(id_conv)

    out = self.b4_step1_conv1(out)
    out = self.b4_step2_bn1(out)
    out = self.relu(out)
    out = self.b4_step4_conv2(out)
    out = self.b4_step5_bn2(out)
    out = out + id_ds
    out = self.relu(out)

    if self.image_size == 64:
      id_conv = self.b45_step6_downsample_conv(out)
      id_ds = self.b45_step7_downsample_gn(id_conv)

      out = self.b45_step1_conv1(out)
      out = self.b45_step2_bn1(out)
      out = self.relu(out)
      out = self.b45_step4_conv2(out)
      out = self.b45_step5_bn2(out)
      out = out + id_ds
      out = self.relu(out)

    # l5
    out = self.l5_step1_pool(out)
    out = self.l5_step2_flat(out)
    if self.labeled:
      out = self.l5_step3_fc(pt.cat((out, labels), dim=1))
      out = self.l5_relu(out)
      out = self.l5_step3_fc2(out)
    else:
      out = self.l5_step3_fc(out)
    out = self.l5_step4_sig(out)
    return out

  def make_layer_custom_stride_two(self,
                        planes: int,
                        stride: int = 1,
                        ) -> nn.Sequential:
    block = BasicBlockSimpleDS
    affine_gn = True
    downsample = None
    previous_dilation = self.dilation

    if stride != 1 or self.inplanes != planes * block.expansion:
      # if stride != 1 or self.planes != planes * block.expansion:
      downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        nn.GroupNorm(planes * block.expansion, planes * block.expansion, affine=affine_gn),
      )

    layers = []
    layers.append(
      block(self.inplanes, planes, stride)
    )
    self.inplanes = planes * block.expansion

    return nn.Sequential(*layers)


class BasicBlockSimpleNoDS(nn.Module):
  expansion: int = 1

  def __init__(
      self,
      inplanes: int,
      planes: int,
  ) -> None:
    super().__init__()

    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes)
    self.bn1 = nn.GroupNorm(planes, planes, affine=False)
    self.relu = nn.ReLU(inplace=False)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.GroupNorm(planes, planes, affine=False)

  def forward(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + identity
    out = self.relu(out)

    return out


class BasicBlockSimpleDS(nn.Module):
  expansion: int = 1

  def __init__(
      self,
      inplanes: int,
      planes: int,
  ) -> None:
    super().__init__()
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv3x3(inplanes, planes, stride=2)
    self.bn1 = nn.GroupNorm(planes, planes, affine=False)
    self.relu = nn.ReLU(inplace=False)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.GroupNorm(planes, planes, affine=False)
    self.downsample_conv = conv1x1(self.inplanes, planes, stride=2)
    self.downsample_gn = nn.GroupNorm(planes, planes, affine=False)

  def forward(self, x: Tensor) -> Tensor:
    id_conv = self.downsample_conv(x)
    id_ds = self.downsample_gn(id_conv)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + id_ds
    out = self.relu(out)

    return out
