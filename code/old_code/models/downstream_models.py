import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(wide_basic, self).__init__()
        if norm_layer is None:
            self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True, momentum=0.1)
        else:
            self.bn1 = norm_layer(num_channels=in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=True)

        if norm_layer is None:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True, momentum=0.1)
        else:
            self.bn2 = norm_layer(num_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.shortcut is not None:
            out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, num_layers, depth, widen_factor, num_classes, num_input_channels=3,
                 norm_layer=None):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        self.num_layers = num_layers

        assert ((depth-4)%6 ==0), 'wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| wide-resnet %dx%d' %(depth, k))
        nstages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(num_input_channels, nstages[0])
        if self.num_layers >= 1:
            self.layer1 = self._wide_layer(wide_basic, nstages[1], n, stride=1, norm_layer=norm_layer)
        if self.num_layers >= 2:
            self.layer2 = self._wide_layer(wide_basic, nstages[2], n, stride=2, norm_layer=norm_layer)
        if self.num_layers == 3:
            self.layer3 = self._wide_layer(wide_basic, nstages[3], n, stride=2, norm_layer=norm_layer)
        if norm_layer is None:
            self.bn1 = nn.BatchNorm2d(nstages[num_layers], track_running_stats=True, momentum=0.1)
        else:
            self.bn1 = norm_layer(num_channels=nstages[num_layers])
        self.linear = nn.Linear(nstages[num_layers], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride, norm_layer=None):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, input_dict):
        x = input_dict['inputs']
        out = self.conv1(x)
        if self.num_layers == 1:
            out1 = self.layer1(out)
            out = F.relu(self.bn1(out1))
            # out = F.relu(out1)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output_dict = {
                'logits': out, 'features': [out1], 'embedding': emb
            }
        elif self.num_layers == 2:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out = F.relu(self.bn1(out2))
            # out = F.relu(out2)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output_dict = {
                'logits': out, 'features': [out1, out2], 'embedding': emb
            }
        elif self.num_layers == 3:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out = F.relu(self.bn1(out3))
            out = F.adaptive_avg_pool2d(out, (1, 1))
            emb = out.view(out.size(0), -1)
            out = self.linear(emb)

            output_dict = {
                'logits': out, 'features': [out1, out2, out3], 'embedding': emb
            }
        else:
            raise ValueError('Specify valid number of layers. Now it is {}'.format(self.num_layers))
        return output_dict


# WideResNet(1, 28, 10, 10)

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x): return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                        stride=stride, padding=padding, groups=groups, bias=False),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True)
    )


def get_ffcv_model(device):
    num_class = 10
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last).to(device)
    return model
