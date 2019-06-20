import torch.nn as nn
from torch.nn.modules.utils import _pair as make_pair

from models.svdo_layers import Conv2dSVDO


def conv_1x1(in_channels, out_channels, stride=1, sparse=False):
    conv_layer = Conv2dSVDO if sparse else nn.Conv2d
    return conv_layer(in_channels, out_channels, 1, stride=stride, bias=False)


def conv_3x3(in_channels, out_channels, stride=1, sparse=False):
    conv_layer = Conv2dSVDO if sparse else nn.Conv2d
    return conv_layer(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)


class InvertibleDownsample(nn.Module):
    def __init__(self, block_size):
        super(InvertibleDownsample, self).__init__()
        self.block_size = make_pair(block_size)

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        if height % self.block_size[0] != 0:
            raise ValueError('Image height must be divisible by {}'.format(self.block_size[0]))
        if height % self.block_size[1] != 0:
            raise ValueError('Image width must be divisible by {}'.format(self.block_size[1]))
        out = x.view(batch_size, num_channels, height // self.block_size[0],
                     self.block_size[0], width // self.block_size[1], self.block_size[1])
        out = out.permute(0, 3, 5, 1, 2, 4).contiguous()
        out = out.view(batch_size, num_channels * self.block_size[0] * self.block_size[1],
                       height // self.block_size[0], width // self.block_size[1])
        return out

    def extra_repr(self):
        return 'block_size={}'.format(self.block_size)


class DownsampleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, sparse=False, kernel_size=3):
        super(DownsampleConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse
        self.kernel_size = make_pair(kernel_size)
        self.downsample = InvertibleDownsample(stride)
        if kernel_size == 3:
            self.conv = conv_3x3(in_channels * (stride ** 2), out_channels, sparse=sparse)
        else:
            self.conv = conv_1x1(in_channels * (stride ** 2), out_channels, sparse=sparse)

    def forward(self, x):
        return self.conv(self.downsample(x))

    def deterministic_forward(self, x):
        if not self.sparse:
            return self.forward(x)
        else:
            return self.conv.deterministic_forward(self.downsample(x))

    def kl_divergence(self):
        if self.sparse:
            return self.conv.kl_divergence()
        else:
            return 0.0


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse

        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.conv_1 = DownsampleConv2d(in_channels, out_channels, stride=stride, sparse=sparse)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.conv_2 = conv_3x3(out_channels, out_channels, sparse=sparse)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = DownsampleConv2d(in_channels, out_channels,
                                             stride=stride, sparse=sparse, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        output = self.conv_1(self.relu(self.bn_1(x)))
        output = self.conv_2(self.relu(self.bn_2(output)))
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return output + residual

    def deterministic_forward(self, x):
        if not self.sparse:
            return self.forward(x)
        else:
            output = self.conv_1.deterministic_forward(self.relu(self.bn_1(x)))
            output = self.conv_2.deterministic_forward(self.relu(self.bn_2(output)))
            residual = x
            if self.shortcut is not None:
                residual = self.shortcut.deterministic_forward(x)
            return output + residual

    def kl_divergence(self):
        if self.sparse:
            total_kl = self.conv_1.kl_divergence()
            total_kl = total_kl + self.conv_2.kl_divergence()
            if self.shortcut is not None:
                total_kl = total_kl + self.shortcut.kl_divergence()
            return total_kl
        return 0.0


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.bottleneck = out_channels // 4
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse

        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.conv_1 = conv_1x1(in_channels, self.bottleneck, sparse=sparse)
        self.bn_2 = nn.BatchNorm2d(self.bottleneck)
        self.conv_2 = DownsampleConv2d(self.bottleneck, self.bottleneck, stride=stride, sparse=sparse)
        self.bn_3 = nn.BatchNorm2d(self.bottleneck)
        self.conv_3 = conv_1x1(self.bottleneck, out_channels, sparse=sparse)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = DownsampleConv2d(in_channels, out_channels,
                                             stride=stride, sparse=sparse, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv_1(self.relu(self.bn_1(x)))
        out = self.conv_2(self.relu(self.bn_2(out)))
        out = self.conv_3(self.relu(self.bn_3(out)))
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return out + residual

    def deterministic_forward(self, x):
        if not self.sparse:
            return self.forward(x)
        else:
            output = self.conv_1.deterministic_forward(self.relu(self.bn_1(x)))
            output = self.conv_2.deterministic_forward(self.relu(self.bn_2(output)))
            output = self.conv_3.deterministic_forward(self.relu(self.bn_3(output)))
            residual = x
            if self.shortcut is not None:
                residual = self.shortcut.deterministic_forward(x)
            return output + residual

    def kl_divergence(self):
        if self.sparse:
            total_kl = self.conv_1.kl_divergence()
            total_kl = total_kl + self.conv_2.kl_divergence()
            total_kl = total_kl + self.conv_3.kl_divergence()
            if self.shortcut is not None:
                total_kl = total_kl + self.shortcut.kl_divergence()
            return total_kl
        return 0.0
