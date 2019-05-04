import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_resnet.layers import LinearSVDO, Conv2dSVDO


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, sparse=False, conv_first=False, use_bn=True, activation=F.relu):
        super(ResidualLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse
        self.conv_first = conv_first
        self.use_bn = use_bn
        self.activation = activation

        if sparse:
            self.conv_layer = Conv2dSVDO(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        else:
            self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        if self.use_bn:
            self.bn_layer = nn.BatchNorm2d(out_channels if conv_first else in_channels)

    def forward(self, x):
        if self.conv_first:
            output = self.conv_layer(x)
            if self.use_bn:
                output = self.bn_layer(output)
            if self.activation is not None:
                output = self.activation(output)
        else:
            if self.use_bn:
                output = self.bn_layer(x)
            if self.activation is not None:
                output = self.activation(output)
            output = self.conv_layer(output)
        return output

    def kullback_leibler_divergence(self):
        if self.sparse:
            return self.conv_layer.kullback_leibler_divergence()
        else:
            return 0.0


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False, version=2):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse
        self.version = version

        if version == 1:
            self.layer_1 = ResidualLayer(in_channels, out_channels, 3, padding=1,
                                         stride=stride, sparse=sparse, conv_first=True)
            self.layer_2 = ResidualLayer(out_channels, out_channels, 3, padding=1,
                                         sparse=sparse, conv_first=True, activation=None)
        elif version == 2:
            self.layer_1 = ResidualLayer(in_channels, out_channels, 3, padding=1,
                                         stride=stride, sparse=sparse, conv_first=False)
            self.layer_2 = ResidualLayer(out_channels, out_channels, 3, padding=1,
                                         sparse=sparse, conv_first=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if sparse:
                self.shortcut = Conv2dSVDO(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, block_input):
        block_output = self.layer_2(self.layer_1(block_input))
        block_output = self.shortcut(block_input) + block_output
        if self.version == 1:
            return F.relu(block_output)
        elif self.version == 2:
            return block_output

    def kullback_leibler_divergence(self):
        total_kl = 0.0
        if self.sparse:
            total_kl = self.layer_1.kullback_leibler_divergence()
            total_kl = total_kl + self.layer_2.kullback_leibler_divergence()
            if isinstance(self.shortcut, Conv2dSVDO):
                total_kl = total_kl + self.shortcut.kullback_leibler_divergence()
        return total_kl


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False, version=2):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.bottleneck = in_channels
        self.out_channels = out_channels
        self.sparse = sparse
        self.version = version
        if version == 1:
            self.layer_1 = ResidualLayer(in_channels, self.bottleneck, 1, stride=stride,
                                         sparse=sparse, conv_first=True)
            self.layer_2 = ResidualLayer(self.bottleneck, self.bottleneck, 3,
                                         padding=1, sparse=sparse, conv_first=True)
            self.layer_3 = ResidualLayer(self.bottleneck, out_channels, 1,
                                         sparse=sparse, conv_first=True, activation=None)
        else:
            self.layer_1 = ResidualLayer(in_channels, self.bottleneck, 1, stride=stride,
                                         sparse=sparse, conv_first=False)
            self.layer_2 = ResidualLayer(self.bottleneck, self.bottleneck, 3, padding=1,
                                         sparse=sparse, conv_first=False)
            self.layer_3 = ResidualLayer(self.bottleneck, out_channels, 1, sparse=sparse,
                                         conv_first=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if sparse:
                self.shortcut = Conv2dSVDO(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, block_input):
        block_output = self.layer_3(self.layer_2(self.layer_1(block_input)))
        block_output = self.shortcut(block_input) + block_output
        if self.version == 1:
            return F.relu(block_output)
        elif self.version == 2:
            return block_output

    def kullback_leibler_divergence(self):
        total_kl = 0.0
        if self.sparse:
            total_kl = self.layer_1.kullback_leibler_divergence()
            total_kl = total_kl + self.layer_2.kullback_leibler_divergence()
            total_kl = total_kl + self.layer_3.kullback_leibler_divergence()
            if isinstance(self.shortcut, Conv2dSVDO):
                total_kl = total_kl + self.shortcut.kullback_leibler_divergence()
        return total_kl


class LinearClassifier(nn.Module):
    def __init__(self, in_features, out_features, sparse=False, version=2):
        super(LinearClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparse = sparse
        self.version = version
        if version == 2:
            self.bn = nn.BatchNorm2d(in_features)
        if sparse:
            self.linear = LinearSVDO(in_features, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        output = x
        if self.version == 2:
            output = F.relu(self.bn(x))
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = output.view(output.size(0), -1)
        output = F.log_softmax(self.linear(output), dim=-1)
        return output

    def kullback_leibler_divergence(self):
        if self.sparse:
            return self.linear.kullback_leibler_divergence()
        else:
            return 0.0


class WeakClassifier(nn.Module):
    def __init__(self, block, classifier):
        super(WeakClassifier, self).__init__()
        self.block = block
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.block(x))

    def kullback_leibler_divergence(self):
        total_kl = 0.0
        if hasattr(self.block, 'kullback_leibler_divergence'):
            total_kl = total_kl + self.block.kullback_leibler_divergence()
        if hasattr(self.classifier, 'kullback_leibler_divergence'):
            total_kl = total_kl + self.classifier.kullback_leibler_divergence()
        return total_kl


def make_resnet(n, sparse=False, sequential=False, version=2):
    def make_blocks(block_generator, in_channels, out_channels, stride, num_blocks):
        block_layers = [block_generator(in_channels, out_channels, stride=stride, sparse=sparse, version=version)]
        for _ in range(num_blocks):
            block_layers.append(block_generator(out_channels, out_channels, sparse=sparse, version=version))
        return block_layers

    block = BasicBlock if version == 1 else BottleNeck
    channels = [16, 16, 32, 64] if version == 1 else [16, 64, 128, 256]
    layers = [ResidualLayer(3, channels[0], 3, padding=1, sparse=sparse, conv_first=True)]
    layers.extend(make_blocks(block, channels[0], channels[1], 1, n))
    layers.extend(make_blocks(block, channels[1], channels[2], 2, n))
    layers.extend(make_blocks(block, channels[2], channels[3], 2, n))
    if sequential:
        weak_classifiers = []
        for layer in layers:
            weak_classifier = WeakClassifier(layer,
                                             LinearClassifier(layer.out_channels, 10, sparse=sparse, version=version))
            weak_classifiers.append(weak_classifier)
        return weak_classifiers
    else:
        layers.append(LinearClassifier(layers[-1].out_channels, 10, sparse=sparse, version=version))
        return nn.Sequential(*layers)
