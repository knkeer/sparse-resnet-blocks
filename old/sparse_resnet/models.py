import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_resnet.layers import LinearSVDO, Conv2dSVDO


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.bottleneck = out_channels // 4
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse
        
        conv_layer = Conv2dSVDO if sparse else nn.Conv2d
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.conv_1 = conv_layer(in_channels, self.bottleneck, kernel_size=1, stride=self.stride)
        self.bn_2 = nn.BatchNorm2d(self.bottleneck)
        self.conv_2 = conv_layer(self.bottleneck, self.bottleneck, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(self.bottleneck)
        self.conv_3 = conv_layer(self.bottleneck, out_channels, kernel_size=1)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            if sparse:
                self.shortcut = Conv2dSVDO(in_channels, out_channels, kernel_size=1, stride=stride)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_1.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_2.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_2.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_3.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_3.bias, 0.0)
        
        nn.init.constant_(self.bn_1.weight, 1.0)
        nn.init.constant_(self.bn_1.bias, 0.0)
        nn.init.constant_(self.bn_2.weight, 1.0)
        nn.init.constant_(self.bn_2.bias, 0.0)
        nn.init.constant_(self.bn_3.weight, 1.0)
        nn.init.constant_(self.bn_3.bias, 0.0)

    def forward(self, block_input):
        if self.shortcut is None:
            block_output = self.conv_1(F.relu(self.bn_1(block_input)))
            block_output = self.conv_2(F.relu(self.bn_2(block_output)))
            block_output = self.conv_3(F.relu(self.bn_3(block_output)))
            return block_input + block_output
        else:
            x = F.relu(self.bn_1(block_input))
            block_output = self.conv_1(x)
            block_output = self.conv_2(F.relu(self.bn_2(block_output)))
            block_output = self.conv_3(F.relu(self.bn_3(block_output)))
            return self.shortcut(x) + block_output

    def kullback_leibler_divergence(self):
        total_kl = 0.0
        if self.sparse:
            total_kl = self.conv_1.kullback_leibler_divergence()
            total_kl = total_kl + self.conv_2.kullback_leibler_divergence()
            total_kl = total_kl + self.conv_3.kullback_leibler_divergence()
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
        self.bn = nn.BatchNorm2d(in_features)
        linear_layer = LinearSVDO if sparse else nn.Linear
        self.linear = linear_layer(in_features, out_features)

    def forward(self, x):
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
        else:
            for module in self.block.modules():
                if isinstance(module, (LinearSVDO, Conv2dSVDO)):
                    total_kl = total_kl + module.kullback_leibler_divergence()
        if hasattr(self.classifier, 'kullback_leibler_divergence'):
            total_kl = total_kl + self.classifier.kullback_leibler_divergence()
        return total_kl


def make_resnet(num_layers, sparse=False, sequential=False):
    assert (num_layers - 2) % 9 == 0
    n = (num_layers - 2) // 9
    def make_blocks(in_channels, out_channels, stride, num_blocks):
        block_layers = [BottleNeck(in_channels, out_channels, stride=stride, sparse=sparse,)]
        for _ in range(num_blocks):
            block_layers.append(BottleNeck(out_channels, out_channels, sparse=sparse))
        return block_layers

    channels = [16, 64, 128, 256]
    conv_layer = Conv2dSVDO if sparse else nn.Conv2d
    layers = [conv_layer(3, channels[0], kernel_size=3, padding=1)]
    layers.extend(make_blocks(channels[0], channels[1], 1, n))
    layers.extend(make_blocks(channels[1], channels[2], 2, n))
    layers.extend(make_blocks(channels[2], channels[3], 2, n))
    if sequential:
        weak_classifiers = [WeakClassifier(
            nn.Sequential(layers[0], layers[1]),
            LinearClassifier(channels[0], 10, sparse=sparse)
        )]
        for layer in layers[2:]:
            weak_classifier = WeakClassifier(
                layer,
                LinearClassifier(layer.out_channels, 10, sparse=sparse)
            )
            weak_classifiers.append(weak_classifier)
        return weak_classifiers
    else:
        layers.append(LinearClassifier(layers[-1].out_channels, 10, sparse=sparse))
        return nn.Sequential(*layers)

