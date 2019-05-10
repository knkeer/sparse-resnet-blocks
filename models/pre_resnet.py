import torch
import torch.nn as nn
import torch.nn.functional as F

from models import sparse_variational_dropout as svdo


def conv_1x1(in_channels, out_channels, stride=1, sparse=False):
    if sparse:
        return svdo.Conv2dSVDO(in_channels, out_channels, kernel_size=1,
                               stride=stride, bias=False)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False)


def conv_3x3(in_channels, out_channels, stride=1, sparse=False):
    if sparse:
        return svdo.Conv2dSVDO(in_channels, out_channels, kernel_size=3,
                               padding=1, stride=stride, bias=False)
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         padding=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sparse=False):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sparse = sparse

        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.conv_1 = conv_3x3(in_channels, out_channels, stride=stride, sparse=sparse)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.conv_2 = conv_3x3(out_channels, out_channels, sparse=sparse)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_1x1(in_channels, out_channels, stride=stride, sparse=sparse)
        else:
            self.shortcut = None

    def forward(self, x):
        output = self.conv_1(self.relu(self.bn_1(x)))
        output = self.conv_2(self.relu(self.bn_2(output)))
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return output + residual

    def kullback_leibler_divergence(self):
        if self.sparse:
            total_kl = self.conv_1.kullback_leibler_divergence()
            total_kl = total_kl + self.conv_2.kullback_leibler_divergence()
            if self.shortcut is not None:
                total_kl = total_kl + self.shortcut.kullback_leibler_divergence()
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
        self.conv_2 = conv_3x3(self.bottleneck, self.bottleneck, stride=stride, sparse=sparse)
        self.bn_3 = nn.BatchNorm2d(self.bottleneck)
        self.conv_3 = conv_1x1(self.bottleneck, out_channels, sparse=sparse)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_1x1(in_channels, out_channels, stride=stride, sparse=sparse)
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

    def kullback_leibler_divergence(self):
        if self.sparse:
            total_kl = self.conv_1.kullback_leibler_divergence()
            total_kl = total_kl + self.conv_2.kullback_leibler_divergence()
            total_kl = total_kl + self.conv_3.kullback_leibler_divergence()
            if self.shortcut is not None:
                total_kl = total_kl + self.shortcut.kullback_leibler_divergence()
            return total_kl
        return 0.0


class LinearClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=10, sparse=False):
        super(LinearClassifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sparse = sparse

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if sparse:
            self.linear = svdo.LinearSVDO(in_channels, num_classes)
        else:
            self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.avg_pool(self.relu(self.bn(x)))
        return F.log_softmax(self.linear(out.view(out.size(0), -1)), dim=-1)

    def kullback_leibler_divergence(self):
        if self.sparse:
            return self.linear.kullback_leibler_divergence()
        return 0.0


class WeakClassifier(nn.Module):
    def __init__(self, block, num_classes=10, sparse=False, num_channels=None):
        super(WeakClassifier, self).__init__()
        self.num_classes = num_classes
        if num_channels is not None:
            self.num_channels = num_channels
        else:
            self.num_channels = block.out_channels
        self.sparse = sparse
        self.block = block
        self.classifier = LinearClassifier(self.num_channels, num_classes=num_classes, sparse=sparse)

    def forward(self, x):
        return self.classifier(self.block(x))

    def kullback_leibler_divergence(self):
        total_kl = 0.0
        if hasattr(self.block, 'kullback_leibler_divergence'):
            total_kl = total_kl + self.block.kullback_leibler_divergence()
        else:
            for module in self.block.modules():
                if isinstance(module, (svdo.LinearSVDO, svdo.Conv2dSVDO)):
                    total_kl = total_kl + module.kullback_leibler_divergence()
        if self.sparse:
            total_kl = total_kl + self.classifier.kullback_leibler_divergence()
        return total_kl


class LayerwiseSequential(nn.Module):
    def __init__(self, weak_classifier=None):
        super(LayerwiseSequential, self).__init__()
        self.trained_blocks = nn.ModuleList()
        self.weak_classifier = weak_classifier

    def add_weak_classifier(self, new_weak_classifier):
        if self.weak_classifier is not None:
            self.trained_blocks.append(self.weak_classifier.block)
            for trained_block in self.trained_blocks:
                for parameter in trained_block.parameters():
                    parameter.requires_grad = False
        self.weak_classifier = new_weak_classifier

    def residual_init(self, old_weak_classifier):
        with torch.no_grad():
            old_num_channels = old_weak_classifier.classifier.in_channels
            curr_num_channels = self.weak_classifier.classifier.in_channels
            if curr_num_channels % old_num_channels == 0:
                times = curr_num_channels // old_num_channels
                old_weight = old_weak_classifier.classifier.linear.weight.data
                old_bias = old_weak_classifier.classifier.linear.bias.data
                old_weight = old_weight.repeat((1, times)) / times
                if hasattr(self.weak_classifier.classifier.linear, 'log_sigma_2'):
                    if hasattr(old_weak_classifier.classifier.linear, 'log_sigma_2'):
                        old_log_sigma_2 = old_weak_classifier.classifier.linear.log_sigma_2.data
                        old_log_sigma_2 = old_log_sigma_2.repeat((1, times)) - torch.log(times)
                        self.weak_classifier.classifier.linear.log_sigma_2.data = old_log_sigma_2
                self.weak_classifier.classifier.linear.weight.data = old_weight
                self.weak_classifier.classifier.linear.bias.data = old_bias
                if times != 1 and isinstance(self.weak_classifier.block, (BasicBlock, Bottleneck)):
                    shortcut_weight = torch.eye(old_num_channels)
                    shortcut_weight = shortcut_weight.repeat((times, 1))
                    shortcut_weight = shortcut_weight + torch.randn_like(shortcut_weight) * 1e-4
                    self.weak_classifier.block.shortcut.weight.data[:, :, 0, 0] = shortcut_weight
                    if isinstance(self.weak_classifier.block, BasicBlock):
                        nn.init.normal_(self.weak_classifier.block.bn_2.weight, std=1e-4)
                        nn.init.constant_(self.weak_classifier.block.bn_2.bias, 0.0)
                    else:
                        nn.init.normal_(self.weak_classifier.block.bn_3.weight, std=1e-4)
                        nn.init.constant_(self.weak_classifier.block.bn_3.bias, 0.0)

    def forward(self, x):
        out = x
        with torch.no_grad():
            for trained_block in self.trained_blocks:
                out = trained_block(out)
        return self.weak_classifier(out)

    def kullback_leibler_divergence(self):
        return self.weak_classifier.kullback_leibler_divergence()


def make_resnet(num_classes=10, depth=110, sparse=False, sequential=False):
    if depth >= 44:
        assert (depth - 2) % 9 == 0, 'Depth should be equal to 9n+2 for some n.'
        n = (depth - 2) // 9
        block = Bottleneck
    else:
        assert (depth - 2) % 6 == 0, 'Depth should be equal to 6n+2 for some n.'
        n = (depth - 2) // 6
        block = BasicBlock

    def _make_blocks(in_channels, out_channels, stride, num_blocks):
        block_layers = [block(in_channels, out_channels, stride=stride, sparse=sparse)]
        for _ in range(num_blocks):
            block_layers.append(block(out_channels, out_channels, sparse=sparse))
        return block_layers

    layers = [conv_3x3(3, 16, sparse=sparse)]
    layers.extend(_make_blocks(16, 64, 1, n))
    layers.extend(_make_blocks(64, 128, 2, n))
    layers.extend(_make_blocks(128, 256, 2, n))
    if sequential:
        weak_classifiers = [WeakClassifier(nn.Sequential(layers[0], layers[1]), num_channels=layers[1].out_channels)]
        for layer in layers[2:]:
            weak_classifiers.append(WeakClassifier(layer))
        return weak_classifiers
    else:
        layers.append(LinearClassifier(layers[-1].out_channels, num_classes, sparse=sparse))
        return nn.Sequential(*layers)
