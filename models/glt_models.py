import torch.nn as nn
import torch.nn.functional as F

from models.svdo_layers import LinearSVDO


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
            self.linear = LinearSVDO(in_channels, num_classes)
        else:
            self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.avg_pool(self.relu(self.bn(x)))
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.linear(out), dim=-1)

    def kl_divergence(self):
        if self.sparse:
            return self.linear.kl_divergence()
        return 0.0


class WeakClassifier(nn.Module):
    def __init__(self, block, num_features=None, num_classes=10, sparse=False):
        super(WeakClassifier, self).__init__()
        self.num_classes = num_classes
        self.sparse = sparse
        if num_features is None:
            self.num_features = block.out_channels
        else:
            self.num_features = num_features

        self.block = block
        self.classifier = LinearClassifier(self.num_features, num_classes, sparse)

    def forward(self, x):
        return self.classifier(self.block(x))

    def kl_divergence(self):
        total_kl = self.classifier.kl_divergence()
        if hasattr(self.block, 'kl_divergence'):
            total_kl = total_kl + self.block.kl_divergence()
        else:
            for module in self.block.children():
                if hasattr(module, 'kl_divergence'):
                    total_kl = total_kl + module.kl_divergence()
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

    def forward(self, x):
        out = x
        for trained_block in self.trained_blocks:
            out = trained_block(out)
        return self.weak_classifier(out)

    def kl_divergence(self):
        return self.weak_classifier.kl_divergence()
