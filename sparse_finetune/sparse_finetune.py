import copy

import torch.nn as nn

from models.glt_models import LinearClassifier
from models.resnet_blocks import BasicBlock, Bottleneck, DownsampleConv2d
from models.svdo_layers import LinearSVDO, Conv2dSVDO


class SequentialSparsifier(nn.Module):
    def __init__(self, pretrained_model):
        super(SequentialSparsifier, self).__init__()
        self.model = nn.ModuleList()
        for module in pretrained_model:
            self.model.append(self.__get_sparse_layer(module))
        self.train_mask = [False for _ in range(len(pretrained_model))]

    @classmethod
    def __get_sparse_layer(cls, dense_layer):
        if isinstance(dense_layer, nn.Linear):
            sparse_layer = LinearSVDO(dense_layer.in_features, dense_layer.out_features,
                                      dense_layer.bias is not None)
            sparse_layer.weight.data = dense_layer.weight.data.clone()
            if dense_layer.bias is not None:
                sparse_layer.bias.data = dense_layer.bias.data.clone()
            return sparse_layer
        elif isinstance(dense_layer, nn.Conv2d):
            sparse_layer = Conv2dSVDO(dense_layer.in_channels, dense_layer.out_channels,
                                      dense_layer.kernel_size, stride=dense_layer.stride,
                                      padding=dense_layer.padding, dilation=dense_layer.dilation,
                                      groups=dense_layer.groups, bias=dense_layer.bias is not None)
            sparse_layer.weight.data = dense_layer.weight.data.clone()
            if dense_layer.bias is not None:
                sparse_layer.bias.data = dense_layer.bias.data.clone()
            return sparse_layer
        elif isinstance(dense_layer, DownsampleConv2d):
            sparse_layer = DownsampleConv2d(dense_layer.in_channels, dense_layer.out_channels,
                                            stride=dense_layer.stride, sparse=True)
            sparse_layer.conv = cls.__get_sparse_layer(dense_layer.conv)
            return sparse_layer
        elif isinstance(dense_layer, BasicBlock):
            sparse_layer = BasicBlock(dense_layer.in_channels, dense_layer.out_channels,
                                      stride=dense_layer.stride, sparse=True)
            sparse_layer.conv_1 = cls.__get_sparse_layer(dense_layer.conv_1)
            sparse_layer.conv_2 = cls.__get_sparse_layer(dense_layer.conv_2)
            if dense_layer.shortcut is not None:
                sparse_layer.shortcut = cls.__get_sparse_layer(dense_layer.shortcut)
            sparse_layer.bn_1 = copy.copy(dense_layer.bn_1)
            sparse_layer.bn_2 = copy.copy(dense_layer.bn_2)
            return sparse_layer
        elif isinstance(dense_layer, Bottleneck):
            sparse_layer = Bottleneck(dense_layer.in_channels, dense_layer.out_channels,
                                      stride=dense_layer.stride, sparse=True)
            sparse_layer.conv_1 = cls.__get_sparse_layer(dense_layer.conv_1)
            sparse_layer.conv_2 = cls.__get_sparse_layer(dense_layer.conv_2)
            sparse_layer.conv_3 = cls.__get_sparse_layer(dense_layer.conv_3)
            if dense_layer.shortcut is not None:
                sparse_layer.shortcut = cls.__get_sparse_layer(dense_layer.shortcut)
            sparse_layer.bn_1 = copy.copy(dense_layer.bn_1)
            sparse_layer.bn_2 = copy.copy(dense_layer.bn_2)
            sparse_layer.bn_3 = copy.copy(dense_layer.bn_3)
            return sparse_layer
        elif isinstance(dense_layer, LinearClassifier):
            sparse_layer = LinearClassifier(dense_layer.in_channels, num_classes=dense_layer.num_classes,
                                            sparse=True)
            sparse_layer.linear = cls.__get_sparse_layer(dense_layer.linear)
            sparse_layer.bn = copy.copy(dense_layer.bn)
            return sparse_layer
        else:
            return copy.copy(dense_layer)

    @classmethod
    def __get_dense_layer(cls, sparse_layer):
        if isinstance(sparse_layer, LinearSVDO):
            dense_layer = nn.Linear(sparse_layer.in_features, sparse_layer.out_features,
                                    sparse_layer.bias is not None)
            dense_layer.weight.data = sparse_layer.weight.data.clone()
            dense_layer.weight.data *= (sparse_layer.log_alpha.data < sparse_layer.threshold).float()
            if sparse_layer.bias is not None:
                dense_layer.bias.data = sparse_layer.bias.data.clone()
            return dense_layer
        elif isinstance(sparse_layer, Conv2dSVDO):
            dense_layer = nn.Conv2d(sparse_layer.in_channels, sparse_layer.out_channels,
                                    sparse_layer.kernel_size, stride=sparse_layer.stride,
                                    padding=sparse_layer.padding, dilation=sparse_layer.dilation,
                                    groups=sparse_layer.groups, bias=sparse_layer.bias is not None)
            dense_layer.weight.data = sparse_layer.weight.data.clone()
            dense_layer.weight.data *= (sparse_layer.log_alpha.data < sparse_layer.threshold).float()
            if sparse_layer.bias is not None:
                dense_layer.bias.data = sparse_layer.bias.data.clone()
            return dense_layer
        elif isinstance(sparse_layer, DownsampleConv2d):
            if not sparse_layer.sparse:
                return copy.copy(sparse_layer)
            dense_layer = DownsampleConv2d(sparse_layer.in_channels, sparse_layer.out_channels,
                                           stride=sparse_layer.stride, sparse=False)
            dense_layer.conv = cls.__get_dense_layer(sparse_layer.conv)
            return dense_layer
        elif isinstance(sparse_layer, BasicBlock):
            if not sparse_layer.sparse:
                return copy.copy(sparse_layer)
            dense_layer = BasicBlock(sparse_layer.in_channels, sparse_layer.out_channels,
                                     stride=sparse_layer.stride, sparse=False)
            dense_layer.conv_1 = cls.__get_dense_layer(sparse_layer.conv_1)
            dense_layer.conv_2 = cls.__get_dense_layer(sparse_layer.conv_2)
            if sparse_layer.shortcut is not None:
                dense_layer.shortcut = cls.__get_dense_layer(sparse_layer.shortcut)
            dense_layer.bn_1 = copy.copy(sparse_layer.bn_1)
            dense_layer.bn_2 = copy.copy(sparse_layer.bn_2)
            return dense_layer
        elif isinstance(sparse_layer, Bottleneck):
            if not sparse_layer.sparse:
                return copy.copy(sparse_layer)
            dense_layer = Bottleneck(sparse_layer.in_channels, sparse_layer.out_channels,
                                     stride=sparse_layer.stride, sparse=False)
            dense_layer.conv_1 = cls.__get_dense_layer(sparse_layer.conv_1)
            dense_layer.conv_2 = cls.__get_dense_layer(sparse_layer.conv_2)
            dense_layer.conv_3 = cls.__get_dense_layer(sparse_layer.conv_3)
            if sparse_layer.shortcut is not None:
                dense_layer.shortcut = cls.__get_dense_layer(sparse_layer.shortcut)
            dense_layer.bn_1 = copy.copy(sparse_layer.bn_1)
            dense_layer.bn_2 = copy.copy(sparse_layer.bn_2)
            dense_layer.bn_3 = copy.copy(sparse_layer.bn_3)
            return dense_layer
        elif isinstance(sparse_layer, LinearClassifier):
            if not sparse_layer.sparse:
                return copy.copy(sparse_layer)
            dense_layer = LinearClassifier(sparse_layer.in_channels, num_classes=sparse_layer.num_classes,
                                           sparse=False)
            dense_layer.linear = cls.__get_dense_layer(sparse_layer.linear)
            dense_layer.bn = copy.copy(sparse_layer.bn)
            return dense_layer
        else:
            return copy.copy(sparse_layer)

    def update_mask(self, new_mask):
        self.train_mask = new_mask

    def set_gradient_flow(self):
        for module, train_flag in zip(self.model, self.train_mask):
            module.train(mode=train_flag)
            for parameter in module.parameters():
                parameter.requires_grad = train_flag

    def finalize_blocks(self, finalize_mask):
        for i in range(len(self.train_mask)):
            if self.train_mask[i]:
                self.model[i] = self.__get_dense_layer(self.model[i])

    def forward(self, x):
        out = x
        for module in self.model:
            out = module(out)
        return out

    def kl_divergence(self):
        total_kl = 0.0
        for module, train_flag in zip(self.model, self.train_mask):
            if train_flag:
                total_kl = total_kl + module.kl_divergence()
        return total_kl
