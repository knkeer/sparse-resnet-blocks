import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils


class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, bias=True, threshold=3.0):
        super(LinearSVDO, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.log_sigma_2 = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.log_sigma_2, -10.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.training:
            self.log_alpha = self.log_sigma_2 - torch.log(self.weight ** 2 + 1e-8)
            self.log_alpha = torch.clamp(self.log_alpha, -10, 10)
            lrt_mean = F.linear(x, self.weight, self.bias)
            lrt_sigma = torch.sqrt(F.linear(x ** 2, torch.exp(self.log_sigma_2)) + 1e-8)
            return lrt_mean + torch.randn_like(lrt_mean) * lrt_sigma
        mask = (self.log_alpha < self.threshold).float()
        return F.linear(x, self.weight * mask, self.bias)

    def kullback_leibler_divergence(self):
        k_1, k_2, k_3 = 0.63576, 1.8732, 1.48695
        negative_kl = k_1 * torch.sigmoid(k_2 + k_3 * self.log_alpha)
        negative_kl = negative_kl - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -torch.sum(negative_kl - k_1)

    def extra_repr(self):
        output = ['in_features={}'.format(self.in_features)]
        output.append('out_features={}'.format(self.out_features))
        if self.bias is None:
            output.append('bias=False')
        output.append('threshold={}'.format(self.threshold))
        return ', '.join(output)


class Conv2dSVDO(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, threshold=3.0):
        super(Conv2dSVDO, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = utils._pair(kernel_size)
        self.stride = utils._pair(stride)
        self.padding = utils._pair(padding)
        self.dilation = utils._pair(dilation)
        self.groups = groups
        self.threshold = threshold

        weight_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(*weight_shape))
        self.log_sigma_2 = nn.Parameter(torch.Tensor(*weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.log_sigma_2, -10.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.training:
            self.log_alpha = self.log_sigma_2 - torch.log(self.weight ** 2 + 1e-8)
            self.log_alpha = torch.clamp(self.log_alpha, -10, 10)
            lrt_mean = F.conv2d(x, self.weight, bias=self.bias,
                                stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)
            lrt_sigma = F.conv2d(x ** 2, torch.exp(self.log_sigma_2),
                                 stride=self.stride, padding=self.padding,
                                 dilation=self.dilation, groups=self.groups)
            lrt_sigma = torch.sqrt(lrt_sigma + 1e-8)
            return lrt_mean + torch.randn_like(lrt_mean) * lrt_sigma
        mask = (self.log_alpha < self.threshold).float()
        return F.conv2d(x, self.weight * mask, bias=self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)

    def kullback_leibler_divergence(self):
        k_1, k_2, k_3 = 0.63576, 1.8732, 1.48695
        negative_kl = k_1 * torch.sigmoid(k_2 + k_3 * self.log_alpha)
        negative_kl = negative_kl - 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return -torch.sum(negative_kl - k_1)

    def extra_repr(self):
        output = ['in_channels={}'.format(self.in_channels)]
        output.append('out_channels={}'.format(self.out_channels))
        output.append('kernel_size={}'.format(self.kernel_size))
        if self.stride != (1,) * len(self.stride):
            output.append('stride={}'.format(self.stride))
        if self.padding != (0,) * len(self.padding):
            output.append('padding={}'.format(self.padding))
        if self.dilation != (1,) * len(self.dilation):
            output.append('dilation={}'.format(self.dilation))
        if self.groups != 1:
            output.append('groups={}'.format(self.groups))
        if self.bias is None:
            output.append('bias=False')
        output.append('threshold={}'.format(self.threshold))
        return ', '.join(output)


class SGVLB(nn.Module):
    def __init__(self, model, dataset_size):
        super(SGVLB, self).__init__()
        self.model = model
        self.dataset_size = dataset_size
        self.beta = 0.0

    def kullback_leibler_divergence(self):
        if hasattr(self.model, 'kullback_leibler_divergence'):
            return self.model.kullback_leibler_divergence()
        total_kl = 0.0
        for child in self.model.children():
            if hasattr(child, 'kullback_leibler_divergence'):
                total_kl = total_kl + child.kullback_leibler_divergence()
        return total_kl

    def update_beta(self, step=0.02):
        self.beta = min(self.beta + step, 1)

    def forward(self, logits, targets):
        kl = self.kullback_leibler_divergence()
        return self.dataset_size * F.nll_loss(logits, targets) + self.beta * kl
