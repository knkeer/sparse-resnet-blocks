import os
import random
import stat
import sys

import numpy as np
import torch
import torch.optim as optim

from models.svdo_layers import LinearSVDO, Conv2dSVDO
from models.resnet_blocks import BasicBlock, Bottleneck
from models.resnet import make_resnet
from training import datasets


def kl_divergence(model):
    if hasattr(model, 'kl_divergence'):
        return model.kl_divergence()
    total_kl = 0.0
    for module in model.children():
        if hasattr(module, 'kl_divergence'):
            total_kl = total_kl + module.kl_divergence()
    return total_kl


def train_model(model, optimizer, criterion, dataloader, device, beta=None, finetune=False):
    model.to(device)
    if finetune:
        model.set_gradient_flow()
    else:
        model.train()
    train_loss, train_accuracy = 0.0, 0
    if beta is not None:
        kl = 0.0
    for samples, targets in dataloader:
        samples, targets = samples.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        train_loss += loss.item() * targets.size(0)
        if beta is not None:
            current_kl = kl_divergence(model)
            try:
                kl += current_kl.item()
            except AttributeError:
                pass
            loss = loss + beta * (current_kl / len(dataloader.dataset))
        loss.backward()
        optimizer.step()
        train_accuracy += torch.sum(torch.argmax(outputs, dim=-1) == targets).item()
    train_loss /= len(dataloader.dataset)
    train_accuracy /= len(dataloader.dataset)
    train_results = {'loss': train_loss, 'accuracy': train_accuracy * 100.0}
    if beta is not None:
        train_results['kl'] = kl / len(dataloader)
    return train_results


def evaluate_model(model, criterion, dataloader, device, beta=None):
    model.to(device)
    model.eval()
    test_loss, test_accuracy = 0.0, 0
    for samples, targets in dataloader:
        samples, targets = samples.to(device), targets.to(device)
        outputs = model(samples)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * targets.size(0)
        test_accuracy += torch.sum(torch.argmax(outputs, dim=-1) == targets).item()
    test_loss /= len(dataloader.dataset)
    test_accuracy /= len(dataloader.dataset)
    train_results = {'loss': test_loss, 'accuracy': test_accuracy * 100.0}
    if beta is not None:
        train_results['kl'] = kl_divergence(model)
    return train_results


def parameter_statistics(model, eps=1e-8):
    total, non_zero = 0, 0
    children = list(model.children())
    if not children:
        if isinstance(model, (LinearSVDO, Conv2dSVDO)):
            total += model.log_alpha.numel()
            non_zero += torch.sum(model.log_alpha <= model.threshold).item()
            if model.bias is not None:
                total += model.bias.numel()
                non_zero += torch.sum(torch.abs(model.bias) >= eps).item()
        else:
            for parameter in model.parameters():
                total += parameter.numel()
                non_zero += torch.sum(torch.abs(parameter) >= eps).item()
    else:
        for child in children:
            child_total, child_nonzero = parameter_statistics(child, eps=eps)
            total += child_total
            non_zero += child_nonzero
    return total, non_zero


def compression_ratio(model, eps=1e-8):
    total, nonzero = parameter_statistics(model, eps=eps)
    if nonzero == 0:
        return float('inf')
    else:
        return total / nonzero


def make_checkpoint(dir_path, epoch, **kwargs):
    state_dict = {'epoch': epoch}
    state_dict.update(kwargs)
    file_path = os.path.join(dir_path, 'checkpoint-{}.pth'.format(epoch))
    torch.save(state_dict, file_path)


def prepare_checkpoint_folder(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'command.sh'), 'w') as f:
        f.write('#!/usr/bin/sh\n\npython {}\n'.format(' '.join(sys.argv)))
    st = os.stat(os.path.join(args.checkpoint_dir, 'command.sh'))
    os.chmod(os.path.join(args.checkpoint_dir, 'command.sh'), st.st_mode | stat.S_IEXEC)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_dataloader(args):
    return datasets.get_cifar(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)


def get_optimizer(model, args):
    if args.adam:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, momentum=0.9)
    return optimizer


def get_resnet_layers(args):
    if args.depth >= 44:
        if (args.depth - 2) % 9 != 0:
            raise ValueError('Network depth should be equal to 9n+2 for some n.')
        num_blocks = (args.depth - 2) // 9
        block = Bottleneck
        channels = (16, 64 * args.width, 128 * args.width, 256 * args.width)
        stride = (1, 2, 2)
    else:
        if (args.depth - 2) % 6 != 0:
            raise ValueError('Network depth should be equal to 6n+2 for some n.')
        num_blocks = (args.depth - 2) // 6
        block = BasicBlock
        channels = (16, 16 * args.width, 32 * args.width, 64 * args.width)
        stride = (1, 2, 2)
    if args.reverse:
        channels = channels[::-1]
        stride = stride[::-1]
    return make_resnet(block, num_blocks=num_blocks, channels=channels,
                       stride=stride, sparse=args.sparse)


def get_columns(args, end_to_end=True):
    if not args.sparse:
        columns = ['epoch', 'lr', 'tr_ce', 'tr_acc', 'te_ce', 'te_acc', 'time']
        fmt = ['%s', '3.1e', '8.6f', '5.2f', '8.6f', '5.2f', '5.1f']
    else:
        columns = ['epoch', 'lr', 'tr_ce', 'tr_acc', 'te_ce', 'te_acc', 'kl', 'beta', 'time']
        fmt = ['%s', '3.1e', '8.6f', '5.2f', '8.6f', '5.2f', '8.1f', '4.2f', '5.1f']
        if end_to_end:
            columns = columns[:-2] + ['sp'] + columns[-2:]
            fmt = fmt[:-2] + ['6.2f'] + fmt[-2:]
        else:
            columns = columns[:-2] + ['blk_sp', 'clf_sp'] + columns[-2:]
            fmt = fmt[:-2] + ['6.2f', '6.2f'] + fmt[-2:]
    if not end_to_end:
        columns = ['block'] + columns
        fmt = ['%s'] + fmt
    return columns, fmt


def get_beta(epoch):
    if epoch < 50:
        return 0.0
    if epoch > 100:
        return 1.0
    else:
        return (epoch - 50) / 50
