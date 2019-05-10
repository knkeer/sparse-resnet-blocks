import os
import torch
import models.sparse_variational_dropout as svdo

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(mnist_path='./mnist', batch_size=128, num_workers=4, flatten=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1) if flatten else x)
    ])
    mnist_train = DataLoader(
        datasets.MNIST(mnist_path, download=True, train=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    mnist_test = DataLoader(
        datasets.MNIST(mnist_path, download=True, train=False, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return {'train': mnist_train, 'test': mnist_test}


def get_cifar(cifar_path='./cifar10', batch_size=64, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar_train = DataLoader(
        datasets.CIFAR10(cifar_path, download=True, train=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    cifar_test = DataLoader(
        datasets.CIFAR10(cifar_path, download=True, train=False, transform=test_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return {'train': cifar_train, 'test': cifar_test}


def train_model(model, optimizer, criterion, dataloader,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.train()
    train_loss, train_accuracy = 0.0, 0
    for samples, targets in dataloader:
        samples, targets = samples.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * targets.size(0)
        train_accuracy += torch.sum(torch.argmax(outputs, dim=-1) == targets).item()
    train_loss /= len(dataloader.dataset)
    train_accuracy /= len(dataloader.dataset)
    return {'loss': train_loss, 'accuracy': train_accuracy * 100.0}


def validate_model(model, criterion, dataloader,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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
    return {'loss': test_loss, 'accuracy': test_accuracy * 100.0}


def parameter_statistics(model, eps=1e-8):
    total, non_zero = 0, 0
    children = list(model.children())
    if not children:
        if isinstance(model, (svdo.LinearSVDO, svdo.Conv2dSVDO)):
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


def compression(model, eps=1e-8):
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
