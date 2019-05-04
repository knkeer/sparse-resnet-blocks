import errno
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from sparse_resnet.layers import LinearSVDO, Conv2dSVDO


def get_mnist(batch_size=128, train=True, flatten=True):
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1) if flatten else x)
    ])
    mnist_loader = DataLoader(
        datasets.MNIST('./mnist', train=train, download=True, transform=mnist_transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return mnist_loader


def get_cifar(batch_size=32, train=True):
    cifar_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]
    if train:
        train_transform = [transforms.RandomCrop((32, 32), padding=4), transforms.RandomHorizontalFlip()]
        cifar_transform = train_transform + cifar_transform
    cifar_transform = transforms.Compose(cifar_transform)
    cifar_loader = DataLoader(
        datasets.CIFAR10('./cifar10', train=train, download=True, transform=cifar_transform),
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    return cifar_loader


def parameter_statistics(model):
    total, nonzero = 0, 0
    children = list(model.children())
    if not children:
        if isinstance(model, (LinearSVDO, Conv2dSVDO)):
            total += model.log_alpha.numel()
            nonzero += torch.sum(model.log_alpha <= model.threshold).item()
            if model.bias is not None:
                total += model.bias.numel()
                nonzero += torch.sum(model.bias != 0).item()
        else:
            for parameter in model.parameters():
                total += parameter.data.numel()
                nonzero += torch.sum(parameter.data != 0).item()
    else:
        for module in children:
            module_total, module_nonzero = parameter_statistics(module)
            total += module_total
            nonzero += module_nonzero
    return total, nonzero


def model_compression(model):
    total, nonzero = parameter_statistics(model)
    if nonzero == 0:
        return total
    else:
        return total / nonzero


def validate(model, criterion, dataloader, compute_compression=False,
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.eval()
    # Compute average loss and accuracy
    average_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for samples, targets in dataloader:
            samples, targets = samples.to(device), targets.to(device)
            logits = model(samples)
            average_loss += criterion(logits, targets).item() * len(targets)
            correct += torch.sum(torch.argmax(logits, dim=-1) == targets).item()
    average_loss /= len(dataloader.dataset)
    average_accuracy = correct / len(dataloader.dataset)
    # Compute compression if needed
    if compute_compression:
        compression = model_compression(model)
        return average_loss, average_accuracy, compression
    else:
        return average_loss, average_accuracy


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def checkpoint(name_suffix, model, optimizer, checkpoint_dir):
    # Make directory if it does not exist
    create_dir(checkpoint_dir)
    # Save model and optimizer state dicts
    model_path = os.path.join(checkpoint_dir, 'model_{}.pth'.format(name_suffix))
    optimizer_path = os.path.join(checkpoint_dir, 'optimizer_{}.pth'.format(name_suffix))
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def train_epoch(epoch, model, optimizer, criterion, dataloader,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.to(device)
    model.train()
    average_loss, average_accuracy = 0.0, 0.0
    progress_bar = tqdm(dataloader, total=len(dataloader))
    progress_bar.set_description_str('epoch {}'.format(epoch))
    for samples, targets in progress_bar:
        optimizer.zero_grad()
        samples, targets = samples.to(device), targets.to(device)
        logits = model(samples)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = torch.mean((predictions == targets).float()).item()
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy})
            average_accuracy += accuracy * targets.size(0)
            average_loss += loss.item() * targets.size(0)
    average_loss /= len(dataloader.dataset)
    average_accuracy /= len(dataloader.dataset)
    return average_loss, average_accuracy
