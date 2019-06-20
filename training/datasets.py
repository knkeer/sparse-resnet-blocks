from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist(mnist_path='./mnist', batch_size=128, num_workers=4, download=True, flatten=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1) if flatten else x)
    ])
    mnist_train = DataLoader(
        datasets.MNIST(mnist_path, download=download, train=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    mnist_test = DataLoader(
        datasets.MNIST(mnist_path, download=download, train=False, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return {'train': mnist_train, 'test': mnist_test}


def get_cifar(cifar_path='./cifar10', batch_size=64, num_workers=4, download=True):
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
        datasets.CIFAR10(cifar_path, download=download, train=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    cifar_test = DataLoader(
        datasets.CIFAR10(cifar_path, download=download, train=False, transform=test_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return {'train': cifar_train, 'test': cifar_test}
