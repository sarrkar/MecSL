from torchvision.datasets import CIFAR10


def __cifar10(train, transform):
    return CIFAR10(root='datasets', train=train, download=True, transform=transform)


def get_cifar10(transform):
    return __cifar10(True, transform), __cifar10(False, transform)
