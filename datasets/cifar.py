from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def __cifar10(train, transform):
    return CIFAR10(root='datasets', train=train, download=True, transform=transform)


def get_cifar10(transform_train=ToTensor(), transform_test=ToTensor()):
    return __cifar10(True, transform_train), __cifar10(False, transform_test)
