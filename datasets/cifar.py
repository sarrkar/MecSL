from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

NORMALIZATION = {
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.247, 0.243, 0.261]
}


def __cifar10(train, transform):
    return CIFAR10(root='datasets', train=train, download=True, transform=transform)


def get_cifar10(transform_train=None, transform_test=None, normalize=True):
    if transform_train is None:
        transform_train = Compose([
            ToTensor(),
            Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
        ])
    if transform_test is None:
        transform_test = Compose([
            ToTensor(),
            Normalize(NORMALIZATION["mean"], NORMALIZATION["std"]),
        ])
    return __cifar10(True, transform_train), __cifar10(False, transform_test)
