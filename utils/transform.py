import copy
import random

import torch
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomGrayscale, RandomRotation, ToTensor, Normalize, Compose


class Transform:
    def __init__(self, prob):
        self.prob = prob

    def forward(self, img):
        raise NotImplementedError

    def default(self, *kwargs):
        raise NotImplementedError

    def __call__(self, img, **kwargs):
        if random.random() < self.prob:
            return self.forward(img, **kwargs)
        return img, self.default(**kwargs)


class GrayscaleTransform(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomGrayscale(p=1.0)

    def forward(self, img):
        return self.transform(img), [1.0]

    def default(self):
        return [0.0]


class HorizontalFlipTransform(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomHorizontalFlip(p=1.0)

    def forward(self, img):
        return self.transform(img), [1.0]

    def default(self):
        return [0.0]


class VerticalFlipTransform(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomVerticalFlip(p=1.0)

    def forward(self, img):
        return self.transform(img), [1.0]

    def default(self):
        return [0.0]


class RotationTransform(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transforms = {
            1: RandomRotation(degrees=(90, 90)),
            2: RandomRotation(degrees=(180, 180)),
            3: RandomRotation(degrees=(270, 270)),
        }

    def forward(self, img):
        degree = random.randint(1, 3)
        return self.transforms[degree](img), [float(degree) / 4]

    def default(self):
        return [0.0]


class ColorJitterTransform(Transform):
    def __init__(self, prob):
        super().__init__(prob)

    def forward(self, img, **kwargs):
        brightness = kwargs.get('brightness', 0.8)
        brightness = random.uniform(max(0, 1 - brightness), 1 + brightness)
        contrast = kwargs.get('contrast', 0.8)
        contrast = random.uniform(max(0, 1 - contrast), 1 + contrast)
        saturation = kwargs.get('saturation', 0.8)
        saturation = random.uniform(max(0, 1 - saturation), 1 + saturation)
        hue = kwargs.get('hue', 0.2)
        hue = random.uniform(-hue, hue)
        transform = ColorJitter(
            brightness=(brightness, brightness),
            contrast=(contrast, contrast),
            saturation=(saturation, saturation),
            hue=(hue, hue)
        )
        return transform(img), [brightness, contrast, saturation, hue]

    def default(self, **kwargs):
        return [1.0, 1.0, 1.0, 0.0]


class Transformer:
    def __init__(
            self,
            color_jitter_prob=0.8,
            color_jitter_brightness=0.8,
            color_jitter_contrast=0.8,
            color_jitter_saturation=0.8,
            color_jitter_hue=0.2,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.0,
            rotation_prob=0.0,
            grayscale_prob=0.2,
            normalize=None,
    ):
        self.transforms = {
            'color_jitter': ColorJitterTransform(color_jitter_prob),  # 4
            'horizontal_flip': HorizontalFlipTransform(horizontal_flip_prob), # 1
            'vertical_flip': VerticalFlipTransform(vertical_flip_prob),  # 1
            'rotation': RotationTransform(rotation_prob),  # 1
            'grayscale': GrayscaleTransform(grayscale_prob),  # 1
        }
        if normalize is None:
            self.tensor_transform = ToTensor()
        else:
            self.tensor_transform = Compose([
                ToTensor(),
                Normalize(normalize["mean"], normalize["std"]),
            ])
        self.cj_kwargs = {
            'brightness': color_jitter_brightness,
            'contrast': color_jitter_contrast,
            'saturation': color_jitter_saturation,
            'hue': color_jitter_hue,
        }

    def __call__(self, img):
        orig = copy.deepcopy(img)
        embeddings = []
        for transform in self.transforms:
            if transform == 'color_jitter':
                img, emb = self.transforms[transform](img, **self.cj_kwargs)
            else:
                img, emb = self.transforms[transform](img)
            embeddings.extend(emb)
        orig = self.tensor_transform(orig) 
        img = self.tensor_transform(img)
        embeddings = torch.FloatTensor(embeddings)
        return orig, img, embeddings
