import copy
import random

from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomGrayscale, RandomRotation, ToTensor, Normalize, Compose

from mecsl.utils.transform import Transform


class GrayscaleTransformText(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomGrayscale(p=1.0)

    def forward(self, img):
        return self.transform(img), 'grayscale'

    def default(self):
        return 'rgb'


class HorizontalFlipTransformText(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomHorizontalFlip(p=1.0)

    def forward(self, img):
        return self.transform(img), 'horizontal flip'

    def default(self):
        return ''


class VerticalFlipTransformText(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transform = RandomVerticalFlip(p=1.0)

    def forward(self, img):
        return self.transform(img), 'vertical flip'

    def default(self):
        return ''


class RotationTransformText(Transform):
    def __init__(self, prob):
        super().__init__(prob)
        self.transforms = {
            1: RandomRotation(degrees=(90, 90)),
            2: RandomRotation(degrees=(180, 180)),
            3: RandomRotation(degrees=(270, 270)),
        }

    def forward(self, img):
        degree = random.randint(1, 3)
        return self.transforms[degree](img), f'rotate {90 * degree} degrees'

    def default(self):
        return ''


class ColorJitterTransformText(Transform):
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
        return transform(img), f'brightness {brightness:.3f}, contrast {contrast:.3f}, saturation {saturation:.3f}, hue {hue:.3f}'

    def default(self, **kwargs):
        return f'brightness 1.0, contrast 1.0, saturation 1.0, hue 0.0'


class TransformerText:
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
            'color_jitter': ColorJitterTransformText(color_jitter_prob),
            'horizontal_flip': HorizontalFlipTransformText(horizontal_flip_prob),
            'vertical_flip': VerticalFlipTransformText(vertical_flip_prob),
            'rotation': RotationTransformText(rotation_prob),
            'grayscale': GrayscaleTransformText(grayscale_prob),
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
        embeddings = 'image'
        for transform in self.transforms:
            if transform == 'color_jitter':
                img, emb = self.transforms[transform](img, **self.cj_kwargs)
            else:
                img, emb = self.transforms[transform](img)
            embeddings = embeddings + ', ' + emb
        orig = self.tensor_transform(orig) 
        img = self.tensor_transform(img)
        return orig, img, embeddings
