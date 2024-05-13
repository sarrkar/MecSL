from torch import nn

from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Identity()
        self.output_size = 2048

    def forward(self, x):
        return self.resnet(x)
