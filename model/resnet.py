from torch import nn

from torchvision.models import resnet34, ResNet34_Weights


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.output_size = 512

    def forward(self, x):
        return self.resnet(x)
