from mecsl.model.resnet import ResNet50
from mecsl.model.mlp import LinearClassifier
from mecsl.datasets.cifar import get_cifar10
from mecsl.evaluation.top1 import Top1

import numpy as np

from tqdm.auto import tqdm, trange

from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SupervisedClassifier(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone=ResNet50,
            classifier=LinearClassifier,
    ):
        super(SupervisedClassifier, self).__init__()
        self.backbone = backbone()
        self.classifier = classifier(self.backbone.output_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class SupervisedTrainer:
    def __init__(
            self,
            backbone=ResNet50,
            dataset=get_cifar10,
            classifier=LinearClassifier,
            evaluation=Top1,
            epochs=30,
            lr=0.01,
            batch_size=128,
            device='cuda'
    ):
        self.epochs = epochs
        self.device = torch.device(device)
        self.dataset_train, self.test_dataset = dataset(ToTensor())
        self.train_dataloader = DataLoader(
            self.dataset_train, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)
        self.classifier = SupervisedClassifier(
            len(self.dataset_train.classes),
            backbone,
            classifier
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=250, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluation = evaluation()

    def train(self):
        self.classifier.train()
        for epoch in trange(self.epochs):
            losses = []
            pbar = tqdm(self.train_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.classifier(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                losses.append(loss.item())
                pbar.set_description(
                    f'EPOCH {epoch + 1} - LOSS {np.mean(losses):.4f}')
        return self.classifier

    def test(self):
        self.classifier.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.classifier(x)
                self.evaluation.add(
                    y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
        return self.evaluation.evaluate()
