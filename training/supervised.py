from model.resnet import ResNet34
from model.mlp import LinearClassifier
from datasets.cifar import get_cifar10
from evaluation.top1 import Top1

import numpy as np

from tqdm.auto import tqdm, trange

from torchvision.transforms import ToTensor

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SupervisedClassifier:
    def __init__(
            self,
            backbone=ResNet34,
            dataset=get_cifar10,
            classifier=LinearClassifier,
            evaluation=Top1,
            epochs=30,
            lr=0.01,
            device='cuda'
    ):
        self.device = torch.device(device)
        self.backbone = backbone().to(self.device)
        self.dataset_train, self.test_dataset = dataset(ToTensor())
        self.train_dataloader = DataLoader(
            self.dataset_train, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False)
        self.classifier = classifier(
            self.backbone.output_size, 
            len(self.dataset_train.classes)
        ).to(self.device)
        self.epochs = epochs
        params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=250, factor=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluation = evaluation()

    def train(self):
        self.backbone.train()
        self.classifier.train()
        for epoch in trange(self.epochs):
            losses = []
            pbar = tqdm(self.train_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                features = self.backbone(x)
                logits = self.classifier(features)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss)
                losses.append(loss.item())
                pbar.set_description(f'EPOCH {epoch + 1} - LOSS {np.mean(losses):.4f}')

    def test(self):
        self.backbone.eval()
        self.classifier.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                features = self.backbone(x)
                logits = self.classifier(features)
                self.evaluation.add(
                    y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                pbar.set_description(f'EPOCH {self.epochs} - Testing ...')
        return self.evaluation.evaluate()
