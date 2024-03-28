from model.resnet import ResNet34
from model.mlp import LinearClassifier
from datasets.cifar import get_cifar10
from evaluation.top1 import Top1

import numpy as np

from tqdm.auto import tqdm, trange

from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads


class SimCLRFeatureExtractor(nn.Module):
    def __init__(
            self,
            backbone=ResNet34,
    ):
        super(SimCLRFeatureExtractor, self).__init__()
        self.backbone = backbone()
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=self.backbone.output_size,
            hidden_dim=512,
            output_dim=128,
        )
        self.output_size = 128

    def forward(self, x):
        features = self.backbone(x)
        logits = self.projection_head(features)
        return logits


class SimCLRTrainer:
    def __init__(
            self,
            backbone=ResNet34,
            dataset=get_cifar10,
            classifier=LinearClassifier,
            evaluation=Top1,
            unsupervised_epochs=10,
            supervised_epochs=30,
            unsupervised_lr=0.01,
            supervised_lr=0.01,
            batch_size=128,
            device='cuda'
    ):
        self.unsupervised_epochs = unsupervised_epochs
        self.supervised_epochs = supervised_epochs
        self.device = torch.device(device)

        transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)
        self.unsupervised_dataset_train, self.unsupervised_test_dataset = dataset(
            transform)
        self.unsupervised_train_dataloader = DataLoader(
            self.unsupervised_dataset_train, batch_size=batch_size, shuffle=True)
        self.supervised_dataset_train, self.supervised_test_dataset = dataset()
        self.supervised_train_dataloader = DataLoader(
            self.supervised_dataset_train, batch_size=batch_size, shuffle=True)
        self.supervised_test_dataloader = DataLoader(
            self.supervised_test_dataset, batch_size=batch_size, shuffle=False)

        self.feature_extractor = SimCLRFeatureExtractor(
            backbone).to(self.device)
        self.unsupervised_optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=unsupervised_lr, weight_decay=1e-6)
        self.unsupervised_scheduler = ReduceLROnPlateau(
            self.unsupervised_optimizer, 'min', patience=250, factor=0.5)
        self.unsupervised_criterion = loss.NTXentLoss(temperature=0.5)

        self.classifier = classifier(
            self.feature_extractor.backbone.output_size,
            len(self.supervised_dataset_train.classes)
        ).to(self.device)
        self.supervised_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=supervised_lr)
        self.supervised_scheduler = ReduceLROnPlateau(
            self.supervised_optimizer, 'min', patience=250, factor=0.5)
        self.supervised_criterion = torch.nn.CrossEntropyLoss()

        self.evaluation = evaluation()

    def train(self):
        self.feature_extractor.train()
        for epoch in trange(self.unsupervised_epochs):
            losses = []
            pbar = tqdm(self.unsupervised_train_dataloader)
            for (x1, x2), _ in pbar:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1 = self.feature_extractor(x1)
                z2 = self.feature_extractor(x1)
                loss = self.unsupervised_criterion(z1, z2)
                self.unsupervised_optimizer.zero_grad()
                loss.backward()
                self.unsupervised_optimizer.step()
                self.unsupervised_scheduler.step(loss)
                losses.append(loss.item())
                pbar.set_description(
                    f'EPOCH {epoch + 1} - LOSS {np.mean(losses):.4f}')

        self.feature_extractor = self.feature_extractor.backbone
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier.train()

        for epoch in trange(self.supervised_epochs):
            losses = []
            pbar = tqdm(self.supervised_train_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                features = self.feature_extractor(x)
                logits = self.classifier(features)
                loss = self.supervised_criterion(logits, y)
                self.supervised_optimizer.zero_grad()
                loss.backward()
                self.supervised_optimizer.step()
                self.supervised_scheduler.step(loss)
                losses.append(loss.item())
                pbar.set_description(
                    f'EPOCH {epoch + 1} - LOSS {np.mean(losses):.4f}')

        return self.feature_extractor

    def test(self):
        self.classifier.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_dataloader)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                features = self.feature_extractor(x)
                logits = self.classifier(features)
                self.evaluation.add(
                    y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
                pbar.set_description(f'EPOCH {self.epochs} - Testing ...')
        return self.evaluation.evaluate()
