import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            in_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample != None:
            identity = self.downsample(x)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out3 = self.relu(self.bn3(self.conv3(out2)) + identity)
        return out3


class ResNet(pl.LightningModule):
    def __init__(self,  block: nn.Module, layers: list[int], num_classes: int = 10, lr: float = 0.1, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels,
                               kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)

    def _make_layer(self, block: nn.Module, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = nn.functional.cross_entropy(preds, y)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self(X)
        loss = nn.functional.cross_entropy(preds, y)
        self.val_acc(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(
        ), lr=self.hparams.lr, momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def resnet50(num_classes: int = 10, lr: float = 0.1, weight_decay: float = 1e-4) -> ResNet:
    # ResNet-50 layer configuration: [3, 4, 6, 3] blocks per stage
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, lr=lr, weight_decay=weight_decay)
