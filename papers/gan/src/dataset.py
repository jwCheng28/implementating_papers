import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((-0.5,), (-0.5,)),
        ])

    def prepare_data(self):
        datasets.FashionMNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        self.train_dataset = datasets.FashionMNIST(
            self.data_dir, train=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
