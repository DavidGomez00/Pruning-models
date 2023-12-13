import os

import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold



class DogsAndCatsDataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, batch_size:int=32, num_workers:int=0, k:int=1, folds:int=1, split_seed:int=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.k=k
        self.folds=folds
        self.split_seed=split_seed
        assert 0 <= self.k <= self.folds, "Incorrect fold number"

        self.train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def setup(self, stage:str):
        # Single gpu
        full_dataset = ImageFolder(root=os.path.join(self.data_dir, "train"), transform=self.train_transform)
        self.test_data = ImageFolder(root=os.path.join(self.data_dir, "test"), transform=self.test_transform)
        # Choose fold to train on
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(full_dataset)]
        train_indexes, val_indexes = all_splits[self.k]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
        self.train_data, self.val_data = Subset(full_dataset, train_indexes), Subset(full_dataset, val_indexes)

    
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
        
        
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)