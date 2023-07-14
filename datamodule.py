import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import itertools

class data_module(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./stylized_resized/", batch_size=128):  # in the Paper batch = 1

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        #TODO specific data normalization
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])  #pytorch mean and std of imagenet
        '''transforms.RandomHorizontalFlip(),
         transforms.RandomApply(torch.nn.ModuleList([
             transforms.RandomRotation((90, 90)),
             transforms.RandomRotation((180, 180)),
             transforms.RandomRotation((270, 270))]), p=0.25),
         #transforms.RandomErasing(p=0.1, scale=(0.04, 0.05), ratio=(0.3, 3.3), value=0.3, inplace=False),
         transforms.RandomApply(torch.nn.ModuleList([
             transforms.ColorJitter(brightness=(0.4, 1.5)),
             transforms.ColorJitter(contrast=(0.6, 1.7)),
             transforms.ColorJitter(brightness=(0.4, 1.5), contrast=(0.6, 1.7))
         ]), p=0.25)'''

        self.transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    """
    def prepare_data(self):

        datasets.CocoDetection(self.data_dir, train=True, download=True)
        datasets.CocoDetection(self.data_dir, train=False, download=True)
    """

    """
    def balance_dataset(self, dataset):
        datas = []
        targs = []
        classes = 10
        for i in range(classes):
            ci = torch.argwhere(dataset.targets == i)
            fci = dataset.data[ci]
            idx = torch.randperm(len(fci))[:int(len(fci) / (classes - 1)) + 1]  # random selection of images from each class
            fci = fci[idx].squeeze()
            targi = dataset.targets[ci]
            targi = targi[idx].squeeze()
            datas.append(fci)
            targs.append(targi)

        datas = list(itertools.chain(*datas))
        targs = list(itertools.chain(*targs))
        dataset.data = torch.stack(datas)
        dataset.targets = torch.stack(targs)
        return dataset"""

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(self.data_dir+"train/", transform=self.transform)
            self.val_dataset = datasets.ImageFolder(self.data_dir+"val/", transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = datasets.ImageFolder(self.data_dir+"test/", transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size,num_workers=16)  ###CHANGE num_workers

    def validation_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=16)  ##CHANGE num_workers

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=16)