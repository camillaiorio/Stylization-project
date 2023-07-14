import os
import time

import matplotlib.pyplot as plt
import natsort as natsort
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms,datasets
from tqdm import tqdm

from Style_transfer import utils


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        all_imgs = os.listdir(image_paths)
        print(len(all_imgs))
        self.total_imgs = natsort.natsorted(all_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.image_paths, self.total_imgs[index])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    batch_size=16
    a=time.time()
    data_dir="stylized_data/train/"
    #data_dir = "stylized_data/val/"
    #data_dir = "stylized_data/test/"
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128),antialias=True)])  # pytorch mean and std of imagenet
    #my_dataset = MyDataset(data_dir, transform=transform)
    my_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    print(time.time()-a)
    resize_dir="stylized_resized/train/"
    #resize_dir = "stylized_resized/val/"
    #resize_dir = "stylized_resized/test/"
    for idx, (images,labels) in enumerate(tqdm(train_loader)):
        for i in range(images.shape[0]):
            #generated_image = utils.ttoi(images[i])
            image_name = os.path.basename(str(idx) + str(i) + ".jpg")
            if labels[i]==0:
                #utils.saveimg(generated_image, resize_dir+ "mosaic/" + image_name)
                torchvision.utils.save_image(images[i],resize_dir+ "mosaic/" + image_name)
            elif labels[i]==1:
                #utils.saveimg(generated_image, resize_dir+ "starry/" + image_name)
                torchvision.utils.save_image(images[i],resize_dir+ "starry/" + image_name)
            else:
                #utils.saveimg(generated_image, resize_dir+ "udnie/" + image_name)
                torchvision.utils.save_image(images[i],resize_dir+ "udnie/"+ image_name)

    #test(batch_size)