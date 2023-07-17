import itertools

import torch
from torchvision import transforms,datasets

def compute_mean_and_std(dataset):
    sum_batch_means, sum_sqr_batch_means, num_batches = 0, 0, 0

    for data, _ in dataset:  # iterate over batches
        # Mean over batch, height and width, but not over the channels
        batch_mean = torch.mean(data, dim=[1, 2])
        sum_batch_means += batch_mean
        sqr_batch_mean = torch.mean(data ** 2, dim=[1, 2])
        sum_sqr_batch_means += sqr_batch_mean
        num_batches += 1
    mean = sum_batch_means / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (sum_sqr_batch_means / num_batches - mean ** 2) ** 0.5

    return mean, std


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop((32, 32), scale=(0.929, 1.0)),
                                transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation((90, 90)),
                                                                            transforms.RandomRotation((180, 180)),
                                                                            transforms.RandomRotation((270, 270))]),
                                                       p=0.25)])


def balance_dataset(dataset, normal):
    datas = []
    targs = []
    classes = 10
    for i in range(classes):

        if i == normal:
            ci = (dataset.targets == normal)
            fci = dataset.data[ci]
            targi = dataset.targets[ci]
        else:
            ci = torch.argwhere(dataset.targets == i)
            fci = dataset.data[ci]
            idx = torch.randperm(len(fci))[
                  :int(len(fci) / (classes - 1)) + 1]  # random selection of images from each class
            fci = fci[idx].squeeze()
            targi = dataset.targets[ci]
            targi = targi[idx].squeeze()
        datas.append(fci)
        targs.append(targi)

    datas = list(itertools.chain(*datas))
    targs = list(itertools.chain(*targs))
    dataset.data = torch.stack(datas)
    dataset.targets = torch.stack(targs)
    nr = torch.argwhere(dataset.targets == normal)
    dataset.targets[:] = 0
    dataset.targets[nr] = 1

    return dataset


dataset = datasets.ImageNet("./", train=True, download=True, transform=transform)
normal = 1
idx = torch.randperm(len(dataset))
train_idx = idx[:55000]

mnist_train = torch.utils.data.Subset(dataset, train_idx)

mnist_train = balance_dataset(mnist_train.dataset, normal)

mean, std = compute_mean_and_std(mnist_train)

print(dataset.data.shape)

print("Mean:", mean, "std:", std)