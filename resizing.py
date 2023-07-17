import os
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from tqdm import tqdm


if __name__ == "__main__":
    batch_size=16
    data_dir="stylized_data/train/"
    #data_dir = "stylized_data/val/"
    #data_dir = "stylized_data/test/"
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128),antialias=True)])
    my_dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    resize_dir="stylized_resized/train/"
    #resize_dir = "stylized_resized/val/"
    #resize_dir = "stylized_resized/test/"
    for idx, (images,labels) in enumerate(tqdm(loader)):
        for i in range(images.shape[0]):
            image_name = os.path.basename(str(idx) + str(i) + ".jpg")
            if labels[i]==0:
                torchvision.utils.save_image(images[i],resize_dir+ "mosaic/" + image_name)
            elif labels[i]==1:
                torchvision.utils.save_image(images[i],resize_dir+ "starry/" + image_name)
            else:
                torchvision.utils.save_image(images[i],resize_dir+ "udnie/"+ image_name)