import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from Style_transfer import utils,transformer
import os
from torchvision import transforms,datasets

def stylize_imagenet(style_paths, data_dir, save_folders, batch_size=128, PRESERVE_COLOR=False):
    """Stylizes images in a folder by batch
    If the images  are of different dimensions, use transform.resize() or use a batch size of 1
    IMPORTANT: Put content_folder inside another folder folder_containing_the_content_folder

    folder_containing_the_content_folder
        content_folder
            pic1.ext
            pic2.ext
            pic3.ext
            ...

    and saves as the styled images in save_folder as follow:

    save_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((640,640),antialias=True),])
                                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])])  # pytorch mean and std of imagenet
    transform_test = transforms.Compose([transforms.ToTensor(),transforms.Resize((640,640),antialias=True),])
                                         #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #training_data = datasets.ImageFolder(data_dir, transform=transform)
    #loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, num_workers=16)

    val_data = datasets.ImageFolder(data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=16)

    #test_data = datasets.ImageFolder(data_dir, transform=transform)
    #loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=16)

    # Load Transformer Network
    net0 = transformer.TransformerNetwork()
    net0.load_state_dict(torch.load(style_paths[0],map_location=device))
    net0.eval()

    net1 = transformer.TransformerNetwork()
    net1.load_state_dict(torch.load(style_paths[1], map_location=device))
    net1.eval()

    net2 = transformer.TransformerNetwork()
    net2.load_state_dict(torch.load(style_paths[2], map_location=device))
    net2.eval()

    # Stylize batches of images
    with torch.no_grad():
        curr_batch=0
        curr_style=0
        for idx,(images, labels) in enumerate(tqdm(loader)):
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Generate image
            if curr_style==0:
                generated_tensor = net0(images.to(device)).detach()
                save_folder=save_folders+"mosaic/"
            elif curr_style==1:
                generated_tensor = net1(images.to(device)).detach()
                save_folder = save_folders + "starry/"
            else:
                generated_tensor = net2(images.to(device)).detach()
                save_folder = save_folders + "udnie/"
            # Save images
            for i in range(generated_tensor.shape[0]):
                generated_image = utils.ttoi(generated_tensor[i])
                if (PRESERVE_COLOR):
                    content_image = images[i]
                    generated_image = utils.transfer_color(content_image.permute(1,2,0).numpy(), generated_image)
                #pre_image_name = os.path.basename("pre"+str(curr_batch) + str(i) + ".jpg")
                image_name = os.path.basename(str(curr_batch)+str(i)+".jpg")
                #torchvision.utils.save_image(content_image,save_folder + pre_image_name)
                utils.saveimg(generated_image, save_folder + image_name)

            curr_batch+=1
            curr_style+=1

            if curr_style>2:
                curr_style=0