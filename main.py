from argparse import ArgumentParser
import torch
from main2 import start_training
from stylize_imagenet import stylize_imagenet



if __name__=="__main__":
    STYLE_TRANSFORM_PATH1 = "Style_transfer/transforms/mosaic.pth"
    STYLE_TRANSFORM_PATH2 = "Style_transfer/transforms/starry.pth"
    STYLE_TRANSFORM_PATH3 = "Style_transfer/transforms/udnie.pth"
    PRESERVE_COLOR = False
    styles=[STYLE_TRANSFORM_PATH1,STYLE_TRANSFORM_PATH2,STYLE_TRANSFORM_PATH3]
    torch.manual_seed(56)
    parser = ArgumentParser(description='')
    parser.add_argument('--model', type=str, help="Type \"Stylize\" to stylize images, otherwise \"Diffusion\""
                                                  "to start the actual training of the model.",
                        default="Wrong pick")
    args = parser.parse_args()
    if args.model == "Stylize":
        stylize_imagenet(styles, "coco_data/train2017", "stylized_data/train/", 64, PRESERVE_COLOR)
    elif args.model == "Diffusion":
        start_training()
    else:
        print("Choose a valid option")