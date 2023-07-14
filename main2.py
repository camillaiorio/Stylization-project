import imageio as imageio
import torch
import pytorch_lightning as pl
import torchvision
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import data_module
from style_diffusion import DiffusionModel_Cond

# Image loader
"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
"""


def train(data_module_class,model,epochs,device,ckpt_path=None):
    data_module_class.setup(stage='fit')
    logger = TensorBoardLogger("tb_logs", name="diffusion_model")
    checkpoint_callback = ModelCheckpoint(
        filename='DIFFUSION-{epoch}-{Validation Loss:.4f}',
        save_top_k=2,
        monitor='Validation Loss',
        mode='min')
    trainer = pl.Trainer(max_epochs=epochs, logger=logger, default_root_dir="/tb_best",
                         callbacks=[checkpoint_callback],
                         accelerator=device, log_every_n_steps=len(data_module_class.train_dataloader()))
    trainer.fit(model, data_module_class.train_dataloader(), data_module_class.validation_dataloader())
     #ckpt_path = ckpt_path) # resume training
    trainer.save_checkpoint("tb_logs/diffusion_model/last.ckpt")
    data_module_class.setup(stage='test')
    trainer.test(dataloaders=data_module_class.test_dataloader(), ckpt_path='best')




def test(data_module_class,model,ckpt_path,device):
    logger = TensorBoardLogger("tb_logs", name="diffusion_model")
    checkpoint_callback = ModelCheckpoint(
        filename='DIFFUSION-{epoch}-{Validation Loss:.4f}',
        save_top_k=3,
        monitor='Validation Loss',
        mode='min')
    trainer = pl.Trainer(max_epochs=1, logger=logger, default_root_dir="/tb_best",
                         callbacks=[checkpoint_callback],
                         accelerator=device, log_every_n_steps=len(data_module_class.train_dataloader()))
    trainer.fit(model, data_module_class.train_dataloader(), data_module_class.validation_dataloader(),
                ckpt_path = ckpt_path) # resume training
    data_module_class.setup(stage='test')
    trainer.test(dataloaders=data_module_class.test_dataloader(), ckpt_path='best')


def generate_image(n_images,label,num_classes,ckpt_path,device,unet="pro"):
    model = DiffusionModel_Cond(device, num_classes,unet_chosen=unet)
    #model = model.load_from_checkpoint(ckpt_path,map_location=device)
    noisy_imgs, denoised_imgs = model.denoise_sample(n_images,label) # e.g. label=torch.as_tensor([1, 0])
    epoch="34"
    styles={0:"mosaic",
            1:"starry",
            2:"udnie"}
    for i in range(len(noisy_imgs)):
        plt.imshow(noisy_imgs[i].cpu().permute(1,2,0))
        plt.show()
        plt.imshow(denoised_imgs[i].cpu().permute(1,2,0))
        plt.show()
        #plt.savefig('generated/'+styles[i]+epoch+'.png')
        torchvision.utils.save_image(denoised_imgs[i], 'generated/'+styles[i]+epoch+'.png')



def generate_gif(batch_size,classes,device,unet="pro"):
    model = DiffusionModel_Cond(device,num_classes=classes,unet_chosen=unet)
    gif_shape = [3, 3]
    sample_batch_size = gif_shape[0] * gif_shape[1]
    n_hold_final = 10

    ckpt_path = "tb_logs/diffusion_model/version_0/checkpoints/DIFFUSION-epoch=22-Validation Loss=0.050840865820646286.ckpt"
    model = model.load_from_checkpoint(ckpt_path)

    # Generate samples from denoising process
    gen_samples = []
    desired_label = torch.ones(
        batch_size)  # change with torch.zeros(batch_size) for generating anomal images, don't expect good results on anomalies!
    x = torch.randn((sample_batch_size, 3, 128, 128))
    sample_steps = torch.arange(model.T - 1, 0, -1)
    for t in sample_steps:
        x = model.denoise_sample_old(x, t, desired_label)
        if t % 50 == 0:
            gen_samples.append(x)
    for _ in range(n_hold_final):
        gen_samples.append(x)
    gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
    gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

    # Process samples and save as gif
    gen_samples = (gen_samples * 255).type(torch.uint8)
    gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], 32, 32, 1)
    gen_samples = stack_samples(gen_samples, 2)
    gen_samples = stack_samples(gen_samples, 2)

    imageio.mimsave(
        "gifs/pred.gif",
        list(gen_samples),
        fps=5)

    print("Gif saved")


def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)



def start_training():
    pl.seed_everything(56)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "tb_logs/diffusion_model/version_0/checkpoints/"
    batch_size = 64
    num_classes = 3
    epochs=2
    #unet="original"
    unet="Pro"
    print("You chose Diffusion")
    torch.set_float32_matmul_precision("high")
    data_module_class=data_module("./stylized_resized/",batch_size = batch_size)
    model = DiffusionModel_Cond(device,num_classes,unet)
    #compiled_model = torch.compile(model,mode="max-autotune")
    train(data_module_class, model, epochs, device,ckpt_path)
    #train(data_module_class,compiled_model,epochs,device=device)
    #test(data_module_class,model,epochs,device=device)
    #generate_image(3, torch.as_tensor([0, 1, 2]), 3, ckpt_path, device, unet)