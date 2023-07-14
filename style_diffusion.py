import math
import pytorch_lightning as pl
import torchvision
from torch.nn import functional as F
import torch
from UNet import UNet
from diffusers import UNet2DModel
#import segmentation_models_pytorch as smp
#from segmentation_models_pytorch.encoders import get_preprocessing_fn

class DiffusionModel_Cond(pl.LightningModule):

    def __init__(self, device_, num_classes, unet_chosen, T=1000):
        super().__init__()
        self.device_ = device_
        self.unet_chosen=unet_chosen
        if unet_chosen == "original":
            print("Original")
            self.unet = UNet(self.device_, num_classes).to(self.device_)
        else:
            print("Pro")
            self.unet = UNet2DModel.from_pretrained(
                #"stabilityai/stable-diffusion-2-1", subfolder="unet", torch_dtype=torch.float32
                "kaizerkam/sd-class-comics-64"
            ).to(self.device_)

            #self.unet = self.unet
            self.unet.save_pretrained("pro_weights")
        #TODO
        """
        preprocess_input = get_preprocessing_fn('resnet50', pretrained='imagenet')
        self.unet = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
        )
        """
        self.lr = 0.001
        self.save_hyperparameters()
        self.T = T  # total number of timesteps
        self.beta = torch.linspace(0.0001, 0.02, T).to(self.device_)  # variance schedule, could be also learned.
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.beta_0 = self.beta  # fixed variance

    def forward(self, x, input_labels):  # loss computation to plot gaussians

        with torch.no_grad():

            batch_size = x.shape[0]
            t = torch.randint(0, self.T, (batch_size,), device=self.device)
            noise = torch.randn_like(x, device=self.device)
            mean = (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1,
                                                         1)) ** 0.5 * x  # gather αt into alpha_bar_t and compute square root of it, all times x0
            variance = 1 - (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1))
            xt = mean + (variance ** 0.5) * noise  # apply q(xt∣x0), i.e. return noised image
            eps_theta = self.unet(xt, t.unsqueeze(1), input_labels)  # predict noise added to image xt at time t
            loss = F.mse_loss(eps_theta, noise)
        return loss.item()

    def training_step(self, batch, batch_nb):

        input_images, input_labels = batch
        batch_size = input_images.shape[0]

        # for each image in the batch one t is generated randomically in the interval [0,T]:
        t = torch.randint(0, self.T, (batch_size,), device=self.device)

        # unless specified the noise is randomically generated:
        noise = torch.randn_like(input_images, device=self.device)

        # addition of randomly generated noise to input image x to obtain noisy image xt at timestep t:

        mean = (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1)) ** 0.5 * input_images  # gather αt into alpha_bar_t and compute square root of it, all times x0
        variance = 1 - (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1))
        xt = mean + (variance ** 0.5) * noise

        # Unet predicts noise added to image xt at time t:
        if self.unet_chosen == "original":
            eps_theta = self.unet(xt, t.unsqueeze(1), input_labels) #TODO works with smp?
        else:
            eps_theta = self.unet(xt, t, input_labels).sample

        # compute loss between predicted noise and true noise (we tried both mse loss and the huber loss)
        loss = F.mse_loss(eps_theta, noise)

        self.log_dict({"Training Loss": loss}, on_step=False, on_epoch=True)
        # self.log_dict({"Train loss anomaly": loss_a,"Train loss normal": loss_n}, on_step = False, on_epoch = True)

        return loss

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):

        input_images, input_labels = batch
        batch_size = input_images.shape[0]

        # for each image in the batch t is generated randomically in the interval [0,T]:
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        noise = torch.randn_like(input_images,
                                 device=self.device)  # unless specified the noise is randomically generated

        # add noise generated to input image
        mean = (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1,
                                                     1)) ** 0.5 * input_images  # gather αt into alpha_bar_t and compute square root of it, all times x0
        variance = 1 - (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1))
        xt = mean + (variance ** 0.5) * noise  # apply q(xt∣x0), i.e. return noised image

        if self.unet_chosen == "original":
            eps_theta = self.unet(xt, t.unsqueeze(1), input_labels)  # TODO works with smp?
        else:
            eps_theta = self.unet(xt, t, input_labels).sample

        loss = F.mse_loss(eps_theta, noise)

        self.log_dict({"Validation Loss": loss}, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        input_images, input_labels = batch
        batch_size = input_images.shape[0]

        # for each image in the batch one t is generated randomically in the interval [0,T]
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        noise = torch.randn_like(input_images,
                                 device=self.device)  # unless specified the noise is randomically generated

        # Add noise generated to input image
        mean = (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1,
                                                     1)) ** 0.5 * input_images  # gather αt into alpha_bar_t and compute square root of it, all times x0
        variance = 1 - (self.alpha_bar.gather(-1, t).reshape(-1, 1, 1, 1))
        xt = mean + (variance ** 0.5) * noise  # apply q(xt∣x0), i.e. return noised image

        if self.unet_chosen == "original":
            eps_theta = self.unet(xt, t.unsqueeze(1), input_labels)  # TODO works with smp?
        else:
            eps_theta = self.unet(xt, t, input_labels).sample  # predict noise added to image xt at time t

        loss = F.mse_loss(eps_theta, noise)

        self.log_dict({"Test Loss": loss}, on_step=False, on_epoch=True)

    def denoise_sample_gif(self, x, t, input_labels):  # eventually learn beta
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(self.device)
            else:
                z = 0
            e_hat = self.unet(x, t.view(1, 1).repeat(x.shape[0], 1), input_labels)
            pre_scale = 1 / math.sqrt(self.alpha[t])
            e_scale = (1 - self.alpha[t]) / math.sqrt(1 - self.alpha_bar[t])
            post_sigma = math.sqrt(self.beta[t]) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def denoise_sample(self, batch_size, label=None):
        """
        #Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            x_T = torch.randn((batch_size, 3, 128, 128), device=self.device)
            x_t = x_T
            sample_steps = torch.arange(self.T - 1, 0, -1)
            for t in sample_steps:
                if t%100==0:
                    print(t)
                if t > 1:
                    z = torch.randn(x_t.shape, device=self.device)
                else:
                    z = 0
                eps_theta_t = self.unet(x_t, t.view(1, 1).repeat(x_t.shape[0], 1).to(self.device), label.to(self.device))  # provare a inserire solo t
                pre_scale = 1 / math.sqrt(self.alpha[t])
                e_scale = (1 - self.alpha[t]) / math.sqrt(1 - self.alpha_bar[t])
                sigma_t = math.sqrt(self.beta_0[t])
                x_t = pre_scale * (x_t - e_scale * eps_theta_t) + sigma_t * z  # x_t = x_t-1
            x_0 = x_t
        return x_T, x_0