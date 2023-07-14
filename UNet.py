import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self, device, num_classes):
        super().__init__()

        self.num_classes = num_classes
        time_dim = 256
        self.device = device
        self.emb_layer = nn.Embedding(self.num_classes, time_dim)
        self.max_pool = nn.MaxPool2d(2)
        self.max_pool2 = nn.MaxPool2d(2,ceil_mode=True)
        self.act = nn.LeakyReLU()

        self.in_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64)
        )

        self.compress_res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64)
        )
        self.compress1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128)

        )

        self.time_net1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )

        self.compress_res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128)
        )
        self.compress2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256)
        )

        self.time_net2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 256)
        )

        self.compress_res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256)
        )
        self.compress3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256)
        )

        self.attention1 = Multi_head_attention(256)
        self.attention2 = Multi_head_attention(128)

        self.up = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)  # bilinear to lose less data

        self.decompress_res1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 512)
        )
        self.decompress1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128)
        )
        self.decompress_res2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 256)
        )
        self.decompress2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64)
        )
        self.decompress_res3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 128)
        )

        self.time_net4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(256, 64)
        )

        self.decompress3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.GroupNorm(1, 64)
        )
        self.out_conv = nn.Conv2d(64, 3, kernel_size = 1)

    def time_embeddings(self, t, channels):
        # Reference : math inspired by the Paper "Attention is all you need": https://arxiv.org/pdf/1706.03762.pdf
        inv_freq = 1.0 / (10000** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, label):

        t = self.time_embeddings(t, 256)

        if label is not None:
            t += self.emb_layer(label)  # for making the net conditional we ought to add the label somewhere, and t was the choice

        x1 = self.in_conv(x)

        # Down sampling blocks

        # Down 1
        x2 = self.max_pool(x1)
        x2 = self.act(x2 + self.compress_res1(x2))
        x2 = self.compress1(x2) + self.time_net1(t)[:, :, None, None].repeat(1, 1, x2.shape[-2], x2.shape[-1])

        # Down 2
        x3 = self.max_pool2(x2)
        x3 = self.act(x3 + self.compress_res2(x3))
        x3 = self.compress2(x3) + self.time_net2(t)[:, :, None, None].repeat(1, 1, x3.shape[-2], x3.shape[-1])
        x3 = self.attention1(x3)

        # Down 3
        x4 = self.max_pool(x3)
        x4 = self.act(x4 + self.compress_res3(x4))
        x4 = self.compress3(x4) + self.time_net2(t)[:, :, None, None].repeat(1, 1, x4.shape[-2], x4.shape[-1])
        x4 = self.attention1(x4)

        # Up sampling blocks

        # Up 1
        x = self.up(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.act(x + self.decompress_res1(x))
        x = self.decompress1(x) + self.time_net1(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = self.attention2(x)

        # Up 2
        x = self.up(x)
        x = torch.cat([x2, x], dim=1)
        x = self.act(x + self.decompress_res2(x))
        x = self.decompress2(x) + self.time_net4(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        # Up 3
        x = self.up(x)
        x = torch.cat([x1, x], dim=1)
        x = self.act(x + self.decompress_res3(x))
        x = self.decompress3(x) + self.time_net4(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        output = self.out_conv(x)

        return output

class Multi_head_attention(nn.Module):

    def __init__(self, im_size):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(im_size, 8,
                                                          batch_first=True)  # image size must be divisible by n. heads  # alternative 4
        self.layer_norm = nn.LayerNorm(im_size)
        self.linear = nn.Linear(im_size, im_size)
        self.act = nn.LeakyReLU()
        self.image_size = im_size

    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[2] * x.shape[3]).permute(0, 2,
                                                                    1)  # adapt x shape for the multi-head attention nn module
        # x is normalized to have the resulting features of the input image on the same scale such that the n-heads we deploy can very well focus on different
        # aspects of the same image instead of being biased towards those with higher scale.
        x_ln = self.layer_norm(x)
        attention_value, _ = self.multi_head_attention(x_ln, x_ln,
                                                       x_ln)  # for both query, key and value we use x_ln
        x = x + attention_value  # residual connection
        x_ln = self.layer_norm(x)
        x_l = self.act(self.linear(x_ln))
        x = x + self.linear(x_l)  # residual connection
        x = x.permute(0, 2, 1)
        x = x.view(x.shape[0], x.shape[1], int((x.shape[2]) ** 0.5), -1)  # restoring original shape of x
        return x