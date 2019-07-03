import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

from torch import cat
import numpy as np


class UAE_noFC_AfterNoise(nn.Module):
    
    def block(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def block_one(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel, stride, padding),
            nn.ReLU()
        )
    
    def conv(self, ch_in, ch_out, kernel, stride=1, padding=1):
        return nn.Conv2d(ch_in, ch_out, kernel, stride, padding)
        
    def __init__(self, ch_in, repeat_num, hidden_num=128):
        super(UAE_noFC_AfterNoise, self).__init__()
        self.hidden_num = hidden_num
        self.repeat_num = repeat_num
        
        self.block_1 = self.block_one(ch_in, self.hidden_num, 3, 1)
        
        self.block1 = self.block(self.hidden_num, 128, 3, 1)
        self.block2 = self.block(128, 256, 3, 1)
        self.block3 = self.block(256, 512, 3, 1)
        self.block4 = self.block(512, 1024, 3, 1)
            
        self.block_one1 = self.block_one(128, 128, 3, 2)
        self.block_one2 = self.block_one(256, 256, 3, 2)
        self.block_one3 = self.block_one(512, 512, 3, 2)
        
        self.block5 = self.block(2048, 512, 3, 1)
        self.block6 = self.block(1024, 256, 3, 1)
        self.block7 = self.block(256, 128, 3, 1)
        self.block8 = self.block(128, 128, 3, 1)
        
        self.conv_last = self.conv(128, 3, 3, 1)
        
        self.upscale = nn.Upsample(scale_factor=2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        encoder_layer_list = []
        
        x = self.block_1(x)
        
        # 1st encoding layer
        x = self.block1(x)
        encoder_layer_list.append(x)
        x = self.block_one1(x)
        # 2nd encoding layer
        x = self.block2(x)
        encoder_layer_list.append(x)
        x = self.block_one2(x)
        # 3rd encoding layer
        x = self.block3(x)
        encoder_layer_list.append(x)
        x = self.block_one3(x)
        # 4th encoding layer
        x = self.block4(x)
        encoder_layer_list.append(x)
        
        # 1st decoding layer
        x = torch.cat([x, encoder_layer_list[-1]], dim=1)
        x = self.block5(x)
        x = self.upscale(x)
        # 2nd decoding layer
        x = torch.cat([x, encoder_layer_list[-2]], dim=1)
        x = self.block6(x)
        x = self.upscale(x)
        # 3rd decoding layer
        #x = torch.cat([x, encoder_layer_list[-3]], dim=1)
        x = self.block7(x)
        x = self.upscale(x)
        # 4th decoding layer
        #x = torch.cat([x, encoder_layer_list[-4]], dim=1)
        x = self.block8(x)
        
        output = self.conv_last(x)
        output = self.sigmoid(output)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3*2, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block8 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True).cuda()
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
