import torch.nn as nn
import torch
import torchvision.transforms.functional as F

class DoubleConvLayer(nn.Module):
    """
    Encoder Double Convolution Layer with Batchnorm
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.sequential(x)   
    
    
class UNet(nn.Module):
    """
    U-Net Implementation according to paper: https://arxiv.org/pdf/1505.04597.pdf
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        # Encoder Modules
        self.in_channels = in_channels
        self.double_conv1 = DoubleConvLayer(in_channels=in_channels, out_channels=64)
        self.double_conv2 = DoubleConvLayer(in_channels=64, out_channels=128)
        self.double_conv3 = DoubleConvLayer(in_channels=128, out_channels=256)
        self.double_conv4 = DoubleConvLayer(in_channels=256, out_channels=512)
        self.double_conv5 = DoubleConvLayer(in_channels=512, out_channels=1024)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Decoder Modules
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)
        self.double_conv6 = DoubleConvLayer(in_channels=1024, out_channels=512)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)
        self.double_conv7 = DoubleConvLayer(in_channels=512, out_channels=256)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.double_conv8 = DoubleConvLayer(in_channels=256, out_channels=128)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        self.double_conv9 = DoubleConvLayer(in_channels=128, out_channels=64)
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        
    def forward(self, x):
        # Encoder Part
        x = self.double_conv1(x)
        identity1 = x
        x = self.max_pool(x)
        x = self.double_conv2(x)
        identity2 = x
        x = self.max_pool(x)
        x = self.double_conv3(x)
        identity3 = x
        x = self.max_pool(x)
        x = self.double_conv4(x)
        identity4 = x
        x = self.max_pool(x)
        x = self.double_conv5(x)
                
        # Decoder Part
        x = self.deconv1(x)
        identity4 = F.resize(identity4, size=x.shape[2:])
        x = torch.cat((x, identity4), dim=1)
        x = self.double_conv6(x)
        
        x = self.deconv2(x)
        identity3 = F.resize(identity3, size=x.shape[2:])
        x = torch.cat((x, identity3), dim=1)
        x = self.double_conv7(x)
        
        x = self.deconv3(x)
        identity2 = F.resize(identity2, size=x.shape[2:])
        x = torch.cat((x, identity2), dim=1)
        x = self.double_conv8(x)
                              
        x = self.deconv4(x)
        identity1 = F.resize(identity1, size=x.shape[2:])
        x = torch.cat((x, identity1), dim=1)
        x = self.double_conv9(x)  
        
        return self.final_conv(x)