import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2d, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(2, mode='bilinear')
        self.conv = DoubleConv2d(in_channels, out_channels, in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        y = torch.cat([x2, x1], dim=1)
        return self.conv(y)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels

        self.i = DoubleConv2d(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.o = nn.Sequential(
            nn.Conv2d(64, self.n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.i(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.o(x)


