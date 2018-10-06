import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.dconv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InConv, self).__init__()
        self.inconv = DoubleConv(in_channel, out_channel)

    def forward(self, x):
        x = self.inconv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownConv, self).__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel, out_channel),
        )

    def forward(self, x):
        x = self.downconv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(UpConv, self).__init__()

        # UpSampling
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_h = x1.size()[2] - x2.size()[2]
        diff_w = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diff_h // 2, int(diff_h / 2),
                        diff_w // 2, int(diff_w / 2)))
        # (padLeft, padRight, padTop, padBottom)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes

        self.in_planes = 64

        self.inconv = InConv(self.in_channel, self.in_planes)
        self.downconv1 = DownConv(64, 128)
        self.downconv2 = DownConv(128, 256)
        self.downconv3 = DownConv(256, 512)
        self.downconv4 = DownConv(512, 1024)

        self.upconv1 = UpConv(1024, 512)
        self.upconv2 = UpConv(512, 256)
        self.upconv3 = UpConv(256, 128)
        self.upconv4 = UpConv(128, 64)

        self.outconv = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)

        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.upconv4(x, x1)
        x = self.outconv(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channel=3, n_classes=1)
    model = model.to(device)

    summary(model, (3, 128, 128))
