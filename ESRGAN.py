import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=6, downsample_size=(32, 32)):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.downsample_size = downsample_size

    def forward(self, x):
        batch_size, C, width, height = x.size()
        x_down = F.adaptive_avg_pool2d(x, self.downsample_size)

        query = self.query_conv(x_down).view(batch_size, -1, self.downsample_size[0]*self.downsample_size[1]).permute(0, 2, 1)
        key = self.key_conv(x_down).view(batch_size, -1, self.downsample_size[0]*self.downsample_size[1])
        value = self.value_conv(x_down).view(batch_size, -1, self.downsample_size[0]*self.downsample_size[1])

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, self.downsample_size[0], self.downsample_size[1])

        # Upsample to match original size
        out = F.interpolate(out, size=(width, height), mode='bilinear', align_corners=True)

        return self.gamma * out + x

class ModifiedRRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(ModifiedRRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)
        self.attention = SelfAttention(nf)  # added self attention

    def forward(self, x):
        out = self.RDB1(x)
        out = self.attention(out)  # added self attention
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESRGAN_g(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):  # input/output channels, feature maps, rrdb blocks, growth channel
        super(ESRGAN_g, self).__init__()
        # First convolution layer
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(functools.partial(ModifiedRRDB, nf=nf, gc=gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # Learned Upsampling layers
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))  # leanred upscaling
        fea = self.lrelu(self.pixel_shuffle(self.upconv2(fea)))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class ESRGAN_d(nn.Module):
    def __init__(self, input_channels=3, base_filters=64):
        super(ESRGAN_d, self).__init__()

        self.initial_conv = spectral_norm(nn.Conv2d(input_channels, base_filters, kernel_size=3, stride=1, padding=1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks = nn.Sequential(
            spectral_norm(nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 16, base_filters * 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 32, base_filters * 16, 1, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 16, base_filters * 8, 1, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            spectral_norm(nn.Conv2d(base_filters * 8, base_filters * 4, 1, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_filters * 4, base_filters * 2, 1, 1, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 2, 1, 1, 1, 0),
        )

    def forward(self, x):
        x = self.lrelu(self.initial_conv(x))
        x = self.conv_blocks(x)
        x = self.output_layer(x)
        return torch.sigmoid(x.view(x.size(0), -1))
