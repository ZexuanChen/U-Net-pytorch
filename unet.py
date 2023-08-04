import torch
import torch.nn as nn
import numpy as np
# 1-5层的通道数
filters = [64, 128, 256, 512, 1024]


# 进行两次卷积的模块,输入通道数为in_channels,输出通道数为out_channels
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),     # 卷积
            nn.BatchNorm2d(out_channels),                                       # 归一化
            nn.ReLU(inplace=True),                                              # 激活
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Unet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器
        # 五个部分，每个部分就是一层进行两次卷积模块
        self.encoder = nn.ModuleList([
            DoubleConvBlock(in_channels, filters[0]),
            DoubleConvBlock(filters[0], filters[1]),
            DoubleConvBlock(filters[1], filters[2]),
            DoubleConvBlock(filters[2], filters[3]),
            DoubleConvBlock(filters[3], filters[4]),
        ])

        # 解码器
        # 四个部分，每个部分包含了一次反卷积上采样模块和一个进行两次卷积的模块
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2),    # 上采样，在上采样填充的同时通道数减半
            DoubleConvBlock(filters[4], filters[3]),                                # 进行两次卷积，通道数减半

            nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2),
            DoubleConvBlock(filters[3], filters[2]),

            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2),
            DoubleConvBlock(filters[2], filters[1]),

            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2),
            DoubleConvBlock(filters[1], filters[0])
        ])

        # 输出层，将最终结果的通道数调整为out_channels
        self.out = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 提取的特征
        feats = []
        # 主干特征提取网络（编码）
        for block in self.encoder:
            x = block(x)                                     # 进行两次卷积，通道数减半
            feats.append(x)                                  # 保存该层特征
            #print(x.shape)
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)     # 2x2最大池化

        feats = feats[:-1][::-1]                             # 去掉最后一次的特征并倒序

        # 强化特征提取网络（解码）
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)                         # 上采样，行列x2同时通道数减少
            #print(x.shape)
            feat = feats[idx // 2]
            if x.shape != feat.shape:                        # 如果x与要拼接的特征层长宽不一致，将x上采样填充成与特征层一样的尺寸
                x = nn.Upsample(size=feat.shape[2:], mode='bilinear', align_corners=True)(x)

            x = torch.cat((feat, x), dim=1)                  # 拼接，通道数增加
            x = self.decoder[idx + 1](x)                     # 进行一次双卷积，通道数减少
            #print(x.shape)

        return torch.softmax(self.out(x).permute(0, 2, 3, 1), dim=-1)                                # 输出


#  测试
# model = UNet(in_channels=3, out_channels=2)
# print(model)
#
# input_image = torch.rand(1, 3, 512, 512)
#
# output = model(input_image)
#
# print(output.shape)
