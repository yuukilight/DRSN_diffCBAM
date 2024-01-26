# yuukilight
import torch
from torch import nn
import math

def calculate_padding(input_length, kernel_size, stride):
    output_length = input_length // stride + 1
    padding = max((output_length - 1) * stride + kernel_size - input_length, 0)
    return padding // 2

class senet(nn.Module):
    def __init__(self, channel, ratio = 4):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1])
        return x * fc


class channel_attention(nn.Module):
    def __init__(self, inchannel, ratio = 4):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        max_pool_out = self.max_pool(x).view([b,c])
        avg_pool_out = self.avg_pool(x).view([b,c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1])
        return out * x

class spacial_attention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(spacial_attention, self).__init__()
        # padding = 7 // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, 1, padding = 'same', bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        max_pool_out, _ = torch.max(x, dim = 1, keepdim= True)
        avg_pool_out = torch.mean(x, dim = 1, keepdim= True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim = 1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x

class Cbam(nn.Module):
    def __init__(self, inchannel, ratio = 4, kernel_size = 7):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(inchannel, ratio)
        self.spacial_attention = spacial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x

class EcaBlock(nn.Module):
    def __init__(self, channel, gamma = 2, b = 1):
        super(EcaBlock, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, bias = False, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        avg_pool_out = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg_pool_out)
        out = self.sigmoid(out).view([b,c,1])
        # print(out.shape)
        # print(x.shape)
        return out * x



class RSU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # BRC = BatchNormation + ReLU + Convolution
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地修改参数
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)
        result = x + input
        return result

class RSSU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        # BRC = BatchNormation + ReKU + Convolution
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地修改参数
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        # GAP
        # AdaptiveAvgPool1d 会自动调节 kernel and stride
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid()
        )
        # Flatten 需要输入展开的维度范围，默认从 1 到所有维度。（只保留 0 维度，即 batch）
        self.flatten = nn.Flatten()
        self.average_pool = nn.AvgPool1d(kernel_size=1, stride=2)

    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)
        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        # sign(x) 用于返回符号 -1，0，1
        x = torch.mul(torch.sign(x), n_sub)
        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            zero_padding = torch.zeros(input.shape).cuda()
            input = torch.cat((input, zero_padding), dim=1)

        result = x + input
        return result

