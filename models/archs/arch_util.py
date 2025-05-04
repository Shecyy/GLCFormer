import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
import numbers


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.down = nn.AvgPool2d(2, stride=2)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        fea = []
        x_1 = self.act(self.conv1_1(x))
        x_1 = self.act(self.conv1_2(x_1))
        # (b, 64, 224, 224)

        x_1_down = self.down(x_1)

        x_2 = self.act(self.conv2_1(x_1_down))
        x_2 = self.act(self.conv2_2(x_2))
        # (b, 128, 112, 112)

        x_2_down = self.down(x_2)

        x_3 = self.act(self.conv3_1(x_2_down))
        x_3 = self.act(self.conv3_2(x_3))
        x_3 = self.act(self.conv3_3(x_3))
        # (b, 256, 56, 56)

        x_3_down = self.down(x_3)

        x_4 = self.act(self.conv4_1(x_3_down))
        x_4 = self.act(self.conv4_2(x_4))
        x_4 = self.act(self.conv4_3(x_4))
        # (b, 512, 28, 28)

        # x_4_down = self.down(x_4)
        #
        # x_5 = self.act(self.conv5_1(x_4_down))
        # x_5 = self.act(self.conv5_2(x_5))
        # x_5 = self.act(self.conv5_3(x_5))
        # (b, 512, 14, 14)

        # x_5_down = self.down(x_5)

        fea.append(x_1)
        fea.append(x_2)
        fea.append(x_3)
        fea.append(x_4)
        # fea.append(x_5)

        return fea


##  Pool Feed-forward Network (PFN)
class FeedForward4(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False):
        super(FeedForward4, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        in_dim1, in_dim2, in_dim3 = dim // 4, dim // 4, dim - ((dim // 4) * 2)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_dim1, hidden_features, kernel_size=1, bias=bias),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_dim2, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_dim3, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=bias),
            nn.GELU(),
        )

        # 分支权重
        self.branch1_weight = nn.Parameter(torch.ones(size=[1, hidden_features, 1, 1]), requires_grad=True)
        self.branch2_weight = nn.Parameter(torch.ones(size=[1, hidden_features, 1, 1]), requires_grad=True)

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.size()
        b1, b2, b3 = torch.split(x, [C // 4, C // 4, C - (C // 4) * 2], dim=1)

        b1 = F.adaptive_avg_pool2d(b1, 1)
        b1 = self.branch1(b1)
        b1 = F.interpolate(b1, size=[H, W]) * torch.tanh(self.branch1_weight)

        b2 = F.adaptive_avg_pool2d(b2, [H // 2, W // 2])
        b2 = self.branch2(b2)
        b2 = F.interpolate(b2, size=[H, W]) * torch.tanh(self.branch2_weight)

        b3 = self.branch3(b3)

        x = self.project_out(torch.cat([b1, b2, b3], dim=1))
        return x
