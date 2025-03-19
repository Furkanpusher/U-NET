import torch
import torch.nn as nn
import torch.nn.functional as F

class Blk2D(nn.Module):
    def __init__(self, dim, mod="mul"):
        super().__init__()
        self.mod = mod
        self.norm = nn.BatchNorm2d(dim)
        self.f = nn.Conv2d(dim, 6 * dim, kernel_size=1)
        self.g = nn.Conv2d(3 * dim, dim, kernel_size=1)
        
    def forward(self, x):
        input = x
        x = self.f(self.norm(x))
        x1, x2 = torch.split(x, x.size(1)//2, dim=1)
        x = F.relu(x1) + x2 if self.mod == "sum" else F.relu(x1) * x2
        x = self.g(x)
        return input + x

class STARNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, dim=100, mod="mul", depth=4,
                 use_stem=True, use_head=True):
        super().__init__()
        self.use_stem = use_stem
        self.use_head = use_head
        out_channels = out_channels or in_channels  # Set default if None

        if use_stem:
            self.stem = nn.Conv2d(in_channels, dim, kernel_size=1)
        else:
            dim = in_channels  # Directly use input dim
            self.stem = nn.Identity()

        self.net = nn.Sequential(*[Blk2D(dim, mod) for _ in range(depth)])
        self.norm = nn.BatchNorm2d(dim)
        
        if use_head:
            self.head = nn.Conv2d(dim, out_channels, kernel_size=1)
        else:
            self.head = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.net(x)
        x = self.norm(x)
        return self.head(x)
