import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .graph import SkeletonGraph


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x, A):
        # x: (N, C, T, V)  A: (K, V, V)
        x = self.conv(x)
        N, KC, T, V = x.shape
        K = self.kernel_size
        C = KC // K
        x = x.view(N, K, C, T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.0):
        super().__init__()
        kernel_size = A.shape[0]
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))

        self.gcn = SpatialGraphConv(in_channels, out_channels, kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1),
                      stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        if hasattr(self, 'edge_importance'):
            pass
        self.edge_importance = nn.Parameter(torch.ones_like(self.A))

    def forward(self, x):
        res = self.residual(x)
        x = self.bn(x)
        x = self.gcn(x, self.A * self.edge_importance)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCN(nn.Module):
    """
    ST-GCN for distress gesture classification.
    Input:  (N, C, T, V) — batch × channels × frames × joints
    Output: (N, num_classes)
    """

    def __init__(self, in_channels=3, num_classes=4, dropout=0.5):
        super().__init__()

        graph = SkeletonGraph(strategy='spatial')
        A = graph.A.astype(np.float32)
        num_subsets = A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[-1])

        # (in_ch, out_ch, stride)
        block_cfg = [
            (in_channels, 64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
            (256,         256, 1),
        ]

        self.blocks = nn.ModuleList([
            STGCNBlock(ic, oc, A, stride=s, dropout=dropout)
            for ic, oc, s in block_cfg
        ])

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
