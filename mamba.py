# https://github.com/apapiu/mamba_small_bench

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0):
        super().__init__()

        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)


class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, seq_len=None, global_pool=False, dropout=0):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim, dropout_level=dropout) for _ in range(n_layers)])
        self.global_pool = global_pool #for classification or other supervised learning.

    def forward(self, x):
        #for input (bs, n, d) it returns either (bs, n, d) or (bs, d) is global_pool
        out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
        return out
    
class MambaImgClassifier(nn.Module):
    def __init__(self, patch_size=4, img_size=28, n_channels=1, embed_dim=256, n_layers=6, dropout=0):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        seq_len = int((self.img_size/self.patch_size)**2)
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                   p1=self.patch_size, p2=self.patch_size)

        self.func = nn.Sequential(
            self.rearrange,
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            MambaTower(embed_dim, n_layers, seq_len=seq_len, global_pool=True, dropout=dropout),
            nn.Linear(embed_dim, 10))

    def forward(self, x):
        return self.func(x)
