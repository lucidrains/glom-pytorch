import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

# class

class Glom(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        layers = 6,
        image_size = 224,
        patch_size = 14
    ):
        super().__init__()
        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size ** 2 * 3, dim)
        )

        self.init_embeddings = nn.Parameter(torch.randn(layers, dim))

    def forward(self, img):
        b = img.shape[0]
        tokens = self.image_to_tokens(img)
        print(tokens.shape)
        return tokens
