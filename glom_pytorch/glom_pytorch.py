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
        # bottom level - incoming image, tokenize and add position
        num_patches = (image_size // patch_size) ** 2
        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size ** 2 * 3, dim)
        )
        self.pos_emb = nn.Embedding(num_patches, dim)

        # initial embeddings for all levels of a column
        self.init_levels = nn.Parameter(torch.randn(layers, dim))

    def forward(self, img):
        b, device = img.shape[0], img.device

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        pos_embs = self.pos_emb(torch.arange(n, device = device))
        bottom_level = tokens + rearrange(pos_embs, 'n d -> () n d')
        levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        return tokens
