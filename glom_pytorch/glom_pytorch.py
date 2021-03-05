from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# constants

TOKEN_ATTEND_SELF_VALUE = -5e-4

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# class

class GroupedFeedForward(nn.Module):
    def __init__(self, *, dim, groups, mult = 4):
        super().__init__()
        total_dim = dim * groups # levels * dim
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.Conv1d(total_dim, total_dim * 4, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(total_dim * 4, total_dim, 1, groups = groups),
            Rearrange('b (l d) n -> b n l d', l = groups)
        )

    def forward(self, levels):
        return self.net(levels)

class ConsensusAttention(nn.Module):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        if self.local_consensus_radius > 0:
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            h = w = int(sqrt(n))
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out

# main class

class Glom(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        levels = 6,
        image_size = 224,
        patch_size = 14,
        consensus_self = False,
        local_consensus_radius = 0
    ):
        super().__init__()
        # bottom level - incoming image, tokenize and add position
        num_patches_side = (image_size // patch_size)
        num_patches =  num_patches_side ** 2
        self.levels = levels

        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size ** 2 * 3, dim)
        )
        self.pos_emb = nn.Embedding(num_patches, dim)

        # initial embeddings for all levels of a column
        self.init_levels = nn.Parameter(torch.randn(levels, dim))

        # bottom-up and top-down
        self.bottom_up = GroupedFeedForward(dim = dim, groups = levels - 1)
        self.top_down = GroupedFeedForward(dim = dim, groups = levels - 1)

        # consensus attention
        self.attention = ConsensusAttention(num_patches_side, attend_self = consensus_self, local_consensus_radius = local_consensus_radius)

    def forward(self, img, iters = None, levels = None, return_all = False):
        b, h, w, _, device = *img.shape, img.device
        iters = default(iters, self.levels * 2)   # need to have twice the number of levels of iterations in order for information to propagate up and back down. can be overridden

        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        pos_embs = self.pos_emb(torch.arange(n, device = device))
        pos_embs = rearrange(pos_embs, 'n d -> () n () d')

        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        hiddens = [levels]

        num_contributions = torch.empty(self.levels, device = device).fill_(4)
        num_contributions[-1] = 3  # top level does not get a top-down contribution, so have to account for this when doing the weighted mean

        for _ in range(iters):
            levels_with_input = torch.cat((bottom_level, levels), dim = -2)  # each iteration, attach original input (with positional embedding) at the bottom level

            bottom_up_out = self.bottom_up(levels_with_input[..., 1:-1, :])
            top_down_out = self.top_down(levels_with_input[..., 2:, :] + pos_embs) # positional embeddings given to top-down networks

            bottom_up_out = torch.cat((bottom_level, bottom_up_out), dim = -2)
            top_down_out = F.pad(top_down_out, (0, 0, 0, 1), value = 0.)

            consensus = self.attention(levels)

            levels_sum = torch.stack((levels, bottom_up_out, top_down_out, consensus)).sum(dim = 0) # hinton said to use the weighted mean of (1) bottom up (2) top down (3) previous level value {t - 1} (4) consensus value
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            levels = levels_mean  # set for next iteration
            hiddens.append(levels)

        if return_all:
            return torch.stack(hiddens)  # return (time step, batch, num columns, levels, dimension)

        return levels
