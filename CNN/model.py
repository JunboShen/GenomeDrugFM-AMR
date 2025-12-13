import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from typing import Callable

class AttentionPool(nn.Module):
    def __init__(self, dim: int, pool_size: int = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = None,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation_fn: Callable = nn.GELU(),
        attention_pooling: bool = False,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            dilation=dilation,
        )
        self.batch_norm = (
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity()
        )
        if pool_size:
            self.pool = (
                AttentionPool(out_channels, pool_size)
                if attention_pooling
                else nn.MaxPool1d(pool_size)
            )
        else:
            self.pool = nn.Identity()

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.pool(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        layer_norm: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.2,
        activation_fn: Callable = nn.ReLU(),
    ):
        super().__init__()
        if layer_norm and batch_norm:
            batch_norm = False
            logging.info(
                "LayerNorm and BatchNorm both used in the dense layer, "
                "defaulting to LayerNorm only"
            )

        self.dense = nn.Linear(in_features, out_features, bias=use_bias)
        
        self.layer_norm = (
            nn.LayerNorm(out_features, elementwise_affine=False)
            if layer_norm
            else nn.Identity()
        )
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        )
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.batch_norm(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x

# ----------  per‑segment encoder (reuse MD‑CNN tower)  ----------
class SegmentEncoderCNN(nn.Module):
    """
    Encode a single DNA segment (one‑hot [B,4,L]) into a fixed vector.
    """
    def __init__(self, seg_len: int, embed_dim: int = 256, in_chans: int = 4):
        super().__init__()
        self.conv_tower = nn.Sequential(
            ConvLayer(in_chans, 64, 12, activation_fn=nn.ReLU(), batch_norm=False),
            ConvLayer(64, 64, 12, pool_size=3, activation_fn=nn.ReLU(), batch_norm=False),
            ConvLayer(64, 32,  3, activation_fn=nn.ReLU(), batch_norm=False),
            ConvLayer(32, 32,  3, pool_size=3, activation_fn=nn.ReLU(), batch_norm=False)
        )

        # flatten‑dense
        flat_dim = 32 * (seg_len // 3) // 3
        self.head = nn.Sequential(
            Flatten(),
            DenseLayer(flat_dim, embed_dim, activation_fn=nn.ReLU(), dropout=0.0, batch_norm=False)
        )

    def forward(self, x):              # x:[B,4,L]
        return self.head(self.conv_tower(x))   # -> [B, embed_dim]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# ----------  simple attention MIL pooling  ----------
class AttnMILPool(nn.Module):
    """
    H ∈ ℝ[B, N, d]  →  z ∈ ℝ[B, d]
    """
    def __init__(self, d, h=128):
        super().__init__()
        self.V = nn.Linear(d, h, bias=True)
        self.U = nn.Linear(d, h, bias=True)
        self.w = nn.Linear(h, 1, bias=False)

    def forward(self, H):                         # H:[B,N,d]
        A = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))  # gated tanh×sigmoid
        α = torch.softmax(self.w(A).squeeze(-1), dim=1)       # α:[B,N]
        z = torch.sum(α.unsqueeze(-1) * H, dim=1)             # z:[B,d]
        return z, α                                           # return weights if you ever want them

# ----------  full model  ----------
class MDCNN_MIL_Drug(nn.Module):
    """
    Baseline AMR model:
      ‑ Segment‑level MD‑CNN encoder
      ‑ Attention MIL over segments
      ‑ Drug embedding concatenation
    """
    def __init__(
        self,
        seg_len: int,          # length L of every segment
        emb_dim: int = 256,    # segment embedding size
        drug_dim: int = 128,   # dimension of pre‑computed drug embedding you will feed in
        n_classes: int = 2
    ):
        super().__init__()
        self.seg_encoder = SegmentEncoderCNN(seg_len, emb_dim)
        self.mil_pool     = AttnMILPool(emb_dim)
        self.classifier   = nn.Sequential(
            nn.Linear(emb_dim + drug_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, dna, drug):
        """
        dna   : [B, N, L, 4]  one‑hot
        drug  : [B, D]        any drug‑language embedding (e.g., ChemBERTa, ECFP‑MLP)
        """
        B, N, L, C = dna.shape
        dna = rearrange(dna, "b n l c -> (b n) c l")      # fold segments into batch
        seg_repr = self.seg_encoder(dna)                  # [(B·N), emb]
        seg_repr = rearrange(seg_repr, "(b n) d -> b n d", b=B, n=N)

        bag_repr, attn = self.mil_pool(seg_repr)          # [B, emb]
        joint_repr = torch.cat([bag_repr, drug], dim=-1)  # concat drug emb
        logits = self.classifier(joint_repr)              # [B, n_classes]
        return logits, attn            # returning α is handy for inspection

# ---------------------  tiny smoke test  ---------------------
if __name__ == "__main__":
    B, N, L = 1, 13, 4096               # toy batch
    dna  = torch.randint(0, 2, (B, N, L, 4)).float()      # fake one‑hot
    drug = torch.randn(B, 2304)                            # fake drug emb

    model = MDCNN_MIL_Drug(seg_len=L, drug_dim=2304, n_classes=2)
    logits, attn = model(dna, drug)
    print("logits:", logits.shape, "attention:", attn.shape)
    
    


