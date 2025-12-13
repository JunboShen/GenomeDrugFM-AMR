import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of Moore-Penrose iterations
            residual=True,  # extra residual connection
            dropout=0.1
        )
        

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    """
    Parametric Positional Encoding Generator (PPEG) for 1D data.
    Uses depthwise separable Conv1d instead of Conv2d.
    """

    def __init__(self, dim=512, kernel_sizes=[7, 5, 3]):
        super(PPEG, self).__init__()
        self.proj_layers = nn.ModuleList([
            nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim) for k in kernel_sizes
        ])

    def forward(self, x):
        """
        x: [B, N, C]  - Batch, Sequence Length, Embedding Dim
        """
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]  # Separate CLS token
        feat_token = feat_token.transpose(1, 2)  # [B, C, N]

        for layer in self.proj_layers:
            feat_token = feat_token + layer(feat_token)  # Apply PPEG layers

        feat_token = feat_token.transpose(1, 2)  # Back to [B, N, C]
        x = torch.cat((cls_token.unsqueeze(1), feat_token), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, input_dim, dim=1024, n_classes=2, input_dim_kmer=512):
        super(TransMIL, self).__init__()
        self.dim = dim
        self.pos_layer = PPEG(dim=self.dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim+input_dim_kmer, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=self.dim)
        self.layer2 = TransLayer(dim=self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self._fc2 = nn.Linear(self.dim, self.n_classes)
        self.norm_layer = nn.LayerNorm(input_dim_kmer)

    def forward(self, H, H_kmer):
        H_kmer = self.norm_layer(H_kmer)
        #concatenate H_kmer (batch_size, 1, input_dim_kmer) to each instance of H (batch_size, num_instances, input_dim)
        H = torch.cat((H, H_kmer.unsqueeze(1).repeat(1, H.size(1), 1)), dim=2)
        
        h = self._fc1(H)  # [B, n, 512]

        B, N, C = h.shape
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)  # [B, N+1, 512]

        h = self.layer1(h)
        h = self.pos_layer(h)  # Apply 1D PPEG
        h = self.layer2(h)
        h = self.norm(h)[:, 0]  # CLS token

        logits = self._fc2(h)  # [B, n_classes]

        return  logits


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = torch.randn((1, 5999, 1024), device=device)
    data_kmer = torch.randn((1, 512), device=device)
    model = TransMIL(input_dim=1024, dim=512, n_classes=2, input_dim_kmer=512).to(device)
    output = model(data, data_kmer)
    print(output.shape)
    
    


