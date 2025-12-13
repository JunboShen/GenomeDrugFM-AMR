import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

class PatchEmbed1D(nn.Module):
    """
    1D Patch Embedding for Genome Segments.

    Splits each genome segment of length L with in_chans channels into patches of length patch_size.

    Expects input shape: [B, N, L, in_chans], where
      B = batch size,
      N = number of segments per sample,
      L = length of each segment,
      in_chans = number of channels (e.g., 4 for one-hot encoding).

    Returns output shape: [B, N, num_patches, embed_dim] where num_patches = L // patch_size.
    """

    def __init__(self, patch_size=50, in_chans=4, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        # Conv1d will act along the segment length dimension.
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, N, L, in_chans]
        B, N, L, C = x.shape
        # Combine batch and segment dimensions: [B*N, L, C]
        x = x.view(B * N, L, C)
        # For Conv1d, rearrange to: [B*N, C, L]
        x = x.transpose(1, 2)
        # Apply convolution: output shape becomes [B*N, embed_dim, num_patches]
        x = self.proj(x)
        # Rearrange back to: [B*N, num_patches, embed_dim]
        x = x.transpose(1, 2)
        # Reshape to separate batch and segment dimensions: [B, N, num_patches, embed_dim]
        x = x.view(B, N, -1)
        return x
    
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


class CNNTransMIL(nn.Module):
    def __init__(self, input_dim, dim=1024, n_classes=2, input_dim_kmer=512, patch_size=4096, in_chans=4, embed_dim=1536):
        super(CNNTransMIL, self).__init__()
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
        self.patch_embed = PatchEmbed1D(patch_size, in_chans, embed_dim)
    def forward(self, x, drug, H_kmer):
        x = self.patch_embed(x) 
        H_kmer = self.norm_layer(H_kmer)
        H = torch.cat([x, drug.repeat(1, x.shape[1], 1)], dim=2)  # [B, N, L]
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
    model = CNNTransMIL(input_dim=1024, dim=512, n_classes=2, input_dim_kmer=512).to(device)
    output = model(data, data_kmer)
    print(output.shape)
    
    


