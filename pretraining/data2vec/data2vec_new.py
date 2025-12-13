"""
Masked Pretraining for Genome Segments

Input: genome segments with shape [N, L, 4] where L is the sequence length (e.g., 1000).
We use a 1D patch embedder (with patch_size defined as the length of each patch) to split the sequence.
Reconstruction is done at the patch level (each patch is flattened to patch_size * 4).
Positional embeddings are generated using a 1D sincos function.
"""

from functools import partial
import numpy as np
from genome_encoder.torchscale.model.LongNet import make_longnet_from_name
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema import EMA

class SegmentEncoder1D(nn.Module):
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

    def forward(self, x, pad_mask=None):
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
        
        if pad_mask is not None:
            # pad_mask: [B, N], True = padded segment
            x = x * (~pad_mask).unsqueeze(-1)
        return x

# class SegmentEncoder1D(nn.Module):
#     """
#     One-hot segments [B,N,4096,4] -> per-segment embedding [B,N,D].
#     Respects pad_mask: True = padded segment.
#     """
#     def __init__(self, in_chans=4, embed_dim=1536, stem_dim=256):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.stem = nn.Sequential(
#             nn.Conv1d(in_chans, stem_dim, kernel_size=7, padding=3, bias=False),
#             nn.GELU(),
#             nn.Conv1d(stem_dim, stem_dim, kernel_size=3, padding=1, groups=stem_dim, bias=False),  # depthwise
#             nn.GELU(),
#             nn.Conv1d(stem_dim, embed_dim, kernel_size=1, bias=False),  # pointwise to D
#             nn.GELU(),
#         )
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x, pad_mask=None):
#         # x: [B, N, L=4096, C=4]
#         B, N, L, C = x.shape
#         x = x.view(B * N, L, C).transpose(1, 2)  # [B*N, C, L]
#         h = self.stem(x)                         # [B*N, D, L]
#         h = h.mean(dim=-1)                       # global average over length -> [B*N, D]
#         h = self.norm(h)                         # [B*N, D]
#         h = h.view(B, N, self.embed_dim)         # [B, N, D]
#         if pad_mask is not None:
#             h = h * (~pad_mask).unsqueeze(-1)    # keep padded embeddings at 0
#         return h                                  # tokens per segment

class Data2Vec(nn.Module):
    """
    Data2Vec main module.
    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """

    def __init__(self,
                 cfg,
                 max_seq_len=8192, #16384,
                 min_seq_len=1024,
                 patch_size=2000,  # length of each patch; number of patches = seq_length // patch_size
                 in_chans=4,  # one-hot channels (A, C, G, T)
                 embed_dim=1536,  # encoder embedding dim
                 depth=12,  # number of transformer blocks in encoder
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 max_size=6400000,  # maximum sequence length for optimal segment length
                 seg_size=500,
                 dropout=0.25,
                 drop_path_rate=0.1,
                 **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embed = SegmentEncoder1D(patch_size, in_chans, embed_dim)
        # self.patch_embed = SegmentEncoder1D(in_chans, embed_dim, stem_dim=embed_dim//6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_name = "LongNet_{}_layers_{}_dim".format(depth, embed_dim)
        if kwargs.get("mlp_ratio", 4.0) != 4.0:
            self.encoder_name += "_mlp{}".format(kwargs.get("mlp_ratio"))

        # get optimal segment length
        segment_length = self.get_optimal_segment_length(max_seq_len, min_seq_len)
        self.encoder = make_longnet_from_name(self.encoder_name, drop_path_rate=drop_path_rate, dropout=dropout,
                                              segment_length=segment_length)
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        #Token for split position, with ones/zeros for the split position
        #self.split_token = nn.Parameter(torch.ones(1, embed_dim))
        self.split_token = nn.Parameter(torch.zeros(1, embed_dim))
        # Positional embeddings
        #self.register_buffer('pos_embed', torch.zeros(1, max_seq_len + 1, embed_dim), persistent=False)  # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))
        #self.norm = norm_layer(embed_dim)
        self.__dict__.update(kwargs)

        self.cfg = cfg

        self.ema = EMA(self.encoder, self.cfg)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()

        self.ema_decay = self.cfg.model.ema_decay
        self.ema_end_decay = self.cfg.model.ema_end_decay
        self.ema_anneal_end_step = self.cfg.model.ema_anneal_end_step
        self.initialize_weights()

    @staticmethod
    def interpolate_pos_embed(pos_embed, new_seq_len):
        """
        pos_embed: [1, old_seq_len+1, dim]
        new_seq_len: int, new desired sequence length
        """
        cls_token = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]  # [1, old_seq_len, dim]

        pos_tokens = pos_tokens.permute(0, 2, 1)  # [1, dim, old_seq_len]
        pos_tokens = F.interpolate(pos_tokens, size=new_seq_len, mode='linear', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 1)  # [1, new_seq_len, dim]

        new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
        return new_pos_embed

    def sincos_pos_embedding_1d(self, seq_len, dim):
        """Generate 1D fixed sinusoidal positional embeddings"""
        position = np.arange(seq_len)[:, np.newaxis]  # [seq_len, 1]
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))  # [dim/2]
        pos_embed = np.zeros((seq_len, dim))
        pos_embed[:, 0::2] = np.sin(position * div_term)
        pos_embed[:, 1::2] = np.cos(position * div_term)
        return pos_embed

    def get_optimal_segment_length(self, max_seq_len: int = 16384, min_seq_len: int = 1024) -> str:
        # calculate the segment length
        segment_length = np.linspace(np.log2(min_seq_len), int(np.log2(max_seq_len)), 5)
        segment_length = np.power(2, segment_length).astype(int)
        # convert to str format
        segment_length = str(list(segment_length))
        return segment_length

    def initialize_weights(self):
        # Initialize learnable parameters.
        # pos_embed = self.sincos_pos_embedding_1d(self.pos_embed.shape[-2], self.pos_embed.shape[-1])
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize patch_embed
        # for m in self.patch_embed.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.split_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, indice_list, mask_ratio, pad_mask=None):
        """
        Perform per-sample random masking by shuffling.
        x: [N, L, D] where L = num_patches
        Returns:
            x_masked: masked tokens with masked positions replaced by self.mask_token,
            mask: binary mask [N, L] with 0 for kept and 1 for masked,
        """
        N, L, D = x.shape
        x_masked = x.clone()
        if pad_mask is not None:
            valid = ~pad_mask.to(torch.bool)  
        else:
            valid = torch.ones(N, L, dtype=torch.bool, device=x.device)
            
        mask = torch.zeros(N, L, device=x.device, dtype=torch.bool)

        for i in range(N):
            # First, replace tokens at indice_list with split_token
            #if indice_list is not None, and there are 0 in the end of the list due to padding, then remove the 0s
            if indice_list is not None:
                # idx_i = indice_list[i]
                # idx_i = idx_i[:idx_i.index(0)] if (0 in idx_i) else idx_i
                # protected = torch.tensor([p for p in idx_i if p > 0 and p < L], device=x.device, dtype=torch.long)
                #indice_list[i] = indice_list[i][:indice_list[i].index(0)] if 0 in indice_list[i] else indice_list[i]
                if 0 in indice_list[i]:
                    zero_index = (indice_list[i] == 0).nonzero(as_tuple=True)[0][0].item()  # Find the first occurrence of 0
                    sliced_indices = indice_list[i][:zero_index]  # Create a new tensor for the sliced version
                else:
                    sliced_indices = indice_list[i]  # No change if 0 is not in the tensor

                protected = torch.tensor(sliced_indices, device=x.device, dtype=torch.long)
                x_masked[i, protected] += self.split_token.to(x_masked.dtype).expand_as(x_masked[i, protected])
                #x_masked[i, protected] = self.split_token.to(x_masked.dtype).expand_as(x_masked[i, protected])
            else:
                protected = torch.empty(0, dtype=torch.long, device=x.device)
            # Then perform masking on non-protected indices
            all_idx = torch.arange(L, device=x.device)
            allowed = all_idx[valid[i]]
            if protected.numel() > 0:
                allowed = allowed[~torch.isin(allowed, protected)]
                
            num_allowed = allowed.numel()
            len_keep = int(num_allowed * (1 - mask_ratio)) if num_allowed > 0 else 0
            
            # Randomly shuffle allowed indices
            noise = torch.rand(num_allowed, device=x.device)
            _, shuffle_idx = torch.sort(noise)
            allowed_shuffled = allowed[shuffle_idx]
            keep_indices = allowed_shuffled[:len_keep]
            
            # Determine indices to mask: allowed but not kept
            mask_indices = allowed[~torch.isin(allowed, keep_indices)]
            
            if mask_indices.numel() > 0:
                x_masked[i, mask_indices] = self.mask_token.to(x_masked.dtype).expand_as(x_masked[i, mask_indices])
                mask[i, mask_indices] = True
            #else if mask_indices.numel() == 0:, then set mask to all indices, but don't replace with mask_token
            else:
                mask[i, :] = True
    
        return x_masked, mask
    
    def _build_regression_head(self):
        """
        Construct the regression head consisting of linear and activation layers.

        Each modality might have its own regression block.

        Returns:
            A nn.Module layer or block of layers
        """
        return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim))

        # if self.modality in ['audio', 'vision']:
        #     return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward_encoder(self, x, indice_lsit, mask_ratio, pad_mask=None):
        """
        Args:
            x: input genome segments, shape [N, L, in_chans]
        """
        # Embed patches using 1D patch embedding.
        x = self.patch_embed(x, pad_mask)
        # print("x: ", x.shape)
        # Add positional embedding (skip cls token for now)
        N, L, D = x.shape
        # self.pos_embed = self.pos_layer(x)
        # x = x + self.pos_embed[:, 1:, :]

        # Apply random masking.
        x, mask = self.random_masking(x, indice_lsit, mask_ratio, pad_mask)
        # Append cls token.
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        #update pad_mask to account for cls token
        if pad_mask is not None:
            pad_mask = torch.cat([torch.zeros(pad_mask.shape[0], 1, device=pad_mask.device, dtype=pad_mask.dtype), pad_mask], dim=1)
        
        # Interpolate positional embeddings if longer than trained length
        if L + 1 > self.pos_embed.shape[1]:
            pos_embed = self.interpolate_pos_embed(self.pos_embed, L)
        else:
            pos_embed = self.pos_embed[:, :L + 1, :]
        pos_embed = pos_embed.expand(x.shape[0], -1, -1)
        x = x + pos_embed

        # Apply transformer encoder blocks.
        x = self.encoder(src_tokens=None, token_embeddings=x, encoder_padding_mask=pad_mask)["encoder_out"]
        #x = self.norm(x)
        return x, mask

    def forward_latent(self, x,  pad_mask=None):
        """
        Args:
            x: input genome segments, shape [N, L, in_chans]
        """
        # Embed patches using 1D patch embedding.
        x = self.patch_embed(x, pad_mask)

        N, L, D = x.shape
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # Append cls token.
        x = torch.cat([cls_tokens, x], dim=1)

        #update pad_mask to account for cls token
        if pad_mask is not None:
            pad_mask = torch.cat([torch.zeros(pad_mask.shape[0], 1, device=pad_mask.device, dtype=pad_mask.dtype), pad_mask], dim=1)
            
        # Interpolate positional embeddings if longer than trained length
        if L + 1 > self.pos_embed.shape[1]:
            pos_embed = self.interpolate_pos_embed(self.pos_embed, L)
        else:
            pos_embed = self.pos_embed[:, :L + 1, :]
        pos_embed = pos_embed.expand(x.shape[0], -1, -1)
        x = x + pos_embed

        # Apply transformer encoder blocks.
        x = self.ema.model(src_tokens=None, token_embeddings=x, encoder_padding_mask=pad_mask, return_all_hiddens=True)['encoder_states']

        return x

    def forward(self, src, indice_lsit = None, pad_mask=None, mask_ratio = 0.2, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        # model forward in online mode (student)
        x, mask = self.forward_encoder(src, indice_lsit, mask_ratio, pad_mask=pad_mask)  # only compute loss on masked patches

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.forward_latent(src, pad_mask=pad_mask)

            #y = self.ema.model(src_tokens=None, token_embeddings=trg)['encoder_states']  # fetch the last transformer layers outputs
            y = y[-self.cfg.model.average_top_k_layers:]  # take the last k transformer layers
            # Follow the same layer normalization procedure for text and vision
            # if self.modality in ['vision', 'text']:
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)
            if self.cfg.model.normalize_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])
        x = x[:,1:,:][mask]
        y = y[:,1:,:][mask]
        
        x = self.regression_head(x)

        return x, y

# Example usage:
if __name__ == "__main__":
    import omegaconf
    # Dummy one-hot encoded genome segments: batch of 2, sequence length 1000, 4 channels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randint(0, 2, (11, 768, 4)).float().to(device)
    indices_list = [1, 5, 9, 10]  # random indices
    indices_list = [indices_list for _ in range(1)]
    # transform to tensor
    indices_list = torch.tensor(indices_list, device=device)
    x = x.unsqueeze(0)
    cfg_path = "config.yaml"
    cfg = omegaconf.OmegaConf.load(cfg_path)
    print(x.shape)
    model = Data2Vec(cfg,  # length of the genome segment
        patch_size=768,
        embed_dim=1536).to(device)
    # support mixed precision training with autocast
    with torch.cuda.amp.autocast():
        x, y = model(x, indices_list, mask_ratio=0.2)
    print(x.shape, y.shape)