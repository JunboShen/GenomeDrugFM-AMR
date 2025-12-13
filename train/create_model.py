import os
import sys
from pathlib import Path

import torch
from torch import nn

from model import TransMIL

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA2VEC_DIR = _REPO_ROOT / "pretraining" / "data2vec"
sys.path.insert(0, str(_DATA2VEC_DIR))

from genome_encoder.genome_bert import mae_genome_base  # noqa: E402


def load_pretrained_weights(model: nn.Module, pretrained_weights: str) -> None:
    if not pretrained_weights:
        return
    if not os.path.isfile(pretrained_weights):
        raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights}")

    checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {pretrained_weights} (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")

class ClassificationHead(nn.Module):
   
    def __init__(
        self,
        n_classes=2,
        model_arch="bert",
        pretrained="",
        freeze=True,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        # self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        # self.feat_dim = len(self.feat_layer) * latent_dim * 2
        
        if model_arch == "bert":
            self.pretrained_encoder = mae_genome_base(seq_length=6400000, patch_size=768)
        else:
            raise ValueError("Invalid model architecture")
        #print weights
        # for name, param in self.pretrained_model.named_parameters():
        #     print(name, param)
        #     break

        # load pretrained weights
        if pretrained:
            load_pretrained_weights(self.pretrained_encoder, pretrained)
         #print weights
        # for name, param in self.slide_encoder.named_parameters():
        #     print(name, param)
        #     break

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained encoder")
            for name, param in self.pretrained_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        print('Number of classes: ', n_classes)
        # setup the classifier
        self.mil_classifier = TransMIL(input_dim=768+768, n_classes= n_classes, input_dim_kmer=512)

    def forward(self, genome_embs_tensor, embs_tensor, embs_seg_ids, genome_kmer_tensor):

        # inputs: [N, L, D]

        # forward encoder
        latent = self.pretrained_encoder.forward(genome_embs_tensor, embs_seg_ids, no_masking=True)
        #remove embs in latent corresponding to emb_seg_ids
        L = latent.shape[1]
        protected = embs_seg_ids[0].to(device=latent.device, dtype=torch.long).flatten()
        protected = protected[(protected >= 0) & (protected < L)]
        keep_mask = torch.ones(L, dtype=torch.bool, device=latent.device)
        if protected.numel() > 0:
            keep_mask[protected] = False
        latent = latent[:, keep_mask]
        # Reshape embs_tensor to (B, 1, 512) and repeat to match latent's sequence length
        embs_tensor_expanded = embs_tensor.unsqueeze(1).repeat(1, latent.shape[1], 1)
        # Concatenate along the feature dimension to get (B, L, 1280)
        h = torch.cat([latent, embs_tensor_expanded], dim=2)    
        logits = self.mil_classifier(h, genome_kmer_tensor)
        return logits

def get_model(**kwargs):
    model = ClassificationHead(**kwargs)
    return model
