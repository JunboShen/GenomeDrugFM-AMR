import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class DatasetForTasks(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        embs: np.ndarray,
        root_path: str,
        kmer_root_path: str,
        shuffle_tiles: bool = False,
        max_tiles: int = 1000,
    ):

        self.df = df.reset_index(drop=True)
        self.embs = embs
        self.root_path = root_path
        self.shuffle_tiles = shuffle_tiles
        self.max_tiles = max_tiles
        self.kmer_root_path = kmer_root_path

        # Optional: Validate that the number of embeddings matches the number of samples
        if len(self.df) != self.embs.shape[0]:
            raise ValueError(
                f"Mismatch between number of samples in DataFrame ({len(self.df)}) and embeddings ({self.embs.shape[0]})."
            )

        #read genome ids as strings
        self.genome_ids = self.df["genome_id"].values.astype(str)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the embeddings and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'embs' (torch.Tensor): Tensor of embeddings with shape (num_tiles, embedding_dim).
                - 'labels' (torch.Tensor): Tensor containing the label for the sample.
        """
        # Retrieve embeddings for the sample
        embs = self.embs[idx].copy()  # Shape: (num_tiles, embedding_dim)
    
        genome_embs = np.load(os.path.join(self.root_path, str(self.genome_ids[idx]) + '.npy'))
        #genome_seg_ids = np.load(os.path.join(self.split_indice_path, str(self.genome_ids[idx]) + '.npy'))
      
        if self.max_tiles > 0 and genome_embs.shape[0] > self.max_tiles:
            #randomly sample max_tiles number of tiles
            genome_embs = genome_embs[np.random.choice(genome_embs.shape[0], self.max_tiles, replace=False)]
        kmer_dict = np.load(os.path.join(self.kmer_root_path, str(self.genome_ids[idx])  + '.npy'), allow_pickle=True).item()
        genome_kmer = list(kmer_dict.values())
        # Convert kmers to torch.Tensor
        genome_kmer_tensor = torch.tensor(genome_kmer, dtype=torch.float32) # Shape: (1, embedding_dim)
        #embs_seg_ids = torch.tensor(genome_seg_ids, dtype=torch.int8)
        # Convert embeddings to torch.Tensor
        embs_tensor = torch.tensor(embs, dtype=torch.float32) # Shape: (embedding_dim)
        genome_embs_tensor = torch.tensor(genome_embs, dtype=torch.float32) # Shape: (num_tiles, embedding_dim)
        
        #concate embs to each tile of genome_embs
        
        #embs_tensor = torch.cat([genome_embs_tensor, embs_tensor.repeat(genome_embs_tensor.shape[0], 1)], dim=1) 

        # Retrieve and process the label
        # Assuming 'resistant_phenotype' is the target:
        #   'resistant' -> 1
        #   'susceptible' -> 0
        phenotype = self.df.iloc[idx]['resistant_phenotype']
        label = 1 if phenotype.lower() == 'resistant' else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {'embs': genome_embs_tensor, 'drug_emb': embs_tensor, 'labels': label_tensor, 'genome_kmer': genome_kmer_tensor}
