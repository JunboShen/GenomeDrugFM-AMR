# GenomeDrugFM-AMR

This repository contains the codes (pretraining, finetuning, inference utilities, and baselines) for <a href="https://drive.google.com/file/d/1f0zGqQbm5RENCq3H6ZfULcINfY_9uXmd/view">
  <img src="https://img.shields.io/badge/PDF-GenomeDrugFM_AMR-red?style=flat&logo=adobeacrobatreader" alt="PDF" />
</a>.

The implementation focuses on predicting AMR phenotypes across different bacterial strains and drugs.


## Model Overview

![GenomeDrugFM-AMR architecture](assets/arch_genome_drug_FM.png)

## Repository Layout

- `pretraining/data2vec/`: GenomeData2Vec (self-supervised) pretraining code.
  - Entry scripts: `pretraining/data2vec/run_train_new.sh`, `pretraining/data2vec/run_train_new_continue.sh`
  - Python entrypoints: `pretraining/data2vec/train_new.py`, `pretraining/data2vec/train_new_continue.py`
- `train/`: finetuning code for supervised AMR prediction.
  - Entry point: `train/main.py`
- `CNN/`, `cnn_transmil/`: baseline models used in the paper.

Large datasets, embeddings, checkpoints, and experiment outputs are not committed to git (see `.gitignore`).

## Setup

Tested with Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Expectations (Not Included)

The training scripts expect preprocessed artifacts on disk:

- **Genome one-hot segments**: per-genome `.npy` files named `<genome_id>.npy`
- **Segment index arrays**: per-genome `.npy` files named `<genome_id>.npy`
- **Drug embeddings**: `.npy` arrays aligned with your train/test CSV rows
- **5-mer counts**: per-genome `.npy` dictionary files named `<genome_id>.npy` (values used as a 512-d vector)

You can place these under `data/` (ignored by git) or point the scripts to any location via flags.

Optional (data prep): `pretraining/data2vec/get_one_hots_data2vec.py` builds one-hot genome segments from per-genome FASTA files.

## Pretraining (GenomeData2Vec)

```bash
CUDA_VISIBLE_DEVICES=0 \
TRAIN_CSV=./data/PATRIC_genomes_AMR_pretrain.csv \
ONEHOT_DIR=./data/one_hots_data2vec_new \
SPLIT_INDICES_DIR=./data/one_hots_split_indices_data2vec_new \
OUTPUT_DIR=./outputs/pretrain_patch4096 \
bash pretraining/data2vec/run_train_new.sh
```

Continue training from an existing checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
RESUME_CHECKPOINT=./outputs/pretrain_patch4096/checkpoint.pth \
OUTPUT_DIR=./outputs/pretrain_patch4096_more \
bash pretraining/data2vec/run_train_new_continue.sh
```

Weights & Biases is optional:

- Set `REPORT_TO=none` to disable logging
- Or set `WANDB_MODE=offline` to log locally

## Finetuning (Supervised AMR Prediction)

The finetuning code uses fold files under `--fold_data_dir`:

- `train_df_{fold}.csv`, `train_embs_{fold}.npy`
- `test_df_{fold}.csv`, `test_embs_{fold}.npy`

Example:

```bash
cd train
python main.py \
  --task_cfg_path task_configs/default.yaml \
  --fold_data_dir ../data/folds \
  --genome_onehot_dir ../data/genome_onehots \
  --genome_split_indices_dir ../data/genome_split_indices \
  --kmer_dir ../data/5mers_dic \
  --pretrained_checkpoint ../outputs/pretrain_patch4096/checkpoint.pth \
  --save_dir ../outputs/finetune_run
```

## Baselines

- CNN baseline: `CNN/main.py`
- CNN-TransMIL baseline: `cnn_transmil/main.py`

Both baseline entrypoints now accept `--fold_data_dir`, `--genome_onehot_dir`, and `--kmer_dir` similarly to `train/main.py`.

## Acknowledgements

- This repo uses Microsoft `torchscale` (LongNet implementation) for the genome encoder backbone: https://github.com/microsoft/torchscale
- The self-supervised pretraining pipeline is based on the Data2Vec idea (Data2Vec / Data2Vec 2.0): https://ai.meta.com/research/publications/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/
- The MIL aggregator follows the TransMIL architecture: https://arxiv.org/abs/2106.00908
- Parts of the codes for training LongNet-style encoder reference Prov-GigaPath: https://huggingface.co/prov-gigapath/prov-gigapath

## Citation

If you use this code, please cite the paper:

```
TBD
```

If you need access to data, pretrained/fine-tuned checkpoints used in our experiments, please contact the repository owner via email.
