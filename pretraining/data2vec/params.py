import argparse

import utils
def get_pretrain_params():

    parser = argparse.ArgumentParser('genomeData2Vec', add_help=False)

    # Data / I/O
    parser.add_argument(
        "--train_csv",
        type=str,
        default="",
        help="CSV file containing a 'genome_id' column for pretraining.",
    )
    parser.add_argument(
        "--onehot_dir",
        type=str,
        default="",
        help="Directory containing per-genome one-hot arrays as .npy files (named <genome_id>.npy).",
    )
    parser.add_argument(
        "--split_indices_dir",
        type=str,
        default="",
        help="Directory containing per-genome segment indices as .npy files (named <genome_id>.npy).",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="Optional path to a checkpoint to resume from (overrides <output_dir>/checkpoint.pth).",
    )

    # Logging
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "none"],
        help="Logging backend.",
    )
    parser.add_argument("--wandb_project", type=str, default="genomeData2Vec", help="Weights & Biases project name.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode.",
    )

    # Model parameters
    parser.add_argument('--config', default="config.yaml", type=str, help='Path to the config file.')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
   
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
  
    parser.add_argument('--output_dir', default="./test", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.') #new 5 original 1
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument('--gc',             type=int, default=32, help='Gradient accumulation')
  
    parser.add_argument('--dropout',        type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate')

    # Data2Vec architecture
    parser.add_argument("--patch_size", type=int, default=4096, help="Genome segment patch size.")
    parser.add_argument("--embed_dim", type=int, default=1536, help="Encoder embedding dimension.")

    return parser.parse_args()
