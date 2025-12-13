import argparse


def get_finetune_params():

    parser = argparse.ArgumentParser(description='Finetune on downstream tasks')

    def _str2bool(v: str) -> bool:
        v = v.lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")

    # task settings
    parser.add_argument('--task_cfg_path',  type=str, default='task_configs/default.yaml', help='Path to the task configuration file')
    parser.add_argument('--exp_name',       type=str, default='', help='Experiment name')

    # input data settings
    parser.add_argument('--fold_data_dir',  type=str, default='data/folds', help='Directory containing fold files like train_df_{fold}.csv, train_embs_{fold}.npy, test_df_{fold}.csv, test_embs_{fold}.npy')
    parser.add_argument('--train_df_tpl',   type=str, default='train_df_{fold}.csv', help='Filename template (relative to --fold_data_dir) for the train CSV')
    parser.add_argument('--train_embs_tpl', type=str, default='train_embs_{fold}.npy', help='Filename template (relative to --fold_data_dir) for the train embeddings .npy')
    parser.add_argument('--test_df_tpl',    type=str, default='test_df_{fold}.csv', help='Filename template (relative to --fold_data_dir) for the test CSV')
    parser.add_argument('--test_embs_tpl',  type=str, default='test_embs_{fold}.npy', help='Filename template (relative to --fold_data_dir) for the test embeddings .npy')

    parser.add_argument('--genome_onehot_dir',        type=str, default='data/genome_onehots', help='Directory containing per-genome one-hot arrays as .npy files (named <genome_id>.npy)')
    parser.add_argument('--genome_split_indices_dir', type=str, default='data/genome_split_indices', help='Directory containing per-genome split indices as .npy files (named <genome_id>.npy)')
    parser.add_argument('--kmer_dir',                 type=str, default='data/5mers_dic', help='Directory containing per-genome k-mer counts as .npy files (named <genome_id>.npy)')

    parser.add_argument('--max_tiles',      type=int, default=1000000, help='Maximum number of tiles')

    # model settings
    parser.add_argument('--pretrained_checkpoint', type=str, default='', help='Path to a GenomeData2Vec checkpoint (.pth) with a top-level key \"model\"')
    parser.add_argument('--freeze_pretrained_encoder', type=_str2bool, default=True, help='Whether to freeze the pretrained genome encoder (true/false)')
 
    # training settings
    parser.add_argument('--seed',           type=int, default=0, help='Random seed')
    parser.add_argument('--epochs',         type=int, default=5, help='Number of training epochs')
    parser.add_argument('--warmup_epochs',  type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--batch_size',     type=int, default=1, help='Current version only supports batch size of 1')
    parser.add_argument('--lr',             type=float, default=None, help='Learning rate')
    parser.add_argument('--blr',            type=float, default=1e-4, help='Base learning rate, will caculate the learning rate based on batch size')
    parser.add_argument('--min_lr',         type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--lr_scheduler',   type=str, default='cosine', help='Learning rate scheduler', choices=['cosine', 'fixed'])
    parser.add_argument('--gc',             type=int, default=32, help='Gradient accumulation')
    parser.add_argument('--folds',          type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--optim',          type=str, default='adamw', help='Optimizer', choices=['adam', 'adamw'])
    parser.add_argument('--optim_wd',       type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout',        type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--val_r',          type=float, default=0.1, help='Ratio of data used for validation')
    parser.add_argument('--model_select',   type=str, default='val', help='Criteria for choosing the model checkpoint', choices=['val', 'last_epoch'])
    parser.add_argument('--save_dir',       type=str, default='', help='Save directory')
    parser.add_argument('--num_workers',    type=int, default=10, help='Number of workers')
    parser.add_argument('--report_to',      type=str, default='tensorboard', help='Logger used for recording', choices=['wandb', 'tensorboard', 'none'])
    parser.add_argument('--fp16',           action='store_true', default=False, help='Fp16 training')
    parser.add_argument('--weighted_sample',action='store_true', default=False, help='Weighted sampling')

    return parser.parse_args()
