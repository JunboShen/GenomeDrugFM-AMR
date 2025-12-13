import os
import torch
import pandas as pd
import numpy as np

from training import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_loader

from datasets.dataset import DatasetForTasks
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    args = get_finetune_params()
    print(args)

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # set the experiment save directory
    if not args.save_dir:
        args.save_dir = os.path.join("results", args.exp_name or args.task)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # use the dataset
    DatasetClass = DatasetForTasks

    # set up the results dictionary
    results = {}

    # start cross validation
    for fold in range(args.folds):
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)

        train_df_path = os.path.join(args.fold_data_dir, args.train_df_tpl.format(fold=fold))
        train_embs_path = os.path.join(args.fold_data_dir, args.train_embs_tpl.format(fold=fold))
        test_df_path = os.path.join(args.fold_data_dir, args.test_df_tpl.format(fold=fold))
        test_embs_path = os.path.join(args.fold_data_dir, args.test_embs_tpl.format(fold=fold))

        train_data = pd.read_csv(train_df_path, dtype=str)
        train_embs = np.load(train_embs_path)
        test_data = pd.read_csv(test_df_path, dtype=str)
        test_embs = np.load(test_embs_path)
        
        #split train data into train and val, 0.125 for val, 0.875 for train
        train_ids, val_ids = train_test_split(train_data.index, test_size=args.val_r, random_state=42)
        val_data = train_data.iloc[val_ids]
        val_embs = train_embs[val_ids]
        train_data = train_data.iloc[train_ids]
        train_embs = train_embs[train_ids]

        train_data, val_data, test_data = (
            DatasetForTasks(train_data, train_embs, args.genome_onehot_dir, args.kmer_dir, max_tiles=args.max_tiles),
            DatasetForTasks(val_data, val_embs, args.genome_onehot_dir, args.kmer_dir, max_tiles=args.max_tiles),
            DatasetForTasks(test_data, test_embs, args.genome_onehot_dir, args.kmer_dir, max_tiles=args.max_tiles),
        )
       #test DatasetClass
        # print("train_data", train_data)
        # print(train_data.__getitem__(0)["embs"].shape)
        # print(train_data.__getitem__(0)["labels"].shape)
        # exit()
        
        args.n_classes = 2 # get the number of classes
        print('Number of classes: {}'.format(args.n_classes))
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)

        # update the results
        records = {'val': val_records, 'test': test_records}
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key:
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])

    # save the results into a csv file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')
