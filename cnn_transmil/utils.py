import os
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
import torch.optim as optim

from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)
        

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def seed_torch(device, seed=7):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_exp_code(args):
    '''Get the experiment code for the current run.'''
    # set up the model code
    model_code = 'eval'
    if len(args.pretrained) > 0:
        model_code += '_pretrained'
    if args.freeze:
        model_code += '_freeze'
        
    # set up the task code
    task_code = args.task
    if args.pat_strat:
        task_code += '_pat_strat'

    # set up the experiment code
    exp_code = '{model_code}_{task_code}'

    return model_code, task_code, exp_code.format(model_code=model_code, task_code=task_code)


def pad_tensors(embs):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/tree/main
    # ------------------------------------------------------------------------------------------
    '''Pad the tensors to the same size.'''
    max_len = max([e.shape[0] for e in embs])
    pad_embs = torch.zeros(len(embs), max_len, embs[0].shape[1])
    for i, e in enumerate(embs):
        pad_embs[i, :e.shape[0]] = e

    return pad_embs

    


def slide_collate_fn(samples):
    '''Separate the inputs and targets into separate lists
    Return value {imgs: [N, L, 256, 384], pad_mask: [N, L]}'''
    emb_list = [s['embs'] for s in samples]
    label_list = [s['labels'] for s in samples]
    labels = torch.stack(label_list)
    pad_embs = pad_tensors(emb_list)
    
    data_dict = {'embs': pad_embs, 'labels': labels}

    return data_dict


def get_splits(df: pd.DataFrame, 
               val_r: float=0.1, test_r: float=0.2, 
               fold: int=0, 
               split_dir: str='', 
               fetch_splits: bool=True, 
               prop: int=1, 
               split_key='slide_id', 
               **kwargs) -> Tuple[List[str], List[str], List[str]]:
    '''Get the splits for the dataset. The default train/val/test split is 70/10/20.'''
    # get the split names
    files = os.listdir(split_dir)
    train_name, val_name, test_name = f'train_{fold}.csv', f'val_{fold}.csv', f'test_{fold}.csv'
    # check split_key is in the columns
    assert split_key in df.columns, f'{split_key} not in the columns of the dataframe'
    # make sure the dataset exists, otherwise create new datasets
    if train_name not in files or val_name not in files or test_name not in files or not fetch_splits:
        samples = df.drop_duplicates(split_key)[split_key].to_list()
        train_samples, temp_samples = train_test_split(samples, test_size=(val_r + test_r), random_state=fold)
        if val_r > 0:
            val_samples, test_samples = train_test_split(temp_samples, test_size=(test_r / (val_r + test_r)), random_state=fold)
        else:
            val_samples, test_samples = [], temp_samples
        train_data = df[df[split_key].isin(train_samples)]
        val_data = df[df[split_key].isin(val_samples)]
        test_data = df[df[split_key].isin(test_samples)]

        # sample the training data
        if prop > 0:
            train_data = train_data.sample(frac=prop, random_state=fold).reset_index(drop=True)
        # save datasets
        train_data.to_csv(os.path.join(split_dir, train_name))
        val_data.to_csv(os.path.join(split_dir, val_name))
        test_data.to_csv(os.path.join(split_dir, test_name))
    # load the dataframe
    train_splits = pd.read_csv(os.path.join(split_dir, train_name))[split_key].to_list()
    val_splits = pd.read_csv(os.path.join(split_dir, val_name))[split_key].to_list()
    test_splits = pd.read_csv(os.path.join(split_dir, test_name))[split_key].to_list()

    return train_splits, val_splits, test_splits


def get_loader(train_dataset, val_dataset, test_dataset, 
               task_config, weighted_sample=False, 
               batch_size=1, num_workers=10, seed=0, 
               **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Get the dataloader for the dataset.'''

    train_sampler = RandomSampler(train_dataset)

    # set up generator and worker_init_fn
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # if it's the sequence based model, we use the slide collate function to pad
    train_loader = DataLoader(train_dataset, \
                            num_workers=num_workers, \
                            batch_size=batch_size, sampler=train_sampler, \
                            generator=g, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(val_dataset), \
                            worker_init_fn=seed_worker) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, \
                            num_workers=num_workers, \
                            batch_size=1, sampler=SequentialSampler(test_dataset), \
                            worker_init_fn=seed_worker) if test_dataset is not None else None

    return train_loader, val_loader, test_loader

def adjust_learning_rate(optimizer, epoch, args):
    # ------------------------------------------------------------------------------------------
    # References:
    # mae: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
    # ------------------------------------------------------------------------------------------
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def get_loss_function(task_config: dict):
    '''Get the loss function based on the task configuration.'''
    task_setting = task_config.get('setting', 'multi_class')
    if task_setting == 'multi_label':
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif task_setting == 'multi_class' or task_setting == 'binary':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return loss_fn


def get_records_array(record_len: int, n_classes) -> dict:
    '''Get the records array based on the task configuration.'''
    record = {
        'prob': np.zeros((record_len, n_classes)) if n_classes >= 2 else np.zeros(record_len, dtype=np.float32),
        'label': np.zeros((record_len, n_classes)) if n_classes >= 2 else np.zeros(record_len, dtype=np.float32),
        'loss': 0.0,
    }
    return record


class Monitor_Score:
    # ------------------------------------------------------------------------------------------
    # References:
    # MCAT: https://github.com/mahmoodlab/MCAT/blob/master/utils/core_utils.py
    # ------------------------------------------------------------------------------------------
    def __init__(self):
        self.best_score = None

    def __call__(self, val_score, model, ckpt_name:str='checkpoint.pt'):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def log_writer(log_dict: dict, step: int, report_to: str='tensorboard', writer=None):
    '''Log the dictionary to the writer.'''
    if report_to == 'tensorboard':
        for k, v in log_dict.items():
            writer.add_scalar(k, v, step)
    elif report_to == 'wandb':
        writer.log(log_dict, step=step)
    else:
        raise NotImplementedError