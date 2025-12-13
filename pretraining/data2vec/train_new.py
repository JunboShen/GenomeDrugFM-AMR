import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import omegaconf
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import utils
from data2vec_new import Data2Vec
from datasets.dataset import DatasetPretrain
from params import get_pretrain_params
from torch.utils.data import RandomSampler

def _resolve_path(path: str) -> str:
    if not path:
        return path
    return os.path.expanduser(path)


def train(args, cfg, wandb_run=None):

    utils.fix_random_seeds(args.seed)

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    DatasetClass = DatasetPretrain

    # ============ preparing data ... ============
    train_csv = _resolve_path(args.train_csv) or os.environ.get("TRAIN_CSV") or "PATRIC_genomes_AMR_pretrain.csv"
    onehot_dir = _resolve_path(args.onehot_dir) or os.environ.get("ONEHOT_DIR") or "one_hots_data2vec_new"
    split_indices_dir = (
        _resolve_path(args.split_indices_dir) or os.environ.get("SPLIT_INDICES_DIR") or "one_hots_split_indices_data2vec_new"
    )

    train_data = pd.read_csv(train_csv, dtype=str)

    dataset = DatasetClass(df = train_data,
        root_path = onehot_dir,
        split_indice_path = split_indices_dir,
        shuffle_tiles = False,
        max_tiles = 1000000,
        )
    # get the dataloader
    sampler = RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=utils.collate_pretrain,
    )
    print(f"Data loaded: there are {len(dataset)} genomes.")

    # ============ building networks ... ============
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = mae_genome_base(seq_length=6400000, patch_size=768).to(args.device)
    model = Data2Vec(cfg,  # length of the genome segment
        patch_size=args.patch_size,
        embed_dim=args.embed_dim).to(args.device)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs

    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")
    criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.criterion.loss_beta)
    criterion.to(args.device)
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    resume_checkpoint = _resolve_path(args.resume_checkpoint) or os.path.join(args.output_dir, "checkpoint.pth")
    utils.restart_from_checkpoint(
        resume_checkpoint,
        run_variables=to_restore,
        model=model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model,
                                      data_loader, optimizer, lr_schedule, wd_schedule,
                                      epoch, fp16_scaler, criterion, args)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_backup.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(model, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch,
                    fp16_scaler, criterion, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        embs, embs_seg_ids, pad_mask = batch['embs'], batch['embs_seg_ids'],batch['pad_mask']
        embs = embs.to(args.device, non_blocking=True)
        embs_seg_ids = embs_seg_ids.to(args.device, non_blocking=True)
        pad_mask = pad_mask.to(args.device, non_blocking=True)
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        with torch.cuda.amp.autocast(fp16_scaler is not None):

            x, y = model(embs, embs_seg_ids,  pad_mask=pad_mask, mask_ratio=0.2)
            loss = criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
            
        if not math.isfinite(loss.item()):
            print(embs.shape, embs_seg_ids.shape)
            print(x.shape, y.shape)
            print(x)
            print(y)
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        model.ema_step()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if wandb_run is not None:
            wandb_run.log(
                {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "wd": optimizer.param_groups[0]["weight_decay"]}
            )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_pretrain_params()
    wandb_run = None
    if args.report_to == "wandb":
        import wandb

        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # calculate the time for training
    start = time.time()
    train(args, cfg, wandb_run=wandb_run)
    end = time.time()
    print("Time taken for training: ", end - start)
