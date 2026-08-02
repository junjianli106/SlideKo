from pathlib import Path

#---->
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

import os
from pytorch_lightning import loggers as pl_loggers


def load_loggers(cfg):
    'log_path: /multi_scale/logs/ config: “BLCA/AMIL.yaml” '

    '''
    os.path.split(suvival/BLCA/AMIL/AMIL.yaml)-->('Suvival/BLCA/AMIL', 'AMIL.yaml')
    suvival/BLCA/AMIL/AMIL.yaml
    os.path.split(cfg.config)[0]: BLCA
    os.path.split(cfg.config)[1]: AMIL
    os.path.split(cfg.config)[2][:-5]: AMIL
    '''
    log_path = cfg.General.log_path
    os.makedirs(log_path, exist_ok=True)

    task = os.path.split(cfg.config)[0].split('/')[0]  # BLCA/AMIL/AMIL.yaml -> BLCA
    dataset_name = os.path.split(cfg.config)[0].split('/')[1]  # BLCA/AMIL/AMIL.yaml -> BLCA
    model_name = os.path.split(cfg.config)[0].split('/')[2]  # AMIL
    version_identifier = os.path.split(cfg.config)[1][:-5] # AMIL.yaml -> AMIL

    cfg.log_path = os.path.join(log_path, task, dataset_name, model_name, version_identifier, f'Shot{cfg.Data.n_shot}', f'fold{cfg.Data.fold}')

    print(f'---->Log dir: {cfg.log_path}')

    # ---->TensorBoard
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(log_path, task, dataset_name, model_name, version_identifier),
        name=f'Shot{cfg.Data.n_shot}', version=f'fold{cfg.Data.fold}',
        log_graph=True, default_hp_metric=False
    )

    # ---->CSV
    csv_logger = pl_loggers.CSVLogger(
        os.path.join(log_path, task, dataset_name, model_name, version_identifier),
        name=f'Shot{cfg.Data.n_shot}', version=f'fold{cfg.Data.fold}'
    )

    return [tb_logger, csv_logger]

def load_log_path(cfg):
    log_root = cfg.General.log_path
    os.makedirs(log_root, exist_ok=True)

    task = os.path.split(cfg.config)[0].split('/')[0]
    dataset_name = os.path.split(cfg.config)[0].split('/')[1]
    model_name = os.path.split(cfg.config)[0].split('/')[2]

    version_identifier = os.path.split(cfg.config)[1][:-5]

    log_path = os.path.join(
        log_root,
        task,
        dataset_name,
        model_name,
        version_identifier,
        f"Shot{cfg.Data.n_shot}",
    )
    print(f"---->Log dir: {log_path}")

    return log_path

import logging

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, getattr(logger, level))

    return logger

# ---->Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg):
    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    os.makedirs(output_path, exist_ok=True)

    # Determine the monitor metric based on the task
    monitor_metric = 'val_acc'

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='max' if monitor_metric == 'val_acc' else 'min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train':
        Mycallbacks.append(ModelCheckpoint(
            monitor=monitor_metric,
            dirpath=output_path,
            filename='{epoch:02d}-{' + monitor_metric + ':.4f}',
            verbose=True,
            save_last=True,
            save_top_k=1,
            mode='max' if monitor_metric == 'val_acc' else 'min',
            save_weights_only=True
        ))

    return Mycallbacks

#---->
import torch.nn as nn
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

#---->val loss
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def cross_entropy_torch(x, y):
    # 确保 x 和 y 都是 2D 张量
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 0:
        y = y.unsqueeze(0)

    # 确保 y 是长整型
    y = y.long()

    # 对整个批次应用 softmax
    x_softmax = F.softmax(x, dim=-1)

    # 获取每个样本对应的概率
    correct_probs = x_softmax[torch.arange(x.size(0)), y]

    # 计算对数并求平均
    log_probs = torch.log(correct_probs)
    loss = -torch.mean(log_probs)

    return loss

