from __future__ import annotations

from datetime import datetime

import os
from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
from typing import Callable

# import traceback
# import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb

from CustomGeoGNN.model import PredModel
from CustomGeoGNN.module.loss_fn import uncertainty_loss, UncertaintyLoss, rmse_loss

MEAN = None
STD = None
MIN = None
MAX = None

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

qm9_conversion = {
    'mu': 1.,
    'alpha': 1.,
    'homo': HAR2EV,
    'lumo': HAR2EV,
    'gap': HAR2EV,
    'r2': 1.,
    'zpve': HAR2EV,
    'u0': HAR2EV,
    'u298': HAR2EV,
    'h298': HAR2EV,
    'g298': HAR2EV,
    'cv': 1.,
    'u0_atom': KCALMOL2EV,
    'u298_atom': KCALMOL2EV,
    'h298_atom': KCALMOL2EV,
    'g298_atom': KCALMOL2EV,
}

ROOT = os.getcwd()

def load_tasks(tasks_path: str) -> dict:
    tasks_path = os.path.join('configs', tasks_path)

    with open(os.path.join(ROOT, tasks_path), 'r') as f:
        tasks: dict = json.load(f)
    
    return tasks

TASKS = load_tasks('tasks.json')

def load_config(config_path: str) -> dict:
    config_path = os.path.join('configs', config_path)

    with open(os.path.join(ROOT, config_path), 'r') as f:
        config: dict = json.load(f)

    print_config(config)
    
    return config

def print_config(config: dict):
    for key in config:
        print(f'[GeoGNNModel] {key}: {config[key]}')

def load_dataset(data_path: str, raw_data_path: str, dataset: str, tasks: list) -> tuple[list, np.ndarray, list, np.ndarray]:
    train_path = Path(os.path.join(ROOT, data_path, dataset, 'train'))
    val_path = Path(os.path.join(ROOT, data_path, dataset, 'val'))

    global MEAN
    global STD
    global MIN
    global MAX
    global qm9_conversion

    if dataset in ['qm7', 'qm8', 'qm9']:
        y_path = os.path.join(ROOT, raw_data_path, dataset, f'{dataset}.sdf.csv')
        y_all = pd.read_csv(y_path)
        y_all = y_all[tasks]
    elif dataset == 'esol':
        y_path = os.path.join(ROOT, raw_data_path, dataset, 'delaney-processed.csv')
        y_all = pd.read_csv(y_path)
        y_all = y_all[[y_all.columns[-2]]]
    elif dataset == 'freesolv':
        y_path = os.path.join(ROOT, raw_data_path, dataset, 'SAMPL.csv')
        y_all = pd.read_csv(y_path)
        y_all = y_all[['expt']]
    elif dataset == 'lipophilicity':
        y_path = os.path.join(ROOT, raw_data_path, dataset, 'Lipophilicity.csv')
        y_all = pd.read_csv(y_path)
        y_all = y_all[['exp']]
    
    MEAN = torch.tensor(y_all.mean().to_list()).view(-1, 1).cuda()
    STD = torch.tensor(y_all.std(ddof=0).to_list()).view(-1, 1).cuda()
    MIN = torch.tensor(y_all.min().to_list()).view(-1, 1).cuda()
    MAX = torch.tensor(y_all.max().to_list()).view(-1, 1).cuda()

    if dataset == 'qm9':
        qm9_conversion = torch.tensor(list(map(qm9_conversion.get, tasks))).view(-1, 1).cuda()
        MEAN = MEAN * qm9_conversion
        STD = STD * qm9_conversion
        MIN = MIN * qm9_conversion
        MAX = MAX * qm9_conversion

    with open(train_path / 'train_data.pkl', 'rb') as f:
        graph_list_train = pickle.load(f)
    y_list_train = pd.read_csv(train_path / 'train_values.csv')
    y_list_train = y_list_train[tasks].to_numpy()

    # graph_list_train = graph_list_train[:20000]
    # y_list_train = y_list_train[:20000]
    
    with open(val_path / 'val_data.pkl', 'rb') as f:
        graph_list_val = pickle.load(f)
    y_list_val = pd.read_csv(val_path / 'val_values.csv')
    y_list_val = y_list_val[tasks].to_numpy()

    return graph_list_train, y_list_train,\
           graph_list_val, y_list_val

def loss(y_pred: torch.Tensor, y: torch.Tensor, loss_fn: Callable, mode: str='train'):
    y_pred = y_pred.squeeze().cuda()
    y = y.squeeze().cuda()
    # if loss_pe is not None:
    #     loss_pe = loss_pe.cuda()

    # GAP = abs(HOMO - LUMO)
    y_pred[2] = torch.abs(y_pred[0] - y_pred[1])

    # print(y_pred, y_pred.shape)
    # print(y, y.shape)

    if mode == 'train':
        # mean of (num_nodes, num_tasks) -> 1
        loss_batch_mean = loss_fn(y_pred.float(), y.float(), reduction='mean')
        # loss_batch_mean = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='mean')
        # loss_batch_mean = F.mse_loss(y_pred.float(), y.float(), reduction='mean')
        # loss_batch_mean = F.l1_loss(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        loss_tasks = loss_fn(y_pred.float(), y.float(), reduction='none')
        # loss_tasks = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='none')
        # loss_tasks = F.mse_loss(y_pred.float(), y.float(), reduction='none')
        # loss_tasks = F.l1_loss(y_pred.float(), y.float(), reduction='none')

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
        
        # sum of (num_tasks, 1) -> 1
        # loss_batch_sum = torch.sum(loss_tasks)
        loss_batch_sum = loss_fn(y_pred.float(), y.float(), reduction='sum')
        # loss_batch_sum = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='sum')
        # loss_batch_sum = F.mse_loss(y_pred.float(), y.float(), reduction='sum')
        # loss_batch_sum = F.l1_loss(y_pred.float(), y.float(), reduction='sum')

        print(loss_tasks, loss_batch_sum, loss_batch_mean)#, loss_pe)

        # return loss_batch_sum, loss_batch_mean, loss_tasks

        # loss_train = loss_batch_sum + 0.1 * loss_pe
        loss_train = loss_batch_mean

        return loss_train, loss_batch_sum, loss_batch_mean, loss_tasks
    elif mode == 'test':
        # mean of (num_nodes, num_tasks) -> 1
        loss_batch_mean = loss_fn(y_pred.float(), y.float(), reduction='mean')
        # loss_batch_mean = F.l1_loss(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        loss_tasks = loss_fn(y_pred.float(), y.float(), reduction='none')
        # loss_tasks = F.l1_loss(y_pred.float(), y.float(), reduction='none')

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
            # loss_batch_mean = torch.min(loss_batch_mean, loss_tasks)

        print(loss_tasks, loss_batch_mean)

        return loss_batch_mean, loss_tasks

def train_one_epoch(epoch: int, model, batch_size: int, writer: SummaryWriter, data: tuple[list, np.ndarray], *args):
    graph_list_train, y_list_train = data

    scaler, optimizer, lr_scheduler = args

    train_size = len(graph_list_train)

    train_loss = 0.0
    num_train_batches = train_size // batch_size

    # batch_indices = np.arange(train_size)
    # np.random.shuffle(batch_indices)

    model.train()
    for batch_idx in tqdm(range(num_train_batches), desc='Training'):
        batch_graph_list = []
        batch_y_list = []
        # batch_train_idx = batch_indices[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        batch_train_idx = np.arange(batch_size) + batch_idx * batch_size

        batch_graph_list = list(map(lambda i: graph_list_train[i], batch_train_idx))
        batch_y_list = y_list_train[batch_train_idx].T
        batch_y_list = torch.from_numpy(batch_y_list)
        batch_y_list = batch_y_list.to(model.device)
        # batch_y_list = (batch_y_list - MEAN) / STD # mean = 0, std = 1
        # batch_y_list = (batch_y_list - MIN) / (MAX - MIN) # range [0, 1]
        # batch_y_list = 2 * (batch_y_list - MIN) / (MAX - MIN) - 1 # range [-1, 1]

        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        y_pred, _ = model(batch_graph_list)

        batch_loss_train, batch_loss_sum, batch_loss_mean, _ = loss(y_pred, batch_y_list, loss_fn=F.smooth_l1_loss, mode='train')
        # batch_loss_train, batch_loss_sum, batch_loss_mean, _ = loss(y_pred, batch_y_list, loss_pe, mode='train')

        # batch_loss_train = ul(batch_loss_tasks) + 0.1 * loss_pe

        print(f'Batch #{batch_idx + 1} \t\t Train loss = {batch_loss_sum} \t\t Mean loss = {batch_loss_mean}')

        writer.add_scalar('training_loss', batch_loss_mean, epoch * num_train_batches + batch_idx + 1)

        scaler.scale(batch_loss_train).backward()
        scaler.step(optimizer)
        scaler.update()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch * num_train_batches + batch_idx + 1)
        # lr_scheduler.step()
        # lr_scheduler.step(batch_loss_mean)

        train_loss += batch_loss_mean.item()
    train_loss = train_loss / num_train_batches

    return train_loss

def validate_one_epoch(epoch: int, model, batch_size: int, writer: SummaryWriter, data: tuple[list, np.ndarray]):
    graph_list_val, y_list_val = data

    val_size = len(graph_list_val)

    val_loss = 0.0
    num_val_batches = val_size // batch_size
    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_val_batches), 'Validating'):
            batch_graph_list = []
            batch_y_list = []
            batch_val_idx = np.arange(batch_size) + batch_idx * batch_size

            batch_graph_list = list(map(lambda i: graph_list_val[int(i)], batch_val_idx))
            batch_y_list = y_list_val[batch_val_idx].T
            batch_y_list = torch.from_numpy(batch_y_list)
            batch_y_list = batch_y_list.to(model.device)
            # batch_y_list = (batch_y_list - MEAN) / STD                # mean = 0, std = 1
            # batch_y_list = (batch_y_list - MIN) / (MAX - MIN)         # range [0, 1]
            # batch_y_list = 2 * (batch_y_list - MIN) / (MAX - MIN) - 1   # range [-1, 1]

            y_pred, _ = model(batch_graph_list)

            # y_pred = (y_pred + 1) * (MAX - MIN) / 2                     # scale back
            # batch_y_list = (batch_y_list + 1) * (MAX - MIN) / 2         # scale back

            # batch_loss_mean, _ = loss(y_pred * STD, batch_y_list * STD, loss_fn=F.l1_loss, mode='test')
            # batch_loss_mean, _ = loss(y_pred * (MAX - MIN), batch_y_list * (MAX - MIN), loss_fn=rmse_loss, mode='test')
            batch_loss_mean, _ = loss(y_pred, batch_y_list, loss_fn=F.smooth_l1_loss, mode='test')

            writer.add_scalar('validate_loss', batch_loss_mean, epoch * num_val_batches + batch_idx + 1)

            val_loss += batch_loss_mean.item()
        val_loss = val_loss / num_val_batches

    return val_loss

def train(model, config: dict, writer: SummaryWriter, data: tuple[list, np.ndarray, list, np.ndarray], *args):
    epochs = config['epochs']
    batch_size = config['model']['batch_size']
    embed_dim = config['model']['embed_dim']
    layer_num = config['model']['layer_num']
    readout = config['model']['readout']

    graph_list_train, y_list_train,\
    graph_list_val, y_list_val = data
    
    # y_list_train, y_list_val = y_list_train.T, y_list_val.T
    y_list_train, y_list_val = y_list_train.squeeze(), y_list_val.squeeze()

    scaler, optimizer, lr_scheduler, dataset, tasks = args

    model_path = Path(os.path.join(ROOT, 'model'))

    if len(tasks) == len(TASKS[dataset]):
        model_path = model_path / dataset / 'all_tasks'
    else:
        model_path = model_path / dataset / '_'.join([task for task in tasks])
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    min_valid_loss = np.inf

    # ul = UncertaintyLoss(len(tasks)).to(model.device)

    for epoch in range(epochs):
        print(f'Epoch #{epoch + 1}:')

        print('Training:')
        train_loss = train_one_epoch(epoch, model, batch_size, writer,
                                     (graph_list_train, y_list_train),
                                     scaler, optimizer, lr_scheduler)

        print('Training complete\n')

        print('Validating:')
        val_loss = validate_one_epoch(epoch, model, batch_size, writer,
                                      (graph_list_val, y_list_val))
        # writer.add_scalar('validate_loss', val_loss, epoch)
        lr_scheduler.step(val_loss)
        print('Validating complete\n')

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        print(f'Epoch #{epoch + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}')
        if min_valid_loss > val_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
            min_valid_loss = val_loss
            
            now = datetime.now()
            now = now.strftime('%Y%m%d_%H%M%S')

            # Saving State Dict
            torch.save(model.state_dict(), model_path / f'model_{now}.pth')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    config = load_config(args.config)

    now = datetime.now()
    date_now = now.strftime('%Y%m%d')
    time_now = now.strftime('%H%M%S')
    now = now.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/{}/{}/{}_b{}_lr{}_wd{}'.format(
        args.dataset, date_now, time_now, config['model']['batch_size'], config['optim']['lr'], config['optim']['weight_decay']
    ))

    run = wandb.init(config=config, project=args.dataset, name=f'{now}')

    model = PredModel(config).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optim']['lr'], weight_decay=config['optim']['weight_decay'], amsgrad=True)
    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer,
    #     base_lr=config['optim']['base_lr'],
    #     max_lr=config['optim']['max_lr'],
    #     step_size_up=10,
    #     mode="exp_range",
    #     gamma=0.85,
    #     cycle_momentum=False
    # )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        # factor=0.5,
        patience=1,
        threshold=5e-2,
        min_lr=config['optim']['base_lr'],
        verbose=True
    )

    graph_list_train, y_list_train,\
    graph_list_val, y_list_val = load_dataset(args.data_path, args.raw_data_path, args.dataset, config['tasks'])

    train(model, config, writer,
          (graph_list_train, y_list_train, graph_list_val, y_list_val),
          scaler, optimizer, lr_scheduler, args.dataset, config['tasks'])

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9', 'qm9_norm', 'esol', 'freesolv', 'lipophilicity'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    parser.add_argument('--raw_data_path', type=str, default='raw_data', help='Raw data path')
    parser.add_argument('--data_path', type=str, default='dataset', help='Data path')
    parser.add_argument('--config', type=str, default='qm7_train_config.json', help='Model config')

    args = parser.parse_args()

    # assert args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}'

    main(args)

    #TODO: tune hyperparam