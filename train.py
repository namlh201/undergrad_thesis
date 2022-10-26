from __future__ import annotations

from datetime import datetime

import os
from argparse import ArgumentParser
import json
from pathlib import Path
import pickle

# import traceback
# import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb

from CustomGeoGNN.model import PredModel

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

def load_dataset(data_path: str, dataset: str, tasks: list) -> tuple[list, np.ndarray, list, np.ndarray]:
    train_path = Path(os.path.join(ROOT, data_path, dataset, 'train'))
    val_path = Path(os.path.join(ROOT, data_path, dataset, 'val'))

    # y_list_train = []
    with open(train_path / 'train_data.pkl', 'rb') as f:
        graph_list_train = pickle.load(f)
    # for task in tasks:
    #     with open(train_path / f'train_values_{task}.pkl', 'rb') as f:
    #         y_list_train.append(pickle.load(f))
    y_list_train = pd.read_csv(train_path / 'train_values.csv')
    y_list_train = y_list_train[tasks].to_numpy()
    
    # y_list_val = []
    with open(val_path / 'val_data.pkl', 'rb') as f:
        graph_list_val = pickle.load(f)
    # for task in tasks:
    #     with open(val_path / f'val_values_{task}.pkl', 'rb') as f:
    #         y_list_val.append(pickle.load(f))
    y_list_val = pd.read_csv(val_path / 'val_values.csv')
    y_list_val = y_list_val[tasks].to_numpy()

    return graph_list_train, y_list_train,\
           graph_list_val, y_list_val

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

    if len(tasks) == len(TASKS[dataset[:3]]):
        model_path = model_path / dataset / 'all_tasks'
    else:
        model_path = model_path / dataset / '_'.join([task for task in tasks])
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    min_valid_loss = np.inf

    for epoch in range(epochs):
        print(f'Epoch #{epoch + 1}:')

        print('Training:')
        train_loss = 0.0
        num_train_batches = len(graph_list_train) // batch_size
        model.train()
        for batch_idx in tqdm(range(num_train_batches), desc='Training'):
            batch_graph_list = []
            batch_y_list = []
            batch_train_idx = np.random.choice(len(graph_list_train), size=batch_size, replace=False)

            batch_graph_list = list(map(lambda i: graph_list_train[i], batch_train_idx))
            batch_y_list = y_list_train[batch_train_idx].T

            # for i in batch_train_idx:
            #     batch_graph_list.append(graph_list_train[int(i)])
            #     batch_y_list.append(y_list_train[int(i)])

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            batch_loss_train_with_pe, batch_loss_train, batch_loss_mean, batch_loss_tasks = \
                model(batch_graph_list, batch_y_list, iteration=epoch * num_train_batches + batch_idx + 1)

            print(f'Batch #{batch_idx + 1} \t\t Train loss = {batch_loss_train} \t\t Mean loss = {batch_loss_mean}')

            writer.add_scalar('training_loss', batch_loss_train, epoch * num_train_batches + batch_idx + 1)

            scaler.scale(batch_loss_train_with_pe).backward()
            scaler.step(optimizer)
            scaler.update()
            writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch * num_train_batches + batch_idx + 1)
            lr_scheduler.step()

            train_loss += batch_loss_train_with_pe.item()
        train_loss = train_loss / num_train_batches
        # writer.add_scalar('training_loss', train_loss, epoch)
        print('Training complete\n')

        print('Validating:')
        val_loss = 0.0
        num_val_batches = len(graph_list_val) // batch_size
        model.eval()
        for batch_idx in tqdm(range(num_val_batches), 'Validating'):
            batch_graph_list = []
            batch_y_list = []
            # batch_val_idx = np.random.choice(len(graph_list_val), size=batch_size, replace=False)
            batch_val_idx = np.arange(batch_size) + batch_idx

            batch_graph_list = list(map(lambda i: graph_list_val[int(i)], batch_val_idx))
            batch_y_list = y_list_val[batch_val_idx].T

            # for i in batch_val_idx:
            #     batch_graph_list.append(graph_list_val[int(i)])
            #     batch_y_list.append(y_list_val[int(i)])

            batch_loss_train_with_pe, batch_loss_train, batch_loss_mean, batch_loss_tasks = model(batch_graph_list, batch_y_list)

            writer.add_scalar('validate_loss', batch_loss_mean, epoch * num_val_batches + batch_idx + 1)

            val_loss += batch_loss_mean.item()
        val_loss = val_loss / num_val_batches
        # writer.add_scalar('validate_loss', val_loss, epoch)
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
            torch.save(model.state_dict(), model_path / f'model_{now}_b{batch_size}_eb{embed_dim}_l{layer_num}_r{readout}_e{epoch + 1}.pth')

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
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config['optim']['base_lr'],
        max_lr=config['optim']['max_lr'],
        step_size_up=10,
        mode="exp_range",
        gamma=0.85,
        cycle_momentum=False
    )
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     min_lr=config['optim']['base_lr'],
    #     verbose=True
    # )

    graph_list_train, y_list_train,\
    graph_list_val, y_list_val = load_dataset(args.data_path, args.dataset, config['tasks'])

    train(model, config, writer,
          (graph_list_train, y_list_train, graph_list_val, y_list_val),
          scaler, optimizer, lr_scheduler, args.dataset, config['tasks'])

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9', 'qm9_norm'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    # parser.add_argument('--raw_data_path', type=str, default='raw_data', help='Raw data path')
    parser.add_argument('--data_path', type=str, default='dataset', help='Data path')
    parser.add_argument('--config', type=str, default='qm7_train_config.json', help='Model config')

    args = parser.parse_args()

    # assert args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}'

    main(args)

    #TODO: tune hyperparam