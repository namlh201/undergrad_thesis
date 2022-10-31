from __future__ import annotations

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
import torch.nn.functional as F
from tqdm import tqdm

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
        config = json.load(f)

    print_config(config)
    
    return config

def print_config(config: dict):
    for key in config:
        print(f'[GeoGNNModel] {key}: {config[key]}')

def load_dataset(data_path: str, dataset: str, tasks: list) -> tuple[list, np.ndarray]:
    test_path = Path(os.path.join(ROOT, data_path, dataset, 'test'))

    # y_list_test = []
    with open(test_path / 'test_data.pkl', 'rb') as f:
        graph_list_test = pickle.load(f)
    # for task in tasks:
    #     with open(test_path / f'test_values_{task}.pkl', 'rb') as f:
    #         y_list_test.append(pickle.load(f))
    y_list_test = pd.read_csv(test_path / 'test_values.csv')
    y_list_test = y_list_test[tasks]
    y_list_test = y_list_test.to_numpy()

    return graph_list_test, y_list_test

def loss(y_pred: torch.Tensor, y: torch.Tensor, loss_pe: torch.Tensor=None, mode: str='test'):
    if mode == 'train':
        # mean of (num_nodes, num_tasks) -> 1
        loss_batch_mean = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        loss_tasks = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='none')

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
        
        # sum of (num_tasks, 1) -> 1
        loss_batch_sum = torch.sum(loss_tasks)

        print(loss_tasks, loss_batch_sum, loss_batch_mean, loss_pe)

        loss_train = loss_batch_sum + 0.7 * loss_pe

        return loss_train, loss_batch_sum, loss_batch_mean
    elif mode == 'test':
        # mean of (num_nodes, num_tasks) -> 1
        loss_batch_mean = F.l1_loss(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        loss_tasks = F.l1_loss(y_pred.float(), y.float(), reduction='none')

        print(loss_tasks, loss_batch_mean)

        return loss_batch_mean, loss_tasks

def test(model, config: dict, data: tuple[list, np.ndarray]):
    # epochs = config.get('epoch', 10)
    batch_size = config['model']['batch_size']

    graph_list_test, y_list_test = data
    
    # y_list_test = y_list_test.T
    y_list_test = y_list_test.squeeze()

    print('Testing:')
    test_loss = 0.0
    if len(y_list_test.shape) > 1:
        test_loss_tasks = torch.zeros(y_list_test.shape[-1], dtype=torch.float).to(model.device)
    else:
        test_loss_tasks = torch.zeros(1, dtype=torch.float).to(model.device)
    num_test_batches = len(graph_list_test) // batch_size
    model.eval()
    for batch_idx in tqdm(range(num_test_batches), 'Testing'):
        batch_graph_list = []
        batch_y_list = []
        batch_test_idx = np.arange(batch_size) + batch_idx

        batch_graph_list = list(map(lambda i: graph_list_test[int(i)], batch_test_idx))
        batch_y_list = y_list_test[batch_test_idx].T

        # for i in batch_test_idx:
        #     batch_graph_list.append(graph_list_test[int(i)])
        #     batch_y_list.append(y_list_test[int(i)])

        y_pred, _ = model(batch_graph_list)
        batch_loss_mean, batch_loss_tasks = loss(y_pred, batch_y_list, mode='test')

        test_loss += batch_loss_mean.item()
        test_loss_tasks = test_loss_tasks + batch_loss_tasks.clone().detach()
    test_loss = test_loss / num_test_batches
    test_loss_tasks = test_loss_tasks / num_test_batches
    print('Testing complete\n')

    print(f'Testing Loss: {test_loss}')

    return test_loss_tasks

def main(args):
    model_path = Path(os.path.join(ROOT, 'model'))
    model_path = model_path / args.dataset / 'all_tasks'
    MODEL_NAME = 'best_model_20221022_102140_b64_eb32_l8_rmean_e3'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)

    model = PredModel(config).to(device)
    model.load_state_dict(torch.load(model_path / f'{MODEL_NAME}.pth'))

    graph_list_test, y_list_test = load_dataset(args.data_path, args.dataset, config['tasks'])

    test_loss_tasks = test(model, config, (graph_list_test, y_list_test))

    for task, loss in zip(TASKS[args.dataset], test_loss_tasks):
        print(f'Loss {task} = {loss}')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9', 'qm9_norm'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    # parser.add_argument('--raw_data_path', type=str, default='raw_data', help='Raw data path')
    parser.add_argument('--data_path', type=str, default='dataset', help='Data path')
    parser.add_argument('--config', type=str, default='qm7_test_config.json', help='Model config')

    args = parser.parse_args()

    # assert args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}'

    main(args)