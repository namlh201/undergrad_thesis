from __future__ import annotations

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
from tqdm import tqdm

from CustomGeoGNN.model import PredModel
from CustomGeoGNN.module.loss_fn import rmse_loss

MEAN = None
STD = None

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
        config = json.load(f)

    print_config(config)
    
    return config

def print_config(config: dict):
    for key in config:
        print(f'[GeoGNNModel] {key}: {config[key]}')

def load_dataset(data_path: str, raw_data_path: str, dataset: str, tasks: list) -> tuple[list, np.ndarray]:
    test_path = Path(os.path.join(ROOT, data_path, dataset, 'test'))

    global MEAN
    global STD
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

    if dataset == 'qm9':
        qm9_conversion = torch.tensor(list(map(qm9_conversion.get, tasks))).view(-1, 1).cuda()
        MEAN = MEAN * qm9_conversion
        STD = STD * qm9_conversion

    # print(MEAN)
    # print(STD)

    with open(test_path / 'test_data.pkl', 'rb') as f:
        graph_list_test = pickle.load(f)

    y_list_test = pd.read_csv(test_path / 'test_values.csv')
    y_list_test = y_list_test[tasks]
    y_list_test = y_list_test.to_numpy()

    return graph_list_test, y_list_test

def get_loss_fn(dataset: str) -> Callable:
    if dataset in ['qm7', 'qm8', 'qm9', 'lipophilicity']:
        return F.smooth_l1_loss
    elif dataset in ['esol']:
        return rmse_loss

def loss(y_pred: torch.Tensor, y: torch.Tensor, loss_fn: Callable, mode: str='test'):
    # GAP = abs(HOMO - LUMO)
    y_pred[2] = torch.abs(y_pred[0] - y_pred[1])

    if mode == 'train':
        # mean of (num_nodes, num_tasks) -> 1
        # loss_batch_mean = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='mean')
        loss_batch_mean = loss_fn(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        # loss_tasks = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='none')
        loss_tasks = loss_fn(y_pred.float(), y.float(), reduction='none')

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
        
        # sum of (num_tasks, 1) -> 1
        loss_batch_sum = torch.sum(loss_tasks)

        # print(loss_tasks, loss_batch_sum, loss_batch_mean, loss_pe)

        # loss_train = loss_batch_sum + 0.7 * loss_pe
        loss_train = loss_batch_sum

        return loss_train, loss_batch_sum, loss_batch_mean
    elif mode == 'test':
        # mean of (num_nodes, num_tasks) -> 1
        # loss_batch_mean = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='mean')
        # loss_batch_mean = F.l1_loss(y_pred.float(), y.float(), reduction='mean')
        loss_batch_mean = loss_fn(y_pred.float(), y.float(), reduction='mean')

        # (num_nodes, num_tasks)
        # loss_tasks = F.smooth_l1_loss(y_pred.float(), y.float(), reduction='none')
        # loss_tasks = F.l1_loss(y_pred.float(), y.float(), reduction='none')
        loss_tasks = loss_fn(y_pred.float(), y.float(), reduction='none')

        # (num_tasks, 1)
        if len(loss_tasks.shape) > 1:
            loss_tasks = torch.mean(loss_tasks, dim=-1)
        else:
            loss_tasks = torch.mean(loss_tasks)
            # loss_batch_mean = torch.min(loss_batch_mean, loss_tasks)

        # print(loss_tasks, loss_batch_mean)

        return loss_batch_mean, loss_tasks

def test(model, dataset: str, loss_fn: Callable, config: dict, data: tuple[list, np.ndarray]):
    # epochs = config.get('epoch', 10)
    batch_size = config['model']['batch_size']

    graph_list_test, y_list_test = data
    
    # y_list_test = y_list_test.T
    y_list_test = y_list_test.squeeze()

    print('Testing:')
    test_loss = 0.0
    test_loss_sum = 0.0
    if len(y_list_test.shape) > 1:
        test_loss_tasks = torch.zeros(y_list_test.shape[-1], dtype=torch.float).to(model.device)
    else:
        test_loss_tasks = torch.zeros(1, dtype=torch.float).to(model.device)
    num_test_batches = len(graph_list_test) // batch_size
    model.eval()
    for batch_idx in tqdm(range(num_test_batches), 'Testing'):
        batch_graph_list = []
        batch_y_list = []
        if dataset in ['qm7', 'esol']:
            batch_test_idx = np.arange(batch_size) + batch_idx
        else:
            batch_test_idx = np.arange(batch_size) + batch_idx * batch_size
        # batch_test_idx = np.arange(batch_size, batch_size * 2)# + batch_idx

        batch_graph_list = list(map(lambda i: graph_list_test[int(i)], batch_test_idx))
        batch_y_list = y_list_test[batch_test_idx].T
        batch_y_list = torch.from_numpy(batch_y_list)
        batch_y_list = batch_y_list.to(model.device)

        y_pred, _ = model(batch_graph_list)
        batch_loss_mean, batch_loss_tasks = loss(y_pred, batch_y_list, loss_fn=loss_fn, mode='test')
        # print(batch_loss_tasks.sum())

        test_loss += batch_loss_mean.item()
        test_loss_sum += batch_loss_tasks.sum().item()
        test_loss_tasks = test_loss_tasks + batch_loss_tasks.clone().detach()
    test_loss = test_loss / num_test_batches
    test_loss_sum = test_loss_sum / num_test_batches
    test_loss_tasks = test_loss_tasks / num_test_batches
    print('Testing complete\n')

    if dataset == 'qm8':
        print(f'Testing Loss: {test_loss_sum}')
    else:
        print(f'Testing Loss: {test_loss}')

    # return test_loss_tasks

def main(args):
    model_path = Path(os.path.join(ROOT, 'model'))
    model_path = model_path / args.dataset / args.model_path
    # MODEL_NAME = 'model_20221109_133153'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)

    model = PredModel(config, args.dataset).to(device)
    # model.load_state_dict(torch.load(model_path / f'{args.model}.pth'))
    model.load_state_dict(torch.load(model_path))

    graph_list_test, y_list_test = load_dataset(args.data_path, args.raw_data_path, args.dataset, config['tasks'])

    loss_fn = get_loss_fn(args.dataset)

    # test_loss_tasks = test(model, loss_fn, config, (graph_list_test, y_list_test))
    test(model, args.dataset, loss_fn, config, (graph_list_test, y_list_test))

    # for task, loss in zip(config['tasks'], test_loss_tasks):
    #     print(f'Loss {task} = {loss}')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9', 'qm9_norm', 'esol', 'freesolv', 'lipophilicity'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    parser.add_argument('--raw_data_path', type=str, default='raw_data', help='Raw data path')
    parser.add_argument('--data_path', type=str, default='dataset', help='Data path')
    parser.add_argument('--config', type=str, default='qm7_test_config.json', help='Model config')
    parser.add_argument('--model_path', type=str, help='Model name')

    args = parser.parse_args()

    # assert args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}'

    main(args)