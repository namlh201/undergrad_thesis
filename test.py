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

    return graph_list_test, y_list_test

def test(model, config: dict, data: tuple[list, np.ndarray]):
    # epochs = config.get('epoch', 10)
    batch_size = config['model'].get('batch_size', 32)

    graph_list_test, y_list_test = data
    
    # y_list_test = y_list_test.T
    y_list_test = y_list_test.squeeze()

    print(len(graph_list_test))

    print('Testing:')
    test_loss = 0.0
    num_test_batches = len(graph_list_test) // batch_size
    model.eval()
    for batch_idx in tqdm(range(num_test_batches), 'Testing'):
        batch_graph_list = []
        batch_y_list = []
        batch_test_idx = np.arange(batch_size) + batch_idx

        batch_graph_list = list(map(lambda i: graph_list_test[int(i)], batch_test_idx))
        batch_y_list = y_list_test[batch_test_idx]

        # for i in batch_test_idx:
        #     batch_graph_list.append(graph_list_test[int(i)])
        #     batch_y_list.append(y_list_test[int(i)])

        batch_loss = model(batch_graph_list, batch_y_list)

        test_loss += batch_loss.item()
    test_loss = test_loss / num_test_batches
    print('Testing complete\n')

    print(f'Testing Loss: {test_loss}')

def main(args):
    model_path = Path(os.path.join(ROOT, 'model'))
    model_path = model_path / args.dataset / args.task
    MODEL_NAME = 'model_batch_size_32_epoch_8'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)

    model = PredModel(config).to(device)
    model.load_state_dict(torch.load(model_path / f'{MODEL_NAME}.pth'))

    graph_list_test, y_list_test = load_dataset(args.data_path, args.dataset, config['tasks'])

    test(model, config, (graph_list_test, y_list_test))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    # parser.add_argument('--raw_data_path', type=str, default='raw_data', help='Raw data path')
    parser.add_argument('--data_path', type=str, default='dataset', help='Data path')
    parser.add_argument('--config', type=str, default='qm7_test_config.json', help='Model config')

    args = parser.parse_args()

    # assert args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}'

    main(args)