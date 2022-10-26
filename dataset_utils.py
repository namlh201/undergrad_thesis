from __future__ import annotations

import os
from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
from pprint import pprint

import traceback
import sys

from numpy.testing import assert_almost_equal
from pandas import concat, read_csv, DataFrame
from tqdm import tqdm
from rdkit.Chem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles, SDMolSupplier

from CustomGeoGNN.classes import AtomBondGraphFromMol, BondAngleGraphFromMol
from CustomGeoGNN.utils.split import ScaffoldSplit

ROOT = os.getcwd()

def load_tasks(tasks_path: str) -> dict:
    tasks_path = os.path.join('configs', tasks_path)

    with open(os.path.join(ROOT, tasks_path), 'r') as f:
        tasks: dict = json.load(f)
    
    return tasks

TASKS = load_tasks('tasks.json')

def split_dataset(smiles_list: list, graph_pair_list: list, y_list: DataFrame,
                  train_size=0.8, val_size=0.1, test_size=0.1) -> tuple[list, DataFrame, list, DataFrame, list, DataFrame]:
    splitter = ScaffoldSplit(train_size, val_size, test_size)
    train_indices, val_indices, test_indices = splitter.split(smiles_list)

    graph_list_train = list(map(lambda i: graph_pair_list[i], train_indices))
    y_list_train = list(map(lambda i: y_list.iloc[[i]], train_indices))
    y_list_train = concat(y_list_train, ignore_index=True)

    graph_list_val = list(map(lambda i: graph_pair_list[i], val_indices))
    y_list_val = list(map(lambda i: y_list.iloc[[i]], val_indices))
    y_list_val = concat(y_list_val, ignore_index=True)

    graph_list_test = list(map(lambda i: graph_pair_list[i], test_indices))
    y_list_test = list(map(lambda i: y_list.iloc[[i]], test_indices))
    y_list_test = concat(y_list_test, ignore_index=True)

    # length = len(graph_pair_list)

    # graph_list_train, y_list_train = [], []
    # graph_list_val, y_list_val = [], []
    # graph_list_test, y_list_test = [], []

    # for i in range(length):
    #     if i in train_indices:
    #         graph_list_train.append(graph_pair_list[i])
    #         y_list_train.append(y_list.iloc[[i]])
    #     elif i in val_indices:
    #         graph_list_val.append(graph_pair_list[i])
    #         y_list_val.append(y_list.iloc[[i]])
    #     elif i in test_indices:
    #         graph_list_test.append(graph_pair_list[i])
    #         y_list_test.append(y_list.iloc[[i]])

    # y_list_train = concat(y_list_train, ignore_index=True)
    # y_list_val = concat(y_list_val, ignore_index=True)
    # y_list_test = concat(y_list_test, ignore_index=True)

    return graph_list_train, y_list_train,\
           graph_list_val, y_list_val,\
           graph_list_test, y_list_test

def clean_dataset(mol_list, y_list: DataFrame) -> tuple[list, list, DataFrame]:
    size = min(len(mol_list), len(y_list))

    cleaned_smiles_list = [] # for scaffold split
    cleaned_graph_pair_list = []
    cleaned_y_list = []

    for i in tqdm(range(size), desc='Cleaning Dataset'):
        if mol_list[i]:
            try:
                graph_pair = molecule_to_graph(mol_list[i])
                cleaned_graph_pair_list.append(graph_pair)

                cleaned_smiles_list.append(MolToSmiles(mol_list[i]))
                cleaned_y_list.append(y_list.iloc[[i]])
            except Exception as e:
                # continue
                traceback.print_exception(*sys.exc_info())
                print(i, mol_list[i].GetProp('_Name'))
    
    cleaned_y_list = concat(cleaned_y_list, ignore_index=True)

    return cleaned_smiles_list, cleaned_graph_pair_list, cleaned_y_list

def molecule_to_graph(mol: Mol) -> tuple[AtomBondGraphFromMol, BondAngleGraphFromMol]:
    atom_bond_graph = AtomBondGraphFromMol(mol)
    bond_angle_graph = BondAngleGraphFromMol(mol)
    return atom_bond_graph, bond_angle_graph

def read_dataset(data_path: str, dataset: str, normalize: bool) -> tuple[list, DataFrame]:
    mol_path = os.path.join(ROOT, data_path, dataset, f'{dataset}.sdf')
    y_path = os.path.join(ROOT, data_path, dataset, f'{dataset}.sdf.csv')

    mol_list = SDMolSupplier(mol_path, removeHs=False)
    y_list = read_csv(y_path)
    # Get the values only
    y_list = y_list[TASKS[dataset]]

    if normalize:
        y_list = (y_list - y_list.mean()) / y_list.std()

    return mol_list, y_list

def save_splitted_dataset(dataset: str, normalize: bool, *args):
    graph_list_train, y_list_train,\
    graph_list_val, y_list_val,\
    graph_list_test, y_list_test = args

    dataset_path = 'dataset'
    if normalize:
        dataset = dataset + '_norm'

    train_path = Path(os.path.join(ROOT, dataset_path, dataset, 'train'))
    val_path = Path(os.path.join(ROOT, dataset_path, dataset, 'val'))
    test_path = Path(os.path.join(ROOT, dataset_path, dataset, 'test'))

    if not train_path.exists():
        train_path.mkdir(parents=True, exist_ok=True)
    if not val_path.exists():
        val_path.mkdir(parents=True, exist_ok=True)
    if not test_path.exists():
        test_path.mkdir(parents=True, exist_ok=True)

    with open(train_path / 'train_data.pkl', 'wb') as f:
        pickle.dump(graph_list_train, f)
    # for task in TASKS[dataset]:
    #     with open(train_path / f'train_values_{task}.pkl', 'wb') as f:
    #         # try:
    #         pickle.dump(y_list_train[task].to_list(), f)
            # except Exception as e:
            #     print(y_list_train[task])
    y_list_train.to_csv(train_path / 'train_values.csv')

    with open(val_path / 'val_data.pkl', 'wb') as f:
        pickle.dump(graph_list_val, f)
    # for task in TASKS[dataset]:
    #     with open(val_path / f'val_values_{task}.pkl', 'wb') as f:
    #         pickle.dump(y_list_val[task].to_list(), f)
    y_list_val.to_csv(val_path / 'val_values.csv')

    with open(test_path / 'test_data.pkl', 'wb') as f:
        pickle.dump(graph_list_test, f)
    # for task in TASKS[dataset]:
    #     with open(test_path / f'test_values_{task}.pkl', 'wb') as f:
    #         pickle.dump(y_list_test[task].to_list(), f)
    y_list_test.to_csv(test_path / 'test_values.csv')

def main(args):
    print(f'Reading dataset {args.dataset} at {args.data_path}', end=' ')
    mol_list, y_list = read_dataset(args.data_path, args.dataset, args.normalize)
    print('[Done]')

    print(f'Cleaning dataset {args.dataset}', end=' ')
    smiles_list, graph_pair_list, y_list = clean_dataset(mol_list, y_list)
    print('[Done]')

    if args.job == 'split':
        print(f'Spliiting dataset {args.dataset}', end=' ')
        graph_list_train, y_list_train,\
        graph_list_val, y_list_val,\
        graph_list_test, y_list_test = split_dataset(smiles_list, graph_pair_list, y_list,
                                                     args.train_size, args.val_size, args.test_size)
        print('[Done]')

    if args.save:
        print(f'Saving splitted dataset {args.dataset} to dataset', end=' ')
        save_splitted_dataset(args.dataset, args.normalize,
                              graph_list_train, y_list_train,
                              graph_list_val, y_list_val,
                              graph_list_test, y_list_test)
        print('[Done]')

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--job', type=str, default='split', help='Job')
    parser.add_argument('--dataset', type=str, choices=['qm7', 'qm8', 'qm9'], help='Name of Dataset')
    # parser.add_argument('--task', type=str, help='Name of Task')
    parser.add_argument('--data_path', type=str, default='raw_data', help='Data path')
    parser.add_argument('--train_size', type=float, default=0.8, help='Size of training set')
    parser.add_argument('--val_size', type=float, default=0.1, help='Size of validating set')
    parser.add_argument('--test_size', type=float, default=0.1, help='Size of testing set')
    parser.add_argument('--normalize', type=bool, default=False, help='Whether to normalize targets')
    parser.add_argument('--save', type=bool, default=False, help='Whether to save dataset in pickle format after splitting')

    args = parser.parse_args()

    pprint(args)

    # assert(args.task in TASKS[args.dataset], f'Task must be one of {TASKS[args.dataset]}')

    assert_almost_equal(actual=args.train_size + args.val_size + args.test_size,
                        desired=1.,
                        err_msg='Sum of train_size, val_size and test_size must equals to 1',
                        verbose=True)

    main(args)
