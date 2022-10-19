import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from split import ScaffoldSplit

torch.autograd.set_detect_anomaly(True)

from rdkit.Chem.rdmolfiles import SDMolSupplier


# from featurize import AtomFeaturize
from model import PredModel
from utils import clean_dataset

# from values import VALUES

batch_size = 32
epochs = 10
min_valid_loss = np.inf

ROOT = os.getcwd()

with open(os.path.join(ROOT, 'CustomGeoGNN/configs/config.json'), 'r') as f:
    config = json.load(f)

mol_list = SDMolSupplier(os.path.join(ROOT, 'raw_data/qm7/qm7.sdf'), removeHs=False)
y_list = pd.read_csv(os.path.join(ROOT, 'raw_data/qm7/qm7.sdf.csv'))

smiles_list, graph_pair_list, y_list = clean_dataset(mol_list, y_list, task='u0_atom')

# graph_pair_list = molecule_to_graph(mol_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_dataset(smiles_list, graph_pair_list, y_list):
    length = len(graph_pair_list)

    splitter = ScaffoldSplit()
    train_indices, val_indices, test_indices = splitter.split(smiles_list)

    graph_list_train, y_list_train = [], []
    graph_list_val, y_list_val = [], []
    graph_list_test, y_list_test = [], []

    for i in range(length):
        if i in train_indices:
            graph_list_train.append(graph_pair_list[i])
            y_list_train.append(y_list[i])
        elif i in val_indices:
            graph_list_val.append(graph_pair_list[i])
            y_list_val.append(y_list[i])
        elif i in test_indices:
            graph_list_test.append(graph_pair_list[i])
            y_list_test.append(y_list[i])

    return graph_list_train, y_list_train,\
           graph_list_val, y_list_val,\
           graph_list_test, y_list_test

# print('Loading Data')
# data_loader = MoleculeDataLoader(mol_list, y_list, config, batch_size=2, num_workers=1, device=device)

# print('Done Loading Data')

model = PredModel(config).to(device)

# atom_bond_graph = atom_bond_graph.to(device)
# bond_angle_graph = bond_angle_graph.to(device)

# def train(model, graph_list_train, y_list_train):
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4, amsgrad=True)

# data_loader.train()
# train_set, val_set, test_set = data_loader()

graph_list_train, y_list_train,\
graph_list_val, y_list_val,\
graph_list_test, y_list_test = split_dataset(smiles_list, graph_pair_list, y_list)

#TODO: cleanup everything and begin training
# works
# with open('train_data.pkl', 'wb') as f:
#     pickle.dump(graph_list_train, f)
# with open('train_values.pkl', 'wb') as f:
#     pickle.dump(y_list_train, f)
# with open('val_data.pkl', 'wb') as f:
#     pickle.dump(graph_list_val, f)
# with open('val_values.pkl', 'wb') as f:
#     pickle.dump(y_list_train, f)
# with open('test_data.pkl', 'wb') as f:
#     pickle.dump(graph_list_test, f)
# with open('test_values.pkl', 'wb') as f:
#     pickle.dump(y_list_train, f)

for epoch in range(epochs):
    print(f'Epoch #{epoch + 1}:')

    print('Training:')
    train_loss = 0.0
    num_train_batches = len(graph_list_train) // batch_size
    model.train()
    for batch_idx in tqdm(range(num_train_batches), desc='Training'):
        batch_graph_list = []
        batch_y_list = []
        batch_train_idx = torch.randint(len(graph_list_train), (batch_size,))
        for i in batch_train_idx:
            batch_graph_list.append(graph_list_train[int(i)])
            batch_y_list.append(y_list_train[int(i)])

        # for batch in train_set:
        optimizer.zero_grad()
        # with torch.cuda.amp.autocast():
        # loss = model(atom_bond_graph, bond_angle_graph)
        # loss = model(graph_list_train, y_list_train)
        batch_loss = model(batch_graph_list, batch_y_list)
    
        print(f'Batch #{batch_idx + 1} loss = {batch_loss}')

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += batch_loss.item()
    train_loss = train_loss / num_train_batches
    print('Training complete\n')

    print('Validating:')
    val_loss = 0.0
    num_val_batches = len(graph_list_val) // batch_size
    model.val()
    for batch_idx in tqdm(range(num_val_batches), 'Validating'):
        batch_graph_list = []
        batch_y_list = []
        batch_val_idx = torch.randint(len(graph_list_val), (batch_size,))
        for i in batch_val_idx:
            batch_graph_list.append(graph_list_val[int(i)])
            batch_y_list.append(y_list_val[int(i)])

        batch_loss = model(batch_graph_list, batch_y_list)

        val_loss += batch_loss.item()
    val_loss = val_loss / num_val_batches
    print('Validating complete\n')

    print(f'Epoch #{epoch + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss}')
    if min_valid_loss > val_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss:.6f}) \t Saving The Model')
        min_valid_loss = val_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

# test_batch = next(iter(test_set))

# model.eval()
# # loss = model(atom_bond_graph, bond_angle_graph)
# loss = model(graph_list_test, y_list_test)
# # loss = F.smooth_l1_loss(out, torch.Tensor([atom_bond_graph.y]).to(device))
# print(f'Accuracy: {loss}')

# atom_features = AtomFeaturize(gdb7)

# print(atom_features.node_features)