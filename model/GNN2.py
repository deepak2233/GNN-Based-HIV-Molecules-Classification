import os
import rdkit
import torch
import cairosvg
import numpy as np
import torch_geometric
import pandas as pd
from tqdm import tqdm
import deepchem as dc
from PIL import Image
from rdkit import Chem
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import Draw
from rdkit.Chem import Draw
from rdkit.Chem import rdBase
import matplotlib.pyplot as plt
import IPython.display as display
from rdkit.Chem.Draw import IPythonConsole
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
from dataset_featurizer import MoleculeDataset
import torch.nn.functional as F 
import seaborn as sns
from torch.nn import Sequential, Linear, BatchNorm1d, ModuleList, ReLU
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.nn.conv.x_conv import XConv
torch.manual_seed(42)


class GNN2(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN2, self).__init__()
        num_classes = 2
        embedding_size = 1024
        
        # GNN2 layers
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size*3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size*3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3 = Linear(embedding_size*3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.3)
        
        # Transformer layer
        self.transformer = TransformerConv(embedding_size, heads=4, num_layers=3)
        
        # Isomorphism layer
        self.isomorphism = XConv(Linear(feature_size, embedding_size), K=3)
        
        # Linear layers
        self.linear1 = Linear(embedding_size*2, 1024)
        self.linear2 = Linear(1024, num_classes)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # Isomorphism layer
        x = self.isomorphism(x, edge_index, edge_attr)
        
        # Transformer layer
        x = self.transformer(x, edge_index)
        
        # First block
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        
        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x,
                                                                 edge_index,
                                                                 None,
                                                                 batch_index)
        x1 = torch.cat([gap(x, batch_index), gap(x, batch_index)], dim=1)
        
        # Second block
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x,
                                                                 edge_index,
                                                                 None,
                                                                 batch_index)
        
        x2 = torch.cat([gap(x, batch_index), gap(x, batch_index)], dim=1)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x,
                                                                 edge_index,
                                                                 None,
                                                                 batch_index)
        
        x3 = torch.cat([gap(x, batch_index), gap(x, batch_index)], dim=1)
        
        # Concat pooled vector
        x = x1 + x2 + x3
        
        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        
        return x
