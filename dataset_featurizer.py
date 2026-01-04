import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm
import numpy as np
from rdkit import Chem

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return [self.filename]

    @property
    def processed_file_names(self):
        data_path = os.path.join(self.raw_dir, self.filename)
        if not os.path.exists(data_path):
            data_path = os.path.join(self.root, self.filename)
            
        self.data = pd.read_csv(data_path).reset_index()

        if self.test:
            return [f'data_test_{self.filename}_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{self.filename}_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        data_path = os.path.join(self.raw_dir, self.filename)
        if not os.path.exists(data_path):
            data_path = os.path.join(self.root, self.filename)
            
        self.data = pd.read_csv(data_path)
        for index, mol_row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Processing molecules"):
            mol_obj = Chem.MolFromSmiles(mol_row["smiles"])
            if mol_obj is None:
                continue
                
            node_feats = self._get_node_features(mol_obj)
            edge_feats = self._get_edge_features(mol_obj)
            edge_index = self._get_adjacency_info(mol_obj)
            label = self._get_labels(mol_row["HIV_active"])

            data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol_row["smiles"]
                        ) 
            
            prefix = 'data_test' if self.test else 'data'
            torch.save(data, os.path.join(self.processed_dir, f'{prefix}_{self.filename}_{index}.pt'))

    def _get_node_features(self, mol):
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.IsInRing()),
                int(atom.GetChiralTag())
            ]
            all_node_feats.append(node_feats)
        return torch.tensor(np.array(all_node_feats), dtype=torch.float)

    def _get_edge_features(self, mol):
        all_edge_feats = []
        for bond in mol.GetBonds():
            edge_feats = [
                bond.GetBondTypeAsDouble(),
                int(bond.IsInRing())
            ]
            all_edge_feats += [edge_feats, edge_feats]
        return torch.tensor(np.array(all_edge_feats), dtype=torch.float)

    def _get_adjacency_info(self, mol):
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        return torch.tensor([label], dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        prefix = 'data_test' if self.test else 'data'
        data = torch.load(os.path.join(self.processed_dir, f'{prefix}_{self.filename}_{idx}.pt'), weights_only=False)
        return data