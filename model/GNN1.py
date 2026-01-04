import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap

class GNN1(torch.nn.Module):
    def __init__(self, feature_size, embedding_size=1024):
        super(GNN1, self).__init__()
        num_classes = 2
        
        # GNN1 layers
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size * 3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size * 3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3 = Linear(embedding_size * 3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.3)
        
        # Linear layers
        self.linear1 = Linear(embedding_size, 1024)
        self.linear2 = Linear(1024, 1)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # First block
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x, edge_index, None, batch_index)
        x1 = gap(x, batch_index)
        
        # Second block
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x, edge_index, None, batch_index)
        x2 = gap(x, batch_index)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, edge_index, None, batch_index)
        x3 = gap(x, batch_index)
        
        # Concat pooled vector
        x = x1 + x2 + x3
        
        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        
        return x