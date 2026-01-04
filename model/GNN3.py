import torch
import torch.nn.functional as F 
from torch.nn import Linear
from torch_geometric.nn import GATConv, TopKPooling, TransformerConv, GINConv
from torch_geometric.nn import global_mean_pool as gap

class GNN3(torch.nn.Module):
    def __init__(self, feature_size, edge_feature_size, embedding_size=1024):
        super(GNN3, self).__init__()
        num_classes = 2
        
        # Isomorphism layer (GINConv)
        nn1 = torch.nn.Sequential(Linear(feature_size, embedding_size), torch.nn.ReLU(), Linear(embedding_size, embedding_size))
        self.isomorphism = GINConv(nn1)
        
        # Transformer layer
        self.transformer = TransformerConv(embedding_size, embedding_size, heads=4)
        self.transformer_reduce = Linear(embedding_size * 4, embedding_size)
        
        # GNN3 layers
        self.conv1 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3, edge_dim=edge_feature_size)
        self.head_transform1 = Linear(embedding_size * 3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3, edge_dim=edge_feature_size)
        self.head_transform2 = Linear(embedding_size * 3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3, edge_dim=edge_feature_size)
        self.head_transform3 = Linear(embedding_size * 3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.3)
        
        # Linear layers
        self.linear1 = Linear(embedding_size, 1024)
        self.linear2 = Linear(1024, 1)
        
    def forward(self, x, edge_attr, edge_index, batch_index):
        # Isomorphism layer
        x = self.isomorphism(x, edge_index)
        
        # Transformer layer
        x = self.transformer(x, edge_index)
        x = self.transformer_reduce(x)
        
        # First block
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.head_transform1(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x, edge_index, edge_attr, batch_index)
        x1 = gap(x, batch_index)
        
        # Second block
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x, edge_index, edge_attr, batch_index)
        x2 = gap(x, batch_index)

        # Third block
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, edge_index, edge_attr, batch_index)
        x3 = gap(x, batch_index)
        
        # Sum pooled vectors
        x = x1 + x2 + x3
        
        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        
        return x
