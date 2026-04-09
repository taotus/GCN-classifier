import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F

from sqlalchemy.testing.provision import update_db_opts
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric import seed_everything


class MolecularGCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, output_dim=1, num_layers=3):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            nn_model = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_model))

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, data: Data):
        x = self.node_encoder(data.node_features)
        x = self.activation(x)
        edge_attr = self.edge_encoder(data.edge_features)

        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
            x = self.activation(x)

        x = global_mean_pool(x, data.batch)

        x = self.fc(x)

        return x