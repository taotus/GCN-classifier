from torch_geometric.nn import GATConv, global_mean_pool
from torch import nn
import torch


class ProteinGNN(nn.Module):
    """蛋白质图神经网络"""

    def __init__(self, node_dim, edge_dim=1, hidden_dim=256, num_layers=3):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
            )

        self.pool = global_mean_pool

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.node_encoder(x)

        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)
            x = torch.relu(x)

        if batch is not None:
            x = self.pool(x, batch)

        return x