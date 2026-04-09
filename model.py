import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch_geometric import seed_everything

def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)  # PyG的特殊设置


class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.update_w = nn.Linear(hidden_dim, hidden_dim)

    def message(self, h_u, h_v, E=None):
        return h_u
    def aggregate(self, H, v, messages, norm_factor):
        norm_factor = norm_factor.reshape([-1, 1])
        weight_messages = norm_factor * messages

        aggregated = torch.zeros_like(H)
        aggregated = aggregated.index_add(0, v, weight_messages)

        return aggregated

    def update(self, H, aggregated_messages):
        output = self.update_w(aggregated_messages + H)
        return output

    def forward(self, H, edge_index, norm_factor=None, E=None):

        H = self.node_encoder(H)

        u, v = edge_index[0], edge_index[1]

        h_u = H[u]
        h_v = H[v]

        messages = self.message(h_u, h_v, E)

        aggregated = self.aggregate(H, v, messages, norm_factor)

        H_updated = self.update(H, aggregated)

        return F.relu(H_updated)

class GCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim=1, num_layer=3, dropout=0.3):
        super(GCN, self).__init__()

        self.num_layer = num_layer

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(MPNN(node_dim, edge_dim, hidden_dim))

        for _ in range(num_layer - 1):
            self.gcn_layers.append(MPNN(hidden_dim, edge_dim, hidden_dim))

        self.dropout_layer = nn.Dropout(dropout)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def calc_norm_factor(self, edge_index, num_nodes, self_loop=True):
        device = edge_index.device
        """
        num, _ = adj_matrix.shape
        if self_loop:
            adj_matrix = adj_matrix + torch.eye(num, device=device)
        """

        row, col = edge_index[0], edge_index[1]

        deg = torch.zeros(num_nodes, dtype=torch.float, device=device)

        deg = deg.scatter_add(0, row, torch.ones(row.size(), dtype=torch.float,
                                                 device=device))

        if self_loop:
            deg = deg + 1.0

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_factors = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm_factors

    def readout_function(self, H):
        return torch.mean(H, dim=0, keepdim=True)

    def forward(self, H, edge_index, E, batch=None):
        num_nodes = H.shape[0]
        norm_factor = self.calc_norm_factor(edge_index, num_nodes=num_nodes)
        for i, mpnn in enumerate(self.gcn_layers):
            H = mpnn(H, edge_index, norm_factor, E)

            if i < self.num_layer -1:
                H = self.dropout_layer(H)

        if batch is not None:
            batch_size = batch.max().item() + 1
            graph_embeddings = []

            for i in range(batch_size):
                mask = (batch == i)
                graph_output = self.readout_function(H[mask])
                graph_embeddings.append(graph_output)

            graph_embedding = torch.cat(graph_embeddings, dim=0)

        else:
            graph_embedding = self.readout_function(H)

        output = self.output_layer(graph_embedding)

        return output

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"不可训练参数数量: {total_params - trainable_params:,}")

    return total_params, trainable_params

if __name__ == "__main__":
    device = 'cuda:0'
    set_seed()
    from MoleculeGraph import smiles_to_graph, graph_to_PyGData
    from torch_geometric.loader import DataLoader
    smiles = ["C", "CC"]
    dataset = []
    for smile in smiles:
        graph = smiles_to_graph(smile)
        data = graph_to_PyGData(graph, target='a', label=1)
        dataset.append(data)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        pin_memory=True
    )
    net = GCN(
            node_dim=12,
            edge_dim=8,
            hidden_dim=128,
            num_layer=3,
            dropout=0.5
        ).to(device)
    total_params, trainable_params = count_parameters(net)
    for data in dataloader:
        data = data.to(device)
        batch = data.batch
        node_dim = data.x.shape[1]
        E = data.edge_attr
        edge_dim = E.shape[1]

        print(net)

        print(f"\n前向传播 (节点分类):")
        node_outputs = net(data.x, data.edge_index, E, batch)
        print(f"节点输出形状: {node_outputs.shape}")

        break



