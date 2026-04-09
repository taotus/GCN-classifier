import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score


# 1. 数据准备：将分子图转换为PyG数据格式
def smiles_to_pyg_data(smiles_string, label=None):
    """
    将SMILES字符串转换为PyTorch Geometric的Data对象

    参数:
        smiles_string (str): 分子的SMILES表示
        label (int/float): 分子的标签（如活性分类或性质值）

    返回:
        Data: PyG的Data对象
    """
    from torch_geometric.data import Data

    # 从SMILES创建分子对象
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"无效的SMILES字符串: {smiles_string}")

    # 添加氢原子
    mol = Chem.AddHs(mol)

    # 获取原子数量
    num_atoms = mol.GetNumAtoms()

    # 初始化节点特征矩阵
    node_features = []

    # 定义原子特征
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # 原子序数
            atom.GetDegree(),  # 连接的非氢原子数
            atom.GetFormalCharge(),  # 形式电荷
            atom.GetChiralTag(),  # 手性标签
            atom.GetTotalNumHs(),  # 连接的氢原子数
            int(atom.GetHybridization()),  # 杂化方式
            int(atom.GetIsAromatic()),  # 是否为芳香原子
        ]
        node_features.append(features)

    # 转换为张量
    x = torch.tensor(node_features, dtype=torch.float)

    # 创建边列表
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        # 获取键连接的两个原子的索引
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 添加双向边（无向图）
        edge_index.append([i, j])
        edge_index.append([j, i])

        # 键特征
        bond_features = [
            bond.GetBondTypeAsDouble(),  # 键类型
            int(bond.GetIsConjugated()),  # 是否共轭
            int(bond.IsInRing()),  # 是否在环中
        ]

        # 每条边添加两次（双向）
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)

    # 转换为张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 创建PyG Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 添加标签（如果有）
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    # 添加SMILES字符串作为标识
    data.smiles = smiles_string

    return data


# 2. 定义图卷积神经网络模型
class MolecularGCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, output_dim=1, num_layers=3):
        """
        分子图卷积神经网络

        参数:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GCN层数
        """
        super(MolecularGCN, self).__init__()

        # 节点特征编码层
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # 边特征编码层
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # GCN卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # 全局池化后的全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, data):
        # 编码节点特征
        x = self.node_encoder(data.x)
        x = self.activation(x)

        # 编码边特征（可选，这里简单示例）
        # edge_attr = self.edge_encoder(data.edge_attr)

        # 应用图卷积层
        for conv in self.convs:
            x = conv(x, data.edge_index)  # 可以添加edge_weight=edge_attr
            x = self.activation(x)

        # 全局平均池化
        x = global_mean_pool(x, data.batch)  # data.batch用于批处理

        # 全连接层
        x = self.fc(x)

        return x


# 3. 训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# 4. 评估函数
def evaluate_model(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            predictions.extend(out.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    return np.array(predictions), np.array(labels)


# 5. 创建示例数据集（实际应用中应从文件加载）
def create_sample_dataset():
    """创建一个小型示例数据集"""
    # 这里使用一些简单的分子和假设的活性标签
    samples = [
        ("CCO", 1),  # 乙醇 - 假设有活性
        ("C1=CC=CC=C1", 0),  # 苯 - 假设无活性
        ("CC(=O)O", 1),  # 乙酸 - 假设有活性
        ("C1CCCCC1", 0),  # 环己烷 - 假设无活性
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 1),  # 咖啡因 - 假设有活性
        ("C1CC1", 0),  # 环丙烷 - 假设无活性
        ("CCN(CC)CC", 1),  # 三乙胺 - 假设有活性
        ("C1CCOC1", 0),  # 四氢呋喃 - 假设无活性
    ]

    dataset = []
    for smiles, label in samples:
        try:
            data = smiles_to_pyg_data(smiles, label)
            dataset.append(data)
        except:
            print(f"无法处理SMILES: {smiles}")

    return dataset


# 6. 主程序
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建示例数据集
    dataset = create_sample_dataset()
    print(f"数据集大小: {len(dataset)}")

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False
    )

    # 获取特征维度
    sample_data = dataset[0]
    node_dim = sample_data.x.shape[1]
    edge_dim = sample_data.edge_attr.shape[1] if hasattr(sample_data, 'edge_attr') else 0

    # 初始化模型
    model = MolecularGCN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=3
    ).to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()  # 二分类任务

    # 训练模型
    num_epochs = 50
    train_losses = []

    print("开始训练...")
    for epoch in range(num_epochs):
        loss = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')

    # 评估模型
    print("评估模型...")
    train_preds, train_labels = evaluate_model(model, train_loader, device)
    test_preds, test_labels = evaluate_model(model, test_loader, device)

    # 将预测概率转换为二分类结果
    train_preds_binary = (train_preds > 0.5).astype(int)
    test_preds_binary = (test_preds > 0.5).astype(int)

    # 计算准确率
    train_acc = accuracy_score(train_labels, train_preds_binary)
    test_acc = accuracy_score(test_labels, test_preds_binary)

    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'molecular_gcn_model.pth')
    print("模型已保存到 molecular_gcn_model.pth")