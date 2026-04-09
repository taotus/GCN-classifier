import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR, LambdaLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from MoleculeGraph import smiles_to_graph, graph_to_PyGData
from model_3ed import GCN


def plot_predictions(y_true, y_pred, title_suffix: str = "") -> None:
    """绘制预测值与实际值的散点图"""
    y_true = np.array(y_true).flatten()  # 确保是一维数组
    y_pred = np.array(y_pred).flatten()  # 确保是一维数组

    plt.figure(figsize=(8, 6))

    # 计算误差
    errors = y_pred - y_true

    # 创建散点图
    plt.scatter(y_true, y_pred, alpha=0.6, c=np.abs(errors), cmap='viridis')
    plt.colorbar(label='Absolute Error')

    # 添加参考线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')


    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title(f'prediction {title_suffix}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def data_normalization(dataset):
    train_labels = np.array([data.y.item() for data in train_data])
    train_mean = train_labels.mean()
    train_std = train_labels.std()
    for data in dataset:
        data.y = (data.y - train_mean) / train_std

    return train_mean, train_std

def visualize_label_distribution(y_train: np.ndarray, y_test: np.ndarray,
                                 dataset_name: str = "Dataset",
                                 figsize: tuple = (15, 12)):
    """
    可视化训练集和测试集的标签分布（适用于回归任务）

    参数:
        y_train: 训练集标签（连续值）
        y_test: 测试集标签（连续值）
        dataset_name: 数据集名称
        figsize: 图表大小
    """
    # 创建DataFrame用于可视化
    train_df = pd.DataFrame({
        'Set': ['Train'] * len(y_train),
        'Value': y_train
    })

    test_df = pd.DataFrame({
        'Set': ['Test'] * len(y_test),
        'Value': y_test
    })

    df = pd.concat([train_df, test_df])

    # 计算统计指标
    train_stats = {
        'mean': np.mean(y_train),
        'median': np.median(y_train),
        'std': np.std(y_train),
        'min': np.min(y_train),
        'max': np.max(y_train),
        'range': np.ptp(y_train)
    }

    test_stats = {
        'mean': np.mean(y_test),
        'median': np.median(y_test),
        'std': np.std(y_test),
        'min': np.min(y_test),
        'max': np.max(y_test),
        'range': np.ptp(y_test)
    }

    # 创建图表
    plt.figure(figsize=figsize)
    plt.suptitle(f'Label Distribution for {dataset_name} (Regression)', fontsize=16)

    # 1. 分布直方图与KDE
    plt.subplot(2, 2, 1)
    sns.histplot(y_train, color='blue', label='Train', kde=True, alpha=0.5, bins=30)
    sns.histplot(y_test, color='red', label='Test', kde=True, alpha=0.5, bins=30)
    plt.title('Value Distribution Comparison')
    plt.xlabel('Label Value')
    plt.ylabel('Density')
    plt.legend()

    # 2. 箱线图比较
    plt.subplot(2, 2, 2)
    sns.boxplot(x='Set', y='Value', data=df, palette={'Train': 'blue', 'Test': 'red'})
    plt.title('Boxplot Comparison')
    plt.xlabel('Dataset Split')
    plt.ylabel('Label Value')

    # 3. 累积分布函数(CDF)
    plt.subplot(2, 2, 3)

    # 创建共同的x轴范围
    min_val = min(np.min(y_train), np.min(y_test))
    max_val = max(np.max(y_train), np.max(y_test))
    x_range = np.linspace(min_val, max_val, 500)  # 使用500个点

    # 计算训练集在共同点上的CDF
    sorted_train = np.sort(y_train)
    cdf_train = np.searchsorted(sorted_train, x_range, side='right') / len(y_train)
    plt.plot(x_range, cdf_train, label='Train', color='blue')

    # 计算测试集在共同点上的CDF
    sorted_test = np.sort(y_test)
    cdf_test = np.searchsorted(sorted_test, x_range, side='right') / len(y_test)
    plt.plot(x_range, cdf_test, label='Test', color='red')

    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Label Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)

    # 4. QQ图 (分位数-分位数图)
    plt.subplot(2, 2, 4)
    # 计算分位数
    quantiles = np.linspace(0, 1, 100)
    train_quantiles = np.quantile(y_train, quantiles)
    test_quantiles = np.quantile(y_test, quantiles)

    plt.scatter(train_quantiles, test_quantiles, alpha=0.7)

    # 添加参考线 (y=x)
    min_val = min(np.min(y_train), np.min(y_test))
    max_val = max(np.max(y_train), np.max(y_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

    plt.title('Quantile-Quantile Plot')
    plt.xlabel('Train Quantiles')
    plt.ylabel('Test Quantiles')
    plt.legend()
    plt.grid(True)

    # 调整子图间距和位置
    plt.subplots_adjust(
        top=0.9,  # 增加顶部空间
        bottom=0.07,  # 增加底部空间
        left=0.08,  # 增加左侧空间
        right=0.95,  # 增加右侧空间
        hspace=0.3,  # 增加行间距
        wspace=0.25  # 增加列间距
    )
    plt.savefig(f'{dataset_name}_regression_distribution.png', dpi=300, bbox_inches='tight')

    # 打印统计信息
    print("\n===== 分布统计 =====")
    print(f"训练集样本数: {len(y_train)}")
    print(f"测试集样本数: {len(y_test)}")

    print("\n训练集统计:")
    print(f"  均值: {train_stats['mean']:.4f}")
    print(f"  中位数: {train_stats['median']:.4f}")
    print(f"  标准差: {train_stats['std']:.4f}")
    print(f"  范围: [{train_stats['min']:.4f}, {train_stats['max']:.4f}]")

    print("\n测试集统计:")
    print(f"  均值: {test_stats['mean']:.4f}")
    print(f"  中位数: {test_stats['median']:.4f}")
    print(f"  标准差: {test_stats['std']:.4f}")
    print(f"  范围: [{test_stats['min']:.4f}, {test_stats['max']:.4f}]")

    # 计算分布相似性指标
    # KS检验（近似）
    ks_statistic = np.max(np.abs(cdf_train - cdf_test))
    print(f"\n分布相似性:")
    print(f"  KS统计量: {ks_statistic:.4f}")
    print("  KS值越小表示分布越相似 (理想值接近0)")

    # Wasserstein距离（推土机距离）
    wasserstein = np.mean(np.abs(train_quantiles - test_quantiles))
    print(f"  Wasserstein距离: {wasserstein:.4f}")
    print("  距离越小表示分布越相似")

    plt.show()

    stats = {
        'train_stats': train_stats,
        'test_stats': test_stats,
        'ks_statistic': ks_statistic,
        'wasserstein_distance': wasserstein
    }

    return stats

dataset = []
df = pd.read_csv("train_sampled.csv")
smiles_list  = df["molecule_smiles"].to_list()
labels = df["binds"].to_list()
targets = df["protein_name"].to_list()
for idx, smiles in enumerate(smiles_list):
    graph = smiles_to_graph(smiles, target=targets[idx])
    pygdata = graph_to_PyGData(
        graph,
        target=targets[idx],
        label=labels[idx]
    )
    dataset.append(pygdata)

train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 对数据进行标准化，deepseek说这个很重要
train_mean, train_std = data_normalization(train_data)
for data in test_data:
    data.y = (data.y - train_mean) / train_std

y_train = [data.y.item() for data in train_data]
y_test = [data.y.item() for data in test_data]
y_train = np.array(y_train)
y_test = np.array(y_test)

#visualize_label_distribution(y_train, y_test)

# 设置随机种子以保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

train_dataloader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    pin_memory=True
)
train_size = len(train_data)
test_size = len(test_data)

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

net = GCN(
    edge_dim=8,
    hidden_dim=128,
    output_dim=1,
    num_layer=3,
    dropout=0.3
).to(device)
loss_function = nn.MSELoss(reduction='mean')

learning_rate = 0.001
def lr_lamdba(epoch):
    if epoch < 10:
        return epoch / 10
    else:
        return 0.95 ** (epoch - 10)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#scheduler = LambdaLR(optimizer, lr_lambda=lr_lamdba)

#训练次数
total_train_step = 0
#测试次数
total_test_step = 0
#训练轮数
total_epoch = 300



for epoch in range(total_epoch):
    print("-----------第{}轮训练---------".format(epoch + 1))

    net.train()
    total_train_loss = 0
    for data in train_dataloader:
        data = data.to(device)

        H = data.x
        edge_index = data.edge_index
        E = data.edge_attr
        batch = data.batch

        label = data.y
        if label.dim() == 1:
            label = label.view(-1, 1)

        output = net(H, edge_index, E, batch)

        loss = loss_function(output, label)
        total_train_loss += loss.item() * 64

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
    print(f"训练集平均Loss: {total_train_loss/train_size:.4f}")

    net.eval()
    with torch.no_grad():
        total_test_loss = 0
        all_preds = []
        all_reals = []

        for data in test_dataloader:
            data = data.to(device)
            H = data.x
            edge_index = data.edge_index
            E = data.edge_attr
            label = data.y
            batch = data.batch
            if label.dim() == 1:
                label = label.view(-1, 1)
            y_pred = net(H, edge_index, E, batch)
            loss = loss_function(y_pred, label)
            total_test_loss += loss.item()

            # 收集预测值和真实值（注意去批处理维度）
            all_preds.append(y_pred.cpu().numpy())
            all_reals.append(label.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_reals = np.concatenate(all_reals, axis=0)


        def calculate_r2(y_pred, y_true):
            # 计算总平方和
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            # 计算残差平方和
            ss_res = np.sum((y_true - y_pred) ** 2)
            # 计算R²
            r2 = 1 - (ss_res / (ss_tot + 1e-8))  # 加小量防止除零
            return r2


        r2_score = calculate_r2(all_preds, all_reals)

        print(f"测试集平均Loss: {total_test_loss/test_size:.4f}")
        print(f"决定系数R²: {r2_score:.4f}")
        total_test_step += 1

        #if (epoch+1) % 10 == 0:
            #plot_predictions(all_reals, all_preds, "GCN")

    if (epoch + 1) % 10 == 0:
        torch.save(net, "NET_{}".format(epoch + 1))
        print("第{}轮训练的模型已保存".format(epoch + 1))




