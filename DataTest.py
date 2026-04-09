from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


from datamanger import *
from DataSampler import *


import pandas as pd
import torch

from model_4ed import GCN
from MoleculeGraph import smiles_to_graph, graph_to_PyGData

def load_test_data():
    dataset = []
    df = pd.read_csv("data/test.csv")

    print("读取数据完成")
    print(len(df))
    smiles_list = df["molecule_smiles"].to_list()
    print(len(smiles_list))
    targets = df["protein_name"].to_list()

    for idx, smiles in enumerate(smiles_list):
        graph = smiles_to_graph(smiles, target=targets[idx])
        pygdata = graph_to_PyGData(
            graph,
            target=targets[idx],
            label=0
        )
        if (idx + 1) % 100 == 0:
            print(f"处理完成第 {idx + 1} 个分子图")
        dataset.append(pygdata)
    torch.save(dataset, "data/test.pt")
    print(f"保存至 data/test.pt")

def load_model(device):

    net = GCN(
        edge_dim=8,
        hidden_dim=64,
        output_dim=2,
        num_layer=3,
        dropout=0.3
    ).to(device)

    net.load_state_dict(torch.load("Model_4ed_epoch_30.pth"))
    net.eval()
    print("模型权重加载成功！")
    return net


def test_model(test_data, device, net):
    test_loader = DataLoader(
        test_data,
        batch_size=128,
        collate_fn=pyg_collate_fn
    )
    y_pred = []
    with torch.no_grad():
        idx = 0
        for data in test_loader:
            data = data.to(device)
            idx += 1
            # 前向传播
            logits = net(data.x, data.edge_index, data.edge_attr, data.global_features, data.batch)
            # 统计
            predictions = torch.softmax(logits, dim=1)
            y_pred.append(predictions[:, 1])
            if idx % 100 == 0:
                print(f"处理完成第 {idx * 100} batch")
    y_pred = torch.cat(y_pred, dim=0)
    return y_pred


if __name__ == "__main__":
    #load_test_data()
    # 目前最佳模型：Model_4ed_epodh_30.pth
    """
    net = GCN(
        edge_dim=8,
        hidden_dim=64,
        output_dim=2,
        num_layer=3,
        dropout=0.3
    ).to(device)
    """
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"使用设备: {device}")

    net = load_model(device)
    test_data = torch.load("data/test.pt")
    print(f"训练数据加载成果: {len(test_data)}")
    y_pred = test_model(test_data, device, net).cpu().numpy()
    torch.save(y_pred, "test_pred.pt")
    """
    y_pred = torch.load("test_pred.pt")
    print(len(y_pred))
    test_df = pd.read_csv("data/test.csv")

    id_df = test_df["id"]
    binds_df = y_pred

    test_data = pd.DataFrame({
        "id": id_df,
        "binds": binds_df
    })

    print(test_data.head())
    test_data.to_csv("submission.csv", index=False)








