import os
import re
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc


from model_4ed import GCN
from FocalLoss import FocalLoss, TverskyLoss
from DataSampler import *

Model_Name = "Model_20260312_Tversky"
folder_path = "sampled_data"
train_number = 5
test_number = 2

# 设置随机种子以保证可重复性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"使用设备: {device}")

def merge_train_data(data_path_list: list[str], type: str="train", batch_size:int=64, positive_frac: float=0.2):

    dataset = []
    for path in data_path_list:
        data = torch.load(path)
        dataset.extend(data)
        print(f"load {path} successfully")

    train_labels = [pyg.y for pyg in dataset]
    # 使用示例
    batch_sampler = BalancedBatchSampler(
        labels=np.array(train_labels),
        batch_size=batch_size,
        pos_frac=positive_frac  # 每个batch中正样本的比例
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=pyg_collate_fn,
    )
    positive = sum(pyg.y == 1 for pyg in dataset)
    print(f"{type}: positive_rate: {positive/len(dataset)}")
    return dataloader, len(dataset)

def load_data():
    file_path = "sampled_data/train_sampled_part"
    train_data_paths = [
        f"{file_path}1.pt",
        f"{file_path}2.pt",
        f"{file_path}3.pt",
        f"{file_path}4.pt",
        f"{file_path}5.pt",
        f"{file_path}6.pt",
        f"{file_path}7.pt",
        f"{file_path}8.pt",
    ]

    train_dataloader, train_size = merge_train_data(
        train_data_paths,
        type="train",
        batch_size=128,
        positive_frac=0.1
    )

    testset1 = torch.load("sampled_data/train_sampled_part9.pt")
    testset2 = torch.load("sampled_data/train_sampled_part10.pt")
    testset = testset1 + testset2
    test_dataloader = DataLoader(
        testset,
        batch_size=128,
        collate_fn=pyg_collate_fn,  # 添加这一行
        shuffle=False,
        pin_memory=True
    )
    test_size = len(testset)
    print(f"训练集:{train_size} 测试集:{test_size}")

    return train_dataloader, train_size, test_dataloader, test_size

def main():

    train_dataloader, train_size, test_dataloader, test_size = load_data()

    # 二分类
    num_classes = 2
    net = GCN(
        edge_dim=8,
        hidden_dim=64,
        output_dim=num_classes,
        num_layer=3,
        dropout=0.3
    ).to(device)

    #分类损失函数
    tversky_loss = TverskyLoss(
        alpha=0.2,
        beta=0.8,
    )

    focal_loss = FocalLoss(
        alpha=1,
        gamma=3,
        reduction="mean"
    )


    learning_rate = 0.001
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    # 添加学习率调度器
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练次数
    total_train_step = 0
    # 测试次数
    total_test_step = 0
    # 训练轮数
    total_epoch = 30

    # 记录训练历史
    train_loss_history = []
    test_loss_history = []
    train_acc_history = []
    test_acc_history = []

    print("开始训练...")

    for epoch in range(total_epoch):
        print(f"\n-----------第{epoch + 1}轮训练-----------")


        # 训练阶段
        net.train()
        total_train_loss = 0
        correct_train = 0
        total_train_samples = 0

        for data in train_dataloader:
            data = data.to(device)

            # 获取标签（分类任务，标签应该是整数索引）
            labels = data.y.long().squeeze()  # 确保标签是整数类型
            # 前向传播
            logits = net(data.x, data.edge_index, data.edge_attr, data.global_features, data.batch)
            # 计算损失
            loss = focal_loss(logits, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_train_loss += loss.item() * data.num_graphs  # 乘以实际batch大小
            predictions = torch.argmax(logits, dim=1)
            correct_train += (predictions == labels).sum().item()
            total_train_samples += data.num_graphs

            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f"train Loss: {loss}")

        # 计算训练集指标
        avg_train_loss = total_train_loss / train_size
        train_accuracy = correct_train / total_train_samples
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        print(f"训练集平均Loss: {avg_train_loss:.4f}")
        print(f"训练集准确率: {train_accuracy:.4f}")
        # 测试阶段
        net.eval()
        total_test_loss = 0
        correct_test = 0
        total_test_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for data in test_dataloader:
                data = data.to(device)

                # 获取标签
                labels = data.y.long().squeeze()

                # 前向传播
                logits = net(data.x, data.edge_index, data.edge_attr, data.global_features, data.batch)

                # 计算损失
                loss = focal_loss(logits, labels)

                # 统计
                total_test_loss += loss.item() * data.num_graphs
                predictions = torch.argmax(logits, dim=1)
                correct_test += (predictions == labels).sum().item()
                total_test_samples += data.num_graphs

                # 收集预测结果用于后续分析
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 计算概率（用于ROC曲线）
                probabilities = torch.softmax(logits, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                total_test_step += 1

        # 计算测试集指标
        avg_test_loss = total_test_loss / test_size
        test_accuracy = correct_test / total_test_samples
        test_loss_history.append(avg_test_loss)
        test_acc_history.append(test_accuracy)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')

        print(f"测试集平均Loss: {avg_test_loss:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")

        # 更新学习率
        #scheduler.step()

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f"{Model_Name}_epoch_{epoch + 1}.pth")
            print(f"模型已保存: {Model_Name}_epoch_{epoch + 1}.pth")

    print("训练完成！")

if __name__ == "__main__":
    main()


