from model_4ed import GCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


from datamanger import *
from DataSampler import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"使用设备: {device}")

net = GCN(
        edge_dim=8,
        hidden_dim=64,
        output_dim=2,
        num_layer=3,
    ).to(device)

net.load_state_dict(torch.load("Model_4ed_epoch_30.pth"))
net.eval()
print("模型权重加载成功！")

#test_data = create_dataset("val_data.csv")
#torch.save(test_data, "test_data.pt")

test_data = torch.load("test_data.pt")
test_data2 = torch.load("sampled_data/train_sampled_part29.pt")
labels = [pyg.y for pyg in test_data2]
print("正样本个数: ")
print(sum(labels))

test_loader = DataLoader(
    test_data2,
    batch_size=32,
    collate_fn=pyg_collate_fn
)

test_size = len(test_data2)
total_test_loss = 0
correct_test = 0
total_test_samples = 0
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        # 获取标签
        labels = data.y.long().squeeze()

        # 前向传播
        logits = net(data.x, data.edge_index, data.edge_attr, data.global_features, data.batch)


        # 统计
        predictions = torch.argmax(logits, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test_samples += data.num_graphs

        # 收集预测结果用于后续分析
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 计算概率（用于ROC曲线）
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities.extend(probabilities.cpu().numpy())


# 计算测试集指标
avg_test_loss = total_test_loss / test_size
test_accuracy = correct_test / total_test_samples

# 转换为numpy数组
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

# 计算分类指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='binary')
recall = recall_score(all_labels, all_predictions, average='binary')
f1 = f1_score(all_labels, all_predictions, average='binary')

# 计算AUC（对于二分类）
positive_probs = all_probabilities[:, 1]
auc_score = roc_auc_score(all_labels, positive_probs)
cm = confusion_matrix(all_labels, all_predictions)

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")
print(f"\n混淆矩阵:")
print(cm)
# 绘制热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1'],
            yticklabels=['0', '1'])
plt.xlabel('predict')
plt.ylabel('true')
plt.title('confusion_matrix')
plt.show()
# 分类报告
print("\n分类报告:")
print(classification_report(all_labels, all_predictions, target_names=['类别0', '类别1']))

