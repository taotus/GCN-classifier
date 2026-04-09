import numpy as np
import torch
import os
import re

from torch_geometric.data import Batch as PyGBatch
from torch.utils.data import DataLoader, Sampler


class BalancedBatchSampler(Sampler):
    """确保每个batch都有正负样本"""

    def __init__(self, labels, batch_size, pos_frac=0.1):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.pos_frac = pos_frac

        # 分离正负样本索引
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        # 计算每个batch中正样本数量
        self.pos_per_batch = int(batch_size * pos_frac)
        self.neg_per_batch = batch_size - self.pos_per_batch

        # 确保有足够的正样本进行采样
        assert len(self.pos_indices) >= self.pos_per_batch, "正样本不足！"

    def __iter__(self):
        n_batches = len(self) // self.batch_size

        for _ in range(n_batches):
            # 从正样本中随机采样
            pos_batch = np.random.choice(
                self.pos_indices,
                self.pos_per_batch,
                replace=True  # 允许重复采样
            )

            # 从负样本中随机采样
            neg_batch = np.random.choice(
                self.neg_indices,
                self.neg_per_batch,
                replace=False
            )

            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return len(self.labels)

def pyg_collate_fn(batch):
    """将 PyG Data 对象列表转换为批处理"""
    return PyGBatch.from_data_list(batch)


if __name__ == "__main__":

    pattern = re.compile(r'train_sampled_part(\d+)\.pt$')
    folder_path = "sampled_data"
    train_files = []
    # 遍历文件夹中的所有条目
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            # 提取数字部分（字符串形式）
            number_str = match.group(1)
            # 转换为整数以便排序
            number = int(number_str)
            full_path = os.path.join(folder_path, filename)
            train_files.append(full_path)

    data_path_list = train_files[:11]

    dataset = []
    for path in data_path_list:
        data = torch.load(path)
        dataset.extend(data)
        print(f"load {path} successfully")

    train_labels = [pyg.y for pyg in dataset]
    # 使用示例
    batch_sampler = BalancedBatchSampler(
        labels=np.array(train_labels),
        batch_size=32,
        pos_frac=0.3  # 每个batch中30%是正样本
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=pyg_collate_fn
    )
    for data in train_loader:
        print(data.y)
        break


