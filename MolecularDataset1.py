import os
import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
import pandas as pd
from pathlib import Path


from MoleculeGraph import smiles_to_graph, graph_to_PyGData

class MolecularDataset(Dataset):
    """支持分块保存和加载的分子数据集类"""

    def __init__(self,
                 data_dir: str,
                 chunk_size: int = 1000,
                 mode: str = 'read',  # 'read' or 'write'
                 transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.transform = transform
        self.mode = mode

        if mode == 'write':
            self.current_chunk = []
            self.current_chunk_idx = 0
            self.total_samples = 0
            self.metadata = {
                'chunk_size': chunk_size,
                'num_chunks': 0,
                'total_samples': 0,
                'chunk_sample_counts': []
            }
        else:
            # 加载模式
            self.metadata = torch.load(self.data_dir / 'metadata.pt')
            self.total_samples = self.metadata['total_samples']

    def add_sample(self, pygdata):
        """添加单个样本"""
        if self.mode != 'write':
            raise ValueError("数据集处于只读模式")

        self.current_chunk.append(pygdata)
        self.total_samples += 1

        if len(self.current_chunk) >= self.chunk_size:
            self._save_current_chunk()

    def _save_current_chunk(self):
        """保存当前块"""
        if not self.current_chunk:
            return

        # 使用PyTorch保存块数据
        chunk_file = self.data_dir / f'chunk_{self.current_chunk_idx:04d}.pt'
        torch.save({
            'data': self.current_chunk,
            'chunk_idx': self.current_chunk_idx,
            'num_samples': len(self.current_chunk)
        }, chunk_file)

        # 更新元数据
        self.metadata['num_chunks'] += 1
        self.metadata['chunk_sample_counts'].append(len(self.current_chunk))
        self.metadata['total_samples'] = self.total_samples

        # 保存元数据
        torch.save(self.metadata, self.data_dir / 'metadata.pt')

        self.current_chunk_idx += 1
        self.current_chunk = []
        print(f"已保存块 {self.current_chunk_idx - 1}，包含 {self.metadata['chunk_sample_counts'][-1]} 个样本")

    def finalize(self):
        """完成写入，保存最后一个块"""
        if self.mode == 'write' and self.current_chunk:
            self._save_current_chunk()
        print(f"数据集保存完成，共 {self.total_samples} 个样本")

    def _load_chunk(self, chunk_idx: int):
        """加载指定块"""
        chunk_file = self.data_dir / f'chunk_{chunk_idx:04d}.pt'
        if not chunk_file.exists():
            raise FileNotFoundError(f"块文件不存在: {chunk_file}")

        chunk_data = torch.load(chunk_file, map_location='cpu')
        return chunk_data['data']

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围")

        # 找到对应的块
        cumulative_samples = 0
        for chunk_idx, count in enumerate(self.metadata['chunk_sample_counts']):
            if idx < cumulative_samples + count:
                # 加载对应的块
                chunk_data = self._load_chunk(chunk_idx)
                # 获取块内的数据
                sample = chunk_data[idx - cumulative_samples]
                if self.transform:
                    sample = self.transform(sample)
                return sample
            cumulative_samples += count

        raise IndexError(f"索引 {idx} 计算错误")

    @classmethod
    def load(cls, data_dir: str, transform=None):
        """加载数据集"""
        return cls(data_dir, mode='read', transform=transform)





def process_large_dataset_chunked(input_csv: str,
                                  output_dir: str,
                                  chunk_size: int = 1000,
                                  batch_size: int = 100):
    """
    处理大型数据集，分块保存

    Args:
        input_csv: 输入CSV文件路径
        output_dir: 输出目录
        chunk_size: 每个块包含的样本数
        batch_size: 每批处理的样本数（用于进度显示）
    """

    # 创建数据集实例（写入模式）
    dataset = MolecularDataset(
        data_dir=output_dir,
        chunk_size=chunk_size,
        mode='write'
    )

    # 使用pandas的chunksize分批读取CSV
    print(f"开始读取数据: {input_csv}")

    # 方式1：使用chunksize分批读取（适合非常大的文件）
    chunk_iter = pd.read_csv(input_csv, chunksize=batch_size)

    total_processed = 0
    for chunk_df in chunk_iter:
        smiles_list = chunk_df["molecule_smiles"].to_list()
        labels = chunk_df["binds"].to_list()
        targets = chunk_df["protein_name"].to_list()

        for idx, smiles in enumerate(smiles_list):
            try:
                # 这里假设smiles_to_graph和graph_to_PyGData函数已定义
                graph = smiles_to_graph(smiles, target=targets[idx])
                pygdata = graph_to_PyGData(
                    graph,
                    target=targets[idx],
                    label=labels[idx]
                )
                dataset.add_sample(pygdata)
                total_processed += 1

            except Exception as e:
                print(f"处理第 {total_processed + idx} 个分子时出错: {e}")
                continue

        print(f"已处理 {total_processed} 个分子")

    # 保存最后一个块
    dataset.finalize()

    # 验证数据集
    print(f"\n数据集统计:")
    print(f"总样本数: {dataset.total_samples}")
    print(f"块数量: {dataset.metadata['num_chunks']}")
    print(f"数据已保存到: {output_dir}")

    return dataset


# 使用示例
if __name__ == "__main__":
    # 处理数据
    dataset_train = process_large_dataset_chunked(
        input_csv="train_data.csv",
        output_dir="./train_dataset",
        chunk_size=2000,  # 每个块2000个样本
        batch_size=500  # 每次读取500行
    )

    dataset_validate = process_large_dataset_chunked(
        input_csv="val_data.csv",
        output_dir="./val_dataset",
        chunk_size=2000,  # 每个块2000个样本
        batch_size=500  # 每次读取500行
    )

    # 加载数据（示例）
    loaded_dataset = MolecularDataset.load(
        data_dir="./train_dataset"
    )

    # 创建DataLoader（支持分块加载）
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        loaded_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # 使用多进程加载
        pin_memory=True
    )

    print(f"数据加载器创建完成，共有 {len(loaded_dataset)} 个样本")