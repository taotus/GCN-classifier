from torch_geometric.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import hashlib
import os
from typing import Dict
import json
import datetime

from MoleculeGraph import smiles_to_graph, graph_to_PyGData

class MolecularDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super().__init__()
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
    def save(self, path: str):
        save_data = {
            'size': len(self.data_list),
            'data_list': self.data_list,
        }
        torch.save(save_data, path)
        print(f"数据集已保存到: {path}, 包含 {len(self.data_list)} 个样本")

    @classmethod
    def load(cls, file_path: str, transform=None):
        """从文件加载数据集"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        loaded_data = torch.load(file_path)
        dataset = cls(loaded_data['data_list'], transform)
        print(f"从 {file_path} 加载了 {len(dataset)} 个样本")
        return dataset

class ProteinHashManager:
    """管理蛋白质哈希和序列的映射"""

    def __init__(self, file_path: str = "protein_hash_mapping.json"):
        self.file_path = file_path
        self.hash_to_sequence = self._load_mapping()


    def _load_mapping(self) -> Dict[str, str]:
        """从文件加载哈希映射"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载蛋白质哈希映射失败: {e}")
        return {}

    def save_mapping(self):
        """保存哈希映射到文件"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.hash_to_sequence, f, indent=2, ensure_ascii=False)
            print(f"蛋白质哈希映射已保存到: {self.file_path}, 包含 {len(self.hash_to_sequence)} 个蛋白质")
        except Exception as e:
            print(f"保存蛋白质哈希映射失败: {e}")

    def add_sequence(self, sequence: str) -> str:
        """添加蛋白质序列并返回哈希"""
        import hashlib
        sequence_hash = hashlib.md5(sequence.encode()).hexdigest()

        if sequence_hash not in self.hash_to_sequence:
            self.hash_to_sequence[sequence_hash] = sequence

        return sequence_hash

    def get_sequence(self, sequence_hash: str) -> str:
        """根据哈希获取蛋白质序列"""
        return self.hash_to_sequence.get(sequence_hash, "")

    def get_all_sequences(self) -> Dict[str, str]:
        """获取所有序列映射"""
        return self.hash_to_sequence.copy()



def save_complete_dataset(dataset: MolecularDataset,
                          hash_manager: ProteinHashManager,
                          base_path: str = "data"):
    """保存完整的数据集（分子图 + 蛋白质映射）"""
    os.makedirs(base_path, exist_ok=True)

    # 保存分子图数据集
    dataset_path = os.path.join(base_path, "molecular_dataset.pt")
    dataset.save(dataset_path)

    # 保存蛋白质哈希映射
    hash_manager.save_mapping()

    # 保存数据集信息
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info = {
        "dataset_size": len(dataset),
        "protein_count": len(hash_manager.hash_to_sequence),
        "save_time": current_time,
        "dataset_file": "molecular_dataset.pt",
        "protein_mapping_file": "protein_hash_mapping.json"
    }

    info_path = os.path.join(base_path, "dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"完整数据集已保存到: {base_path}")
    return base_path


def load_complete_dataset(base_path: str = "data"):
    """加载完整的数据集"""
    # 加载数据集信息
    info_path = os.path.join(base_path, "dataset_info.json")
    with open(info_path, 'r') as f:
        info = json.load(f)

    # 加载分子图数据集
    dataset_path = os.path.join(base_path, info["dataset_file"])
    dataset = MolecularDataset.load(dataset_path)

    # 加载蛋白质哈希管理器
    hash_manager = ProteinHashManager(os.path.join(base_path, "protein_hash_mapping.json"))

    print(f"从 {base_path} 加载了完整数据集")
    print(f"数据集大小: {len(dataset)}, 蛋白质数量: {len(hash_manager.hash_to_sequence)}")

    return dataset, hash_manager


def get_sequence_hash(sequence: str) -> str:
    """为蛋白质序列生成唯一的哈希值"""
    return hashlib.md5(sequence.encode()).hexdigest()

def load_dataset_first(data_path: str, save_dir: str = "data"):
    """加载原始数据并保存为处理后的数据集"""
    hash_manager = ProteinHashManager()
    pyg_list = []

    data_df = pd.read_csv(data_path)
    smiles_list = data_df["compound_iso_smiles"].tolist()
    target_list = data_df["target_sequence"].tolist()
    affinity_list = data_df["affinity"].tolist()

    for idx, smiles in enumerate(smiles_list):
        graph = smiles_to_graph(smiles)
        sequence = target_list[idx]
        sequence_hash = hash_manager.add_sequence(sequence)

        pyg = graph_to_PyGData(graph, label=affinity_list[idx], target=sequence_hash)
        pyg_list.append(pyg)

    dataset = MolecularDataset(pyg_list)

    # 保存处理后的数据集
    save_complete_dataset(dataset, hash_manager, save_dir)

    return dataset, hash_manager

if __name__ == "__main__":
    #data_path = "data/davis_test.csv"
    mol_dataset, hash_manger = load_complete_dataset()
    print(mol_dataset[0])
    hash_map = hash_manger.get_all_sequences()
    #print(hash_map)





