import os.path

import pandas as pd
import torch

from MoleculeGraph import smiles_to_graph, graph_to_PyGData


def create_dataset(data_path):
    dataset = []
    df = pd.read_csv(data_path)
    print("读取数据完成")
    smiles_list = df["molecule_smiles"].to_list()
    labels = df["binds"].to_list()
    targets = df["protein_name"].to_list()

    for idx, smiles in enumerate(smiles_list):
        graph = smiles_to_graph(smiles, target=targets[idx])
        pygdata = graph_to_PyGData(
            graph,
            target=targets[idx],
            label=labels[idx]
        )
        if (idx + 1) % 100 == 0:
            print(f"处理完成第 {idx + 1} 个分子图")
        dataset.append(pygdata)
    return dataset

def create_dataset_by_chunk(data_path, save_dict, sample_ratio: float = 0.1, chunk_size: int = 100000, random_state: int = 42):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)

    chunk_id = 0
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunk_id += 1
        sampled_data = []
        # 从当前chunk中随机抽取样本
        size = min(chunk_size, len(chunk))
        sampled_chunk = chunk.sample(n=int(sample_ratio*size), random_state=random_state)
        sampled_data.append(sampled_chunk)
        sampled_df = pd.concat(sampled_data, ignore_index=True)

        smiles_list = sampled_df["molecule_smiles"].to_list()
        labels = sampled_df["binds"].to_list()
        targets = sampled_df["protein_name"].to_list()

        sampled_dataset = []
        for idx, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, target=targets[idx])
            pygdata = graph_to_PyGData(
                graph,
                target=targets[idx],
                label=labels[idx]
            )
            if (idx + 1) % 100 == 0:
                print(f"处理完成第 {idx + 1} 个分子图")
            sampled_dataset.append(pygdata)

        torch.save(sampled_dataset, f"{save_dict}/train_sampled_part{chunk_id}.pt")

if __name__ == "__main__":

    dataset = torch.load("sampled_data/train_sampled_part2.pt")
    y_list = [data.y for data in dataset]
    print(max(y_list))



