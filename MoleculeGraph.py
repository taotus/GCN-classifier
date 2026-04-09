from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Tuple
import torch
from torch_geometric.data import Data


class FEATURE:
    BOND_IDX = "bond_index"
    BOND_ATTR = "bond_attr"
    ATOM = "atom_features"
    GLOBAL = "global_features"

def get_atom_features(atom: AllChem.Atom) -> int:
    features = atom.GetAtomicNum(),
    return features

def get_bond_features(bond: AllChem.Bond) -> Tuple[list, list]:
    bond_index = [
        bond.GetBeginAtomIdx(),
        bond.GetEndAtomIdx()
    ]
    bond_attr = [
        bond.GetBondTypeAsDouble(),
        1 if bond.GetIsAromatic() else 0,
        1 if bond.IsInRing() else 0,
        1 if bond.IsInRingSize(size=3) else 0,
        1 if bond.IsInRingSize(size=4) else 0,
        1 if bond.IsInRingSize(size=5) else 0,
        1 if bond.IsInRingSize(size=6) else 0,
        1 if bond.IsInRingSize(size=7) else 0,
    ]
    return bond_index, bond_attr

def get_global_features(target: str) -> int:
    if target.lower() == "brd4":
        features = 1
    elif target.lower() == "hsa":
        features = 2
    elif target.lower() == "seh":
        features = 3
    else:
        features = 0
    return features

def smiles_to_graph(smiles: str, target: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    #FIXME: 此处应该添加极性氢
    mol = AllChem.AddHs(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)
        atom_features.append(features)

    bond_index = []
    bond_attr = []
    for bond in mol.GetBonds():
        index, attr = get_bond_features(bond)
        bond_index.append(index)
        bond_attr.append(attr)

    global_features = get_global_features(target)

    graph = {
        FEATURE.ATOM: atom_features,
        FEATURE.BOND_IDX: bond_index,
        FEATURE.BOND_ATTR: bond_attr,
        FEATURE.GLOBAL: global_features
    }
    return graph

def graph_to_PyGData(graph: dict, label: float, target: str) -> Data:
    node_features = torch.tensor(graph[FEATURE.ATOM], dtype=torch.long)
    edge_attr = torch.tensor(graph[FEATURE.BOND_ATTR], dtype=torch.float)
    global_features = torch.tensor(graph[FEATURE.GLOBAL], dtype=torch.long)
    edge_index = torch.tensor(graph[FEATURE.BOND_IDX], dtype=torch.long).t().contiguous()

    num_nodes = len(graph[FEATURE.ATOM])
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    for bond in graph[FEATURE.BOND_IDX]:
        i, j = bond
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    batch = torch.zeros(num_nodes, dtype=torch.long)

    data = Data(
        x=node_features,
        edge_attr=edge_attr,
        edge_index=edge_index,
        global_features=global_features,
        batch=batch,
        target=target,
        y=torch.tensor([label], dtype=torch.float)
    )
    return data

if __name__ == "__main__":
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # 咖啡因
    graph = smiles_to_graph(smiles, target="seh")

    print("全局特征:", graph[FEATURE.GLOBAL])
    print("边索引示例:", graph[FEATURE.BOND_IDX])  # 只显示前5个
    print("边特征示例:", graph[FEATURE.BOND_ATTR][:5])  # 只显示前5个
    print("原子特征示例:", graph[FEATURE.ATOM][:5])  # 只显示前5个

    pygdata = graph_to_PyGData(graph, target="a", label=1)
    print("PyG数据对象:")
    print("节点特征形状:", pygdata.x.shape)
    print(pygdata.x.shape[1])
    print("边索引形状:", pygdata.edge_index.shape)
    print("边特征形状:", pygdata.edge_attr.shape)
    print("全局特征形状:", pygdata.global_features.shape)
    print("批处理: ", pygdata.batch)
    print(pygdata.y)
    print("="*50)

    print(pygdata.edge_index)
    idx = pygdata.edge_index
    row, col = idx[0], idx[1]
    print(row)
    print(col)



    """
    smiles_list = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        'CCO',  # 乙醇
        'c1ccccc1',  # 苯
        'C1CCCCC1',  # 环己烷
        'CC(=O)O',  # 乙酸
        'CNC=O'  # 甲酰胺
    ]

    for i, smiles in enumerate(smiles_list):
        file_name = f"opt_mol_{i}.xyz"
        mol = generate_3d_structure(smiles, optimize=True)
        opt_xyz, energy, state, atoms = rdkit_to_xtb_direct(mol, 0)
        print(opt_xyz)
        print(energy)
        conv_info = check_convergence(atoms)
        print(conv_info)
        with open(file_name, 'w') as f:
            f.write(opt_xyz)
        print("="*50)
    """










