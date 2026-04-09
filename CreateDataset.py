import pandas as pd
import numpy as np



def random_sample(file_path, sample_ratio, max_sampled=100000, chunksize=10000, random_state=42):

    np.random.seed(random_state)

    sampled_data = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # 从当前chunk中随机抽取样本
        sampled_chunk = chunk.sample(n=int(sample_ratio*chunksize), random_state=random_state)

        sampled_data.append(sampled_chunk)

        # 如果已经收集了足够的样本，提前停止
        total_sampled = sum(len(df) for df in sampled_data)
        if total_sampled >= max_sampled:
            break

    # 合并所有抽样结果
    result = pd.concat(sampled_data, ignore_index=True)

    # 如果样本数超过需求，再次抽样
    if len(result) > max_sampled:
        result = result.sample(n=max_sampled, random_state=random_state)

    return result


def sample_active(file_path, sample_ratio, max_sampled=50000, chunksize=10000, random_state=42):
    np.random.seed(random_state)

    sampled_data = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):

        active_chunk = chunk[chunk["binds"] != 0.]
        sampled_chunk = chunk[chunk["binds"] == 0.].sample(n=int(sample_ratio * chunksize), random_state=random_state)

        sampled_data.append(sampled_chunk)
        sampled_data.append(active_chunk)

        # 如果已经收集了足够的样本，提前停止
        total_sampled = sum(len(df) for df in sampled_data)
        if total_sampled >= max_sampled:
            break

    # 合并所有抽样结果
    result = pd.concat(sampled_data, ignore_index=True)

    return result

def random_sample_from(file_path, sample_ratio, max_sampled=50000, sample_from=2000000, chunksize=10000, random_state=42):

    np.random.seed(random_state)

    sampled_data = []
    total_skip = int(sample_from // chunksize)
    skip = 0

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if skip <= total_skip:
            skip += 1
            continue
        # 从当前chunk中随机抽取样本
        sampled_chunk = chunk.sample(n=int(sample_ratio*chunksize), random_state=random_state)

        sampled_data.append(sampled_chunk)

        # 如果已经收集了足够的样本，提前停止
        total_sampled = sum(len(df) for df in sampled_data)
        if total_sampled >= max_sampled:
            break

    # 合并所有抽样结果
    result = pd.concat(sampled_data, ignore_index=True)

    # 如果样本数超过需求，再次抽样
    if len(result) > max_sampled:
        result = result.sample(n=max_sampled, random_state=random_state)

    return result

if __name__ == "__main__":
    # 使用示例
    data_path = r"data/train.csv"
    sample_df = random_sample_from(data_path, 0.05)
    sample_df.to_csv("test_sampled.csv", index=False)
    print(f"抽样结果形状: {sample_df.shape}")
    print(f"总数: {len(sample_df)}")
    print(f"无效: {sum(i == 0. for i in sample_df["binds"])}")

