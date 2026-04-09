import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train_sampled_4ed.csv")
print(f"抽样结果形状: {df.shape}")
print(f"总数: {len(df)}")
print(f"无效: {sum(i == 0. for i in df["binds"])}")

train_data, test_data = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True,
)

print(f"训练集总数: {len(train_data)}")
print(f"无效: {sum(i == 0. for i in train_data["binds"])}")

print(f"测试集总数: {len(test_data)}")
print(f"无效: {sum(i == 0. for i in test_data["binds"])}")

train_data.to_csv("train_data.csv")
test_data.to_csv("val_data.csv")







