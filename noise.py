import pandas as pd
import numpy as np

print("正在从上帝视角数据中提取高精度观测数据...")

# 1. 读取你算出来的完美数据集
try:
    df_truth = pd.read_csv("ground_truth_pde.csv")
except FileNotFoundError:
    print("错误：找不到 ground_truth_pde.csv！请确保它在同级目录下。")
    exit()

# 2. 模拟现实中的稀疏采样 (30个点)
unique_times = df_truth['time'].unique()
sparse_indices = np.linspace(0, len(unique_times) - 1, 30, dtype=int)
sampled_times = unique_times[sparse_indices]

# 过滤数据：只保留采样点，暴露外部培养基浓度 Ge
df_sparse = df_truth[df_truth['r'] == 0.01].copy()
df_sparse = df_sparse[df_sparse['time'].isin(sampled_times)]
df_observed = df_sparse[['time', 'Ge']].copy()

# 3. 【核心修改】将噪声从 5% (0.05) 爆降到 0.1% (0.001)
np.random.seed(42) # 固定随机种子
noise_level = 0.001 

# 生成噪声
noise_array = np.random.normal(0, noise_level, len(df_observed))
df_observed['Ge_noisy'] = df_observed['Ge'] * (1 + noise_array)

# 算一算我们给 PINN 减轻了多少负担
mean_noise_abs = np.mean(np.abs(df_observed['Ge'] * noise_array))
print(f"当前添加的平均绝对噪声仅为: {mean_noise_abs:.5f} g/L")
print("（这已经小于内部物理扩散造成的 0.009 g/L 浓度差了！PINN 终于能看清真相了！）")

# 4. 阅后即焚：只留加噪后的数据
df_observed = df_observed[['time', 'Ge_noisy']]
df_observed.to_csv("observed_noisy_pde.csv", index=False)

print(f"搞定！0.1% 极低噪声训练集已保存 (数据规模: {df_observed.shape})")
print("现在的你可以带着 5.0 的瞎猜初始值，回去重新跑一次 pinn.py 了！")