import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示问题（同时兼容 Windows 和 Mac 常用中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_heatmap(file_path):
    print(f"正在读取 {file_path} 并绘制时空热力图...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"找不到 {file_path}，请确认你已经运行了最新的 pde.py！")
        return

    # 1. 提取网格边界，用于设定坐标轴范围
    t_min, t_max = df['time'].min(), df['time'].max()
    r_min, r_max = df['r'].min(), df['r'].max()

    # 2. 将长格式数据转换为透视表 (行是 r, 列是 time)
    # 这样矩阵的行索引对应纵坐标(深度)，列索引对应横坐标(时间)
    pivot_df = df.pivot(index='r', columns='time', values='G')
    
    # 3. 开启高分辨率画布
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 4. 使用 imshow 绘制热力图
    # interpolation='bilinear' 让像素网格平滑过渡
    # cmap='magma' 让枯竭区(低浓度)呈现深黑色，高浓度呈现明黄/亮橙色
    # origin='lower' 确保 r=0 (球心) 在图表的正下方
    plt.imshow(pivot_df.values, 
               aspect='auto', 
               cmap='magma', 
               interpolation='bilinear', 
               origin='lower', 
               extent=[t_min, t_max, r_min, r_max])
    
    # 5. 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('葡萄糖浓度 G (g/L)', fontsize=12)
    
    # 6. 设置标题和标签
    plt.title("微载体深层营养浓度时空演变热力图\n(呈现向心枯竭效应)", fontsize=15, pad=15)
    plt.xlabel("培养时间 (h)", fontsize=12)
    plt.ylabel("微载体径向深度 r (cm)", fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_heatmap("ground_truth_pde.csv")