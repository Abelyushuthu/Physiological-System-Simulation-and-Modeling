import matplotlib.pyplot as plt
import numpy as np
import os

# =====================================================================
# 1. 提取的训练日志与成果数据 (Epoch 0 到 4000)
# =====================================================================
epochs = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
data_loss = np.array([9.7772, 0.1963, 0.1739, 0.1055, 0.0799, 0.0720, 0.0645, 0.0554, 0.0460])
phys_loss = np.array([0.9040, 0.0090, 0.0031, 0.0010, 0.0004, 0.0002, 0.0015, 0.0001, 0.0001])
ic_loss = np.array([15.992, 0.0010, 0.0003, 0.0001, 0.00007, 0.00004, 0.00003, 0.00002, 0.00001])
d_eff = np.array([1.99e-5, 1.35e-5, 1.28e-5, 1.24e-5, 1.22e-5, 1.21e-5, 1.18e-5, 1.16e-5, 1.12e-5])

true_deff = 1.00e-5
error_pct = np.abs(d_eff - true_deff) / true_deff * 100

# =====================================================================
# 2. 全局字体与科研级排版设置
# =====================================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['figure.facecolor'] = '#ffffff'

# 通用绘图样式参数
lw = 2.5        # 主线条宽度
ms = 7          # 数据点大小
alpha_g = 0.5   # 网格线透明度

# =====================================================================
# 统一的坐标轴美化函数 (去除上方和右侧边框，保留底部和左侧)
# =====================================================================
def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=11)
    ax.grid(True, linestyle='--', color='#d3d3d3', alpha=alpha_g)

print("正在生成并保存 4 张独立的高清分析图，请稍候...")

# =====================================================================
# 图 A：观测数据拟合残差 (Data Loss)
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=300)
ax1.plot(epochs, data_loss, color='#B22222', marker='o', linewidth=lw, markersize=ms, label='Data Loss (表面浓度残差)')
ax1.set_yscale('log')
ax1.set_xlabel('迭代次数 (Epochs)', fontsize=12)
ax1.set_ylabel('均方误差 (MSE, Log Scale)', fontsize=12)
ax1.set_title('图 A：观测数据拟合残差收敛曲线', fontsize=15, fontweight='bold', pad=12)
ax1.legend(loc='upper right', fontsize=11)
style_axis(ax1)
plt.tight_layout()
plt.savefig("FigA_Data_Loss.png", transparent=False, bbox_inches='tight')
plt.close(fig1)

# =====================================================================
# 图 B：偏微分方程与初值残差 (Physics & IC Loss)
# =====================================================================
fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=300)
ax2.plot(epochs, phys_loss, color='#00509E', marker='s', linewidth=lw, markersize=ms, label='PDE Loss (反应扩散方程残差)')
ax2.plot(epochs, ic_loss, color='#2E8B57', marker='^', linewidth=lw, markersize=ms, label='IC Loss (初始条件惩罚)')
ax2.set_yscale('log')
ax2.set_xlabel('迭代次数 (Epochs)', fontsize=12)
ax2.set_ylabel('均方误差 (MSE, Log Scale)', fontsize=12)
ax2.set_title('图 B：控制方程与初始条件残差收敛曲线', fontsize=15, fontweight='bold', pad=12)
ax2.legend(loc='upper right', fontsize=11)
style_axis(ax2)
plt.tight_layout()
plt.savefig("FigB_Physics_Loss.png", transparent=False, bbox_inches='tight')
plt.close(fig2)

# =====================================================================
# 图 C：有效扩散系数反演轨迹 (Parameter Identification)
# =====================================================================
fig3, ax3 = plt.subplots(figsize=(8, 5.5), dpi=300)
# 注意这里的 r 前缀，完美解决 \h 转义报错
ax3.plot(epochs, d_eff, color='#8B008B', marker='D', linewidth=lw, markersize=ms, label=r'反演估计值 ($\hat{D}_{eff}$)')
ax3.axhline(y=true_deff, color='#696969', linestyle='--', linewidth=2.0, label=r'物理真实值 ($1.00 \times 10^{-5}$)')
ax3.set_xlabel('迭代次数 (Epochs)', fontsize=12)
ax3.set_ylabel('有效扩散系数 $D_{eff}$ (cm²/h)', fontsize=12)
ax3.set_title('图 C：有效扩散系数参数辨识轨迹', fontsize=15, fontweight='bold', pad=12)
ax3.legend(loc='upper right', fontsize=11)

# 标注最终反演值
ax3.text(epochs[-1], d_eff[-1] + 0.08e-5, f'{d_eff[-1]:.2e}', 
         ha='center', va='bottom', fontsize=12, fontweight='bold', color='#8B008B')
style_axis(ax3)
plt.tight_layout()
plt.savefig("FigC_Parameter_Tracking.png", transparent=False, bbox_inches='tight')
plt.close(fig3)

# =====================================================================
# 图 D：参数辨识相对误差演化 (Relative Error)
# =====================================================================
fig4, ax4 = plt.subplots(figsize=(8, 5.5), dpi=300)
ax4.fill_between(epochs, error_pct, 0, color='#4682B4', alpha=0.15)
ax4.plot(epochs, error_pct, color='#2F4F4F', marker='v', linewidth=lw, markersize=ms, label='辨识相对误差 (%)')
ax4.set_xlabel('迭代次数 (Epochs)', fontsize=12)
ax4.set_ylabel('绝对相对误差 (%)', fontsize=12)
ax4.set_title('图 D：参数辨识相对误差演化', fontsize=15, fontweight='bold', pad=12)
ax4.legend(loc='upper right', fontsize=11)

# 标注最终误差
ax4.text(epochs[-1], error_pct[-1] + 5, f'{error_pct[-1]:.2f}%', 
         ha='center', va='bottom', fontsize=14, fontweight='bold', color='#2F4F4F')
style_axis(ax4)
plt.tight_layout()
plt.savefig("FigD_Error_Evolution.png", transparent=False, bbox_inches='tight')
plt.close(fig4)

print("全部完成！请在当前目录下查看 FigA 到 FigD 的 4 张高清独立 PNG 图片。")