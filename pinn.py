import os
# 解决 Windows 下 PyTorch 与 NumPy 多线程引擎(OpenMP)冲突导致的闪退问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 物理参数字典
# ==========================================
params = {
    "R": 0.01, "V_ratio": 0.01, "k_L": 0.5, "mu_max": 0.02,
    "K_g": 0.15, "q": 0.02, "gamma": 0.2, "alpha": 5.0,
    "beta": 10.0, "k_PQ_max": 0.05, "k_QN_max": 0.08        
}

# ==========================================
# 2. PINN 神经网络架构定义
# ==========================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)
        )
        
        # 物理先验注入：从 2.0e-5 开始搜索
        self.D_scale = nn.Parameter(torch.tensor([2.0], requires_grad=True))

    def forward(self, t, r):
        t_norm = (t / 336.0) * 2.0 - 1.0
        r_norm = (r / params["R"]) * 2.0 - 1.0
        x = torch.cat([t_norm, r_norm], dim=1)
        
        out = self.net(x)
        # Softplus 替代 ReLU，打通物理量恒为0的死胡同
        G = torch.nn.functional.softplus(out[:, 0:1])
        P = torch.nn.functional.softplus(out[:, 1:2])
        Q = torch.nn.functional.softplus(out[:, 2:3])
        N_cell = torch.nn.functional.softplus(out[:, 3:4])
        return G, P, Q, N_cell

# ==========================================
# 3. 核心损失函数 (Data + Physics + IC)
# ==========================================
def compute_loss(model, t_obs, Ge_obs, t_pde, r_pde, r_ic):
    D_eff_current = model.D_scale * 1e-5
    
    # --------------------------------
    # Part A: Data Loss (罗宾边界)
    # --------------------------------
    t_obs_flux = t_obs.clone().detach().requires_grad_(True)
    r_surf = torch.full_like(t_obs_flux, params["R"], requires_grad=True)
    
    G_surf_pred, _, _, _ = model(t_obs_flux, r_surf)
    dG_dr_surf = torch.autograd.grad(G_surf_pred, r_surf, torch.ones_like(G_surf_pred), create_graph=True)[0]
    
    Ge_pred = G_surf_pred + (D_eff_current / params["k_L"]) * dG_dr_surf
    loss_data = torch.mean((Ge_pred - Ge_obs)**2)
    
    # --------------------------------
    # Part B: Physics Loss (PDE 约束)
    # --------------------------------
    t_pde.requires_grad = True
    r_pde.requires_grad = True
    G, P, Q, N_cell = model(t_pde, r_pde)
    
    dG_dt = torch.autograd.grad(G, t_pde, torch.ones_like(G), create_graph=True)[0]
    dP_dt = torch.autograd.grad(P, t_pde, torch.ones_like(P), create_graph=True)[0]
    dQ_dt = torch.autograd.grad(Q, t_pde, torch.ones_like(Q), create_graph=True)[0]
    
    dG_dr = torch.autograd.grad(G, r_pde, torch.ones_like(G), create_graph=True)[0]
    d2G_dr2 = torch.autograd.grad(dG_dr, r_pde, torch.ones_like(dG_dr), create_graph=True)[0]
    
    laplacian = d2G_dr2 + (2.0 / (r_pde + 1e-6)) * dG_dr
    mu = params["mu_max"] * G / (params["K_g"] + G)
    kPQ = params["k_PQ_max"] * torch.exp(-params["alpha"] * G)
    kQN = params["k_QN_max"] * torch.exp(-params["beta"] * G)
    consumption = params["q"] * (P + params["gamma"] * Q) * G / (params["K_g"] + G)
    
    res_G = dG_dt - D_eff_current * laplacian + consumption
    res_P = dP_dt - (mu * P - kPQ * P)
    res_Q = dQ_dt - (kPQ * P - kQN * Q)
    
    loss_physics = torch.mean(res_G**2) + torch.mean(res_P**2) + torch.mean(res_Q**2)
    
    # --------------------------------
    # Part C: Initial Condition Loss (IC 强效防脱耦)
    # --------------------------------
    t_ic = torch.zeros_like(r_ic)
    G_ic, P_ic, Q_ic, N_ic = model(t_ic, r_ic)
    loss_ic = torch.mean((G_ic - 4.5)**2) + torch.mean((P_ic - 0.1)**2) + torch.mean(Q_ic**2) + torch.mean(N_ic**2)

    return loss_data, loss_physics, loss_ic

# ==========================================
# 4. 主训练循环 (纯 Adam 抗过拟合版本)
# ==========================================
if __name__ == "__main__":
    print("1. 加载包含 5% 高斯噪声的稀疏传感器观测数据...")
    try:
        df_obs = pd.read_csv("observed_noisy_pde.csv")
    except FileNotFoundError:
        print("错误：找不到 observed_noisy_pde.csv！请确保已运行之前的生成脚本。")
        exit()
        
    t_obs = torch.tensor(df_obs['time'].values, dtype=torch.float32).view(-1, 1)
    Ge_obs = torch.tensor(df_obs['Ge_noisy'].values, dtype=torch.float32).view(-1, 1)
    
    print("2. 在微载体内部构建随机配点与初始条件网络...")
    N_colloc = 2500
    t_pde = torch.rand(N_colloc, 1) * 336.0
    r_pde = torch.rand(N_colloc, 1) * params["R"]
    
    r_ic = torch.rand(500, 1) * params["R"]
    
    model = PINN()
    loss_history = []
    
    lambda_physics = 100.0 
    lambda_ic = 1000.0  
    
    print("\n>>> 启动 Adam 优化器进行全流程反演 (实施早停策略，屏蔽二阶过拟合)...")
    optimizer_adam = torch.optim.Adam([
        {'params': model.net.parameters(), 'lr': 1e-3},
        {'params': model.D_scale, 'lr': 1e-2} 
    ])
    
    # 将 Adam 迭代增加到 4000 步，让它自主寻找到物理与数据的最佳平衡点
    for epoch in range(4000):
        optimizer_adam.zero_grad()
        loss_data, loss_physics, loss_ic = compute_loss(model, t_obs, Ge_obs, t_pde, r_pde, r_ic)
        total_loss = loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic
        total_loss.backward()
        optimizer_adam.step()
        
        # 记录每一步的 loss 以便后续作图
        loss_history.append(total_loss.item())
        
        if epoch % 500 == 0:
            current_D = model.D_scale.item() * 1e-5
            print(f"Adam Epoch {epoch:04d} | Data: {loss_data.item():.4e} | Phys: {loss_physics.item():.4e} | IC: {loss_ic.item():.4e} | D_eff: {current_D:.2e}")

    # 获取最后一步的精确信息打印
    loss_data, loss_physics, loss_ic = compute_loss(model, t_obs, Ge_obs, t_pde, r_pde, r_ic)
    current_D = model.D_scale.item() * 1e-5
    print(f"Adam Epoch 3999 | Data: {loss_data.item():.4e} | Phys: {loss_physics.item():.4e} | IC: {loss_ic.item():.4e} | D_eff: {current_D:.2e}")

    # ==========================================
    # 最终结算输出
    # ==========================================
    final_D = model.D_scale.item() * 1e-5
    true_D = 1.00e-05
    error_rate = abs(final_D - true_D) / true_D * 100
    
    print("\n" + "★"*40)
    print("【早停策略：参数辨识完成】")
    print(f"上帝视角的真实 D_eff : {true_D:.2e}")
    print(f"PINN 反演的 D_eff    : {final_D:.2e}")
    print(f"参数反演绝对误差率   : {error_rate:.2f}%")
    print("★"*40)