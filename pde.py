import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ==========================================
# 1. 物理与生物学核心参数 (对齐 PINN 需求)
# ==========================================
params = {
    "R": 0.01,              # 微载体半径 (cm)
    "N_grid": 20,           
    "D_eff": 1.0e-5,        # 【大降】模拟 ECM 基质严重堵塞孔道，扩散极度受限
    "k_L": 0.5,             
    "V_ratio": 0.01,        # 【核心改动】1%的体积占比，模拟真实的工业大罐，外部营养不会瞬间崩溃
    "G_e0": 4.5,            
    "mu_max": 0.02,         
    "K_g": 0.15,            
    "q": 0.02,              # 【调高】确保指数生长期后，局部细胞呈现暴食状态
    "gamma": 0.2,           
    "alpha": 5.0,           
    "beta": 10.0,           
    "k_PQ_max": 0.05,       
    "k_QN_max": 0.08        
}

# 空间离散化设置
N = params["N_grid"]
r_grid = np.linspace(0, params["R"], N)
dr = r_grid[1] - r_grid[0]
a_specific_area = params["V_ratio"] * 3 / params["R"] # 宏观比表面积

# ==========================================
# 2. PDE 转 ODE 核心系统 (线方法 Method of Lines)
# ==========================================
def pde_system(t, y):
    # 状态向量解包 (共 4N + 1 个变量)
    # y[0] 是外部浓度 G_e
    # y[1 : N+1] 是内部浓度场 G(r)
    # 后面依次是 P(r), Q(r), N_cells(r)
    Ge = y[0]
    G = y[1 : N+1]
    P = y[N+1 : 2*N+1]
    Q = y[2*N+1 : 3*N+1]
    N_cells = y[3*N+1 : 4*N+1]
    
    # 截断极小值，防止数值不稳定计算出负浓度
    G = np.maximum(G, 1e-6)
    
    # 1. 细胞状态动力学 (平滑的指数衰减，PINN 狂喜)
    mu = params["mu_max"] * G / (params["K_g"] + G)
    kPQ = params["k_PQ_max"] * np.exp(-params["alpha"] * G)
    kQN = params["k_QN_max"] * np.exp(-params["beta"] * G)
    consumption = params["q"] * (P + params["gamma"] * Q) * G / (params["K_g"] + G)
    
    dP_dt = mu * P - kPQ * P
    dQ_dt = kPQ * P - kQN * Q
    dN_dt = kQN * Q
    
    # 2. 营养扩散 PDE 差分离散化
    dG_dt = np.zeros(N)
    
    # -- 球心边界条件 (r=0): 对称边界 dG/dr = 0
    # 利用洛必达法则处理 1/r 奇异性，Laplacian退化为 3 * d2G/dr2
    dG_dt[0] = params["D_eff"] * 3 * (2 * (G[1] - G[0]) / dr**2) - consumption[0]
    
    # -- 内部连续网格 (0 < r < R)
    for i in range(1, N-1):
        d2G_dr2 = (G[i+1] - 2*G[i] + G[i-1]) / dr**2
        dG_dr = (G[i+1] - G[i-1]) / (2 * dr)
        laplacian = d2G_dr2 + (2.0 / r_grid[i]) * dG_dr
        dG_dt[i] = params["D_eff"] * laplacian - consumption[i]
        
    # -- 表面边界条件 (r=R): Robin 液膜传质
    # 引入虚拟节点 (Ghost Point) 匹配外部通量
    G_ghost = G[N-2] + (2 * dr * params["k_L"] / params["D_eff"]) * (Ge - G[N-1])
    d2G_dr2_surf = (G_ghost - 2*G[N-1] + G[N-2]) / dr**2
    dG_dr_surf = (G_ghost - G[N-2]) / (2 * dr)
    laplacian_surf = d2G_dr2_surf + (2.0 / params["R"]) * dG_dr_surf
    dG_dt[N-1] = params["D_eff"] * laplacian_surf - consumption[N-1]
    
    # 3. 外部培养基消耗 ODE
    dGe_dt = -a_specific_area * params["k_L"] * (Ge - G[N-1])
    
    return np.concatenate(([dGe_dt], dG_dt, dP_dt, dQ_dt, dN_dt))

# ==========================================
# 3. 求解与数据导出
# ==========================================
if __name__ == "__main__":
    print("开始进行反应-扩散时空 PDE 求解，请稍候...")
    
    # 初始状态设置
    # 全部细胞初始处于增殖态 (1.0 代表归一化后的基础密度)
    y0 = np.concatenate((
        [params["G_e0"]],               # 外部初始 G_e
        np.full(N, params["G_e0"]),     # 内部 G 场初始化
        np.full(N, 1.0),                # P 场初始化
        np.full(N, 0.0),                # Q 场初始化
        np.full(N, 0.0)                 # N 场初始化
    ))
    
    # 仿真 336 小时 (14天)，截取 150 个时间点
    t_span = (0, 336)
    t_eval = np.linspace(t_span[0], t_span[1], 150)
    
    # 使用 BDF 或 Radau 处理刚性方程组
    sol = solve_ivp(pde_system, t_span, y0, method='BDF', t_eval=t_eval)
    
    if sol.success:
        print("求解成功！正在组装长格式 DataFrame...")
        
        # 解析高维矩阵为可读的长格式表
        records = []
        for j, t_val in enumerate(sol.t):
            Ge_val = sol.y[0, j]
            for i, r_val in enumerate(r_grid):
                records.append({
                    "time": t_val,
                    "r": r_val,
                    "Ge": Ge_val,
                    "G": sol.y[1 + i, j],
                    "P": sol.y[1 + N + i, j],
                    "Q": sol.y[1 + 2*N + i, j],
                    "N": sol.y[1 + 3*N + i, j]
                })
                
        df = pd.DataFrame(records)
        df.to_csv("ground_truth_pde.csv", index=False)
        print(f"数据已保存至 ground_truth_pde.csv (数据规模: {df.shape})")
        print("现在的 CSV 拥有完整的 [time, r, G, P, Q, N] 维度！可以去画热力图了！")
    else:
        print("求解器崩溃了，请检查物理参数设定。")