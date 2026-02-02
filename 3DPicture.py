#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ================= 1. 模型参数定义 (来自 Power_options.py) =================
# 屏幕参数
A_ref = 100
A = 110
beta_br = 612  # mW

# CPU 参数
P_CPUstatic = 121.46  # mW
beta_f1 = 10  # mW / %
beta_f2 = 3.42  # mW / %
a = 15
u0 = 0.4

# GPS 参数
beta_Gon = 429.55  # mW

# 静态参数
P_static = 100  # mW

# 网络参数 (假设绘图时网络处于关闭状态或恒定状态，为了突出手绘图的变量)
P_network_const = 0  # 设为0以匹配"仅关注Brightness/CPU/GPS"的意图

# ================= 2. 向量化计算函数 =================

def get_sigmoid_weight(u):
    """计算 sigmoid 权重 w(u)"""
    return 1 / (1 + np.exp(-a * (u - u0)))

def calculate_power_matrix(brightness_grid, cpu_grid, gps_on=False):
    """
    根据网格计算功率矩阵 Z
    """
    # 1. 屏幕功耗
    p_screen = beta_br * (A / A_ref) * brightness_grid
    
    # 2. CPU 功耗
    u_percent = cpu_grid * 100  # 转换为百分比
    wu = get_sigmoid_weight(cpu_grid)
    
    p_low = P_CPUstatic + beta_f1 * u_percent
    p_high = P_CPUstatic + beta_f2 * u_percent
    
    p_cpu = (1 - wu) * p_low + wu * p_high
    
    # 3. GPS 功耗
    p_gps = beta_Gon if gps_on else 0
    
    # 4. 总功耗
    return p_screen + p_cpu + p_gps + P_static + P_network_const

# ================= 3. 生成绘图数据 =================

# 定义 X (Brightness) 和 Y (CPU Load) 的范围
x = np.linspace(0, 1, 50)  # Brightness 0~1
y = np.linspace(0, 1, 50)  # CPU Load 0~1

# 生成网格
X, Y = np.meshgrid(x, y)

# 计算两个平面的 Z 值 (Power)
Z_gps_off = calculate_power_matrix(X, Y, gps_on=False)
Z_gps_on = calculate_power_matrix(X, Y, gps_on=True)

# ================= 4. 绘制 3D 图 =================

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 绘制 GPS OFF 层 (下方平面) - 使用冷色调
surf1 = ax.plot_surface(X, Y, Z_gps_off, cmap='viridis', alpha=0.6, rstride=2, cstride=2, edgecolor='none')

# 绘制 GPS ON 层 (上方平面) - 使用暖色调或单一颜色区分
# 为了模仿手绘的层级感，这里用半透明的灰色或橙色网格
surf2 = ax.plot_surface(X, Y, Z_gps_on, color='orange', alpha=0.5, rstride=2, cstride=2, edgecolor='none')

# ================= 5. 装饰与标注 =================

# 设置坐标轴标签
ax.set_xlabel('Brightness (0-1)', fontsize=12, labelpad=10)
ax.set_ylabel('CPU Load (0-1)', fontsize=12, labelpad=10)
ax.set_zlabel('Power (mW)', fontsize=12, labelpad=10)

# 设置视角 (调整这里可以改变观察角度)
ax.view_init(elev=25, azim=-45)

# 创建图例 (Surface plot 不自动支持 legend，需要创建代理对象)
legend_elements = [
    Line2D([0], [0], color='orange', lw=4, label='GPS ON'),
    Line2D([0], [0], color=plt.cm.viridis(0.5), lw=4, label='GPS OFF'),
]
ax.legend(handles=legend_elements, loc='upper left')

# 添加标题
plt.title('Power Model: CPU vs Brightness (GPS On/Off) mW', fontsize=14)

plt.tight_layout()
plt.show()
# %%
