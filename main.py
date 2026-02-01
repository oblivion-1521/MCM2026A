#%%
import numpy as np
import matplotlib.pyplot as plt
from nasa_battery_reader import load_battery_data
from battery_physics import cul_SOC, get_OCV

def main():
    # 1. 加载数据
    file_path = '../Battery Data Set/1/B0005.mat' 
    all_cycles = load_battery_data(file_path)
    
    # 选取第 22 个放电周期 (Cycle 22)
    discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
    target_cycle = discharge_cycles[21]
    
    # 2. 获取原始数据
    I_raw = target_cycle['data']['current_measured']
    V_raw = target_cycle['data']['voltage_measured']
    t_raw = target_cycle['data']['time']
    Q = target_cycle['data']['capacity']
    
    # 3. 先计算完整的 SOC
    full_soc = cul_SOC(np.abs(I_raw), t_raw, Q)
    
    # 4. 数据清洗：只保留电流不为0的放电阶段 (去除回弹)
    # 阈值选 0.1A，NASA数据放电电流通常是 ~2A
    valid_indices = np.where(np.abs(I_raw) > 0.1)[0]
    
    soc = full_soc[valid_indices]
    V_meas = V_raw[valid_indices]
    
    # 5. 计算模型 OCV (线性模型)
    ocv_model = get_OCV(soc)
    
    # --- 画图: OCV vs SOC ---
    plt.figure(figsize=(8, 6))
    
    # 画线性模型 OCV
    plt.plot(soc, ocv_model, 'g--', linewidth=2, label='Model OCV (Linear)')
    
    # 画实际测量电压
    plt.plot(soc, V_meas, 'r-', linewidth=2, label='Measured Terminal Voltage $V(t)$')
    
    # 装饰图片
    plt.xlabel('State of Charge (SOC)')
    plt.ylabel('Voltage (V)')
    plt.title(f"OCV-SOC Curve (Cycle {target_cycle['cycle_id']})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置 X 轴从 1 到 0 (符合放电习惯)
    plt.xlim(1.05, -0.05) 
    
    # 填充两条线中间的区域，标注这是我们要拟合的“损耗”
    plt.fill_between(soc, V_meas, ocv_model, color='gray', alpha=0.1, label='Voltage Drop ($IR + U_p$)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
