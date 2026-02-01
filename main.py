#%%
import numpy as np
import matplotlib.pyplot as plt
from nasa_battery_reader import load_battery_data
from battery_physics import cul_SOC, get_OCV, total_voltage_model
from scipy.optimize import curve_fit

def main():
    # 1. 加载数据
    file_path = '../Battery Data Set/2/B0028.mat' 
    all_cycles = load_battery_data(file_path)
    
    # 选取第 22 个放电周期 (Cycle 22)
    discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
    target_cycle = discharge_cycles[21]
    
    # 2. 获取原始数据
    I_raw = target_cycle['data']['current_measured']
    V_raw = target_cycle['data']['voltage_measured']
    t_raw = target_cycle['data']['time']
    T_measured_raw = target_cycle['data']['temperature_measured']
    Q = target_cycle['data']['capacity']
    
    # 3. 先计算完整的 SOC
    full_soc = cul_SOC(np.abs(I_raw), t_raw, Q)
    
    # 4. 数据清洗：只保留电流不为0的放电阶段 (去除回弹)
    # 阈值选 0.1A
    valid_indices = np.where(np.abs(I_raw) > 0.1)[0]
    
    I_valid = np.abs(I_raw[valid_indices])
    soc_valid = full_soc[valid_indices]
    V_meas_valid = V_raw[valid_indices]
    T_valid = T_measured_raw[valid_indices]

    # 5. 计算模型 OCV (线性模型)
    ocv_valid = get_OCV(soc_valid)
    
    # 6. 使用 curve_fit 求解 R_base 和 I0
    # ---------------------------------------------------------
    # 准备 x_data: curve_fit 要求自变量打包在一起
    # 我们将其打包为 (I, OCV, T) 的 tuple
    x_data = (I_valid, ocv_valid, T_valid)
    
    # 初始猜测值 (p0): 
    # R_base 猜 0.1 欧姆 (通常在 0.05-0.2 之间)
    # I0 猜 1.0 A
    p0 = [0.05, 2.0, 31930.0]
    
    # 参数边界 (bounds): ([R_min, I0_min], [R_max, I0_max])
    # R > 0, I0 > 0
    # bounds = ([0.0, 1e-6, 10000.0], [1.5, 20000.0, 60000.0])
    bounds = ([0.0, 1e-6], [1.5, 20000.0])
    print("开始拟合参数 R_base 和 I0 ...")
    
    try:
        popt, pcov = curve_fit(total_voltage_model, x_data, V_meas_valid, p0=p0, bounds=bounds)
        R_fit, I0_fit, Ea_fit = popt
        print(f"拟合成功!")
        print(f"  R_base    = {R_fit:.6f} Ohm")
        print(f"  I0         = {I0_fit:.6f} A")
        print(f"  Ea         = {Ea_fit:.6f} J/mol")
    except Exception as e:
        print(f"拟合失败: {e}")
        return

    # 7. 使用拟合得到的参数计算预测电压 V_pred
    V_pred = total_voltage_model(x_data, R_fit, I0_fit, Ea_fit)
    
    # 8. 误差分析
    rmse = np.sqrt(np.mean((V_meas_valid - V_pred)**2))
    print(f"  RMSE       = {rmse:.6f} V")

    # --- 画图: 结果可视化 ---
    plt.figure(figsize=(12, 10))
    
    # 子图1: 电压拟合对比
    plt.subplot(3, 1, 1)
    plt.plot(t_raw[valid_indices], V_meas_valid, 'k-', linewidth=2, label='Measured Voltage')
    plt.plot(t_raw[valid_indices], V_pred, 'r--', linewidth=2, label='Model Prediction')
    plt.plot(t_raw[valid_indices], ocv_valid, 'g:', label='OCV')
    plt.ylabel('Voltage (V)')
    plt.title(f"Model Fitting with Temperature Correction (Cycle {target_cycle['cycle_id']})")
    plt.legend()
    plt.grid(True)
    
    # 子图2: 温度变化与内阻变化
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    # 绘制温度 (左轴)
    color = 'tab:orange'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(t_raw[valid_indices], T_valid, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 绘制计算出的动态内阻 (右轴)
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    
    # 重新计算一下 R(t) 用于绘图
    T_k = T_valid + 273.15
    R_t_curve = R_fit * np.exp((31930.0 / 8.3145) * (1.0/T_k - 1.0/298.15))
    
    ax2.set_ylabel('Calculated R_int (Ohm)', color=color)
    ax2.plot(t_raw[valid_indices], R_t_curve, color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Temperature Rise vs Internal Resistance Drop")
    plt.grid(True, linestyle='--')

    # 子图3: 误差分析
    plt.subplot(3, 1, 3)
    error = V_meas_valid - V_pred
    plt.plot(t_raw[valid_indices], error, color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel('Error (V)')
    plt.xlabel('Time (s)')
    plt.title("Residual Error (Measured - Predicted)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    main()
# %%
