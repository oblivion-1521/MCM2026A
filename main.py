#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nasa_battery_reader import load_battery_data
from battery_physics import cul_SOC, get_OCV, total_voltage_model

def get_clean_data_from_cycle(cycle_data, cycle_id):
    """
    清洗数据: SOC[0.1, 0.9], Discharge Only, Extract T/V/I
    """
    I_raw = np.abs(cycle_data['data']['current_measured'])
    V_raw = cycle_data['data']['voltage_measured']
    t_raw = cycle_data['data']['time']
    
    if np.isscalar(cycle_data['data'].get('temperature_measured', 0)):
         T_raw = np.full_like(I_raw, cycle_data['ambient_temperature'])
    else:
         T_raw = cycle_data['data']['temperature_measured']
         
    Q = cycle_data['data']['capacity']
    
    # 1. 计算 SOC
    soc_raw = cul_SOC(I_raw, t_raw, Q)
    
    # 2. 筛选掩码: 放电且 SOC 在有效区间
    mask = (I_raw > 0.05) & (soc_raw >= 0.1) & (soc_raw <= 0.9)
    
    clean_data = {
        'cycle_id': cycle_id,
        'I': I_raw[mask],
        'V': V_raw[mask],
        'T': T_raw[mask],
        'SOC': soc_raw[mask],
        't': t_raw[mask]
    }
    return clean_data

def main():
    # === 1. 定义数据源 ===
    # 建议尽量混合不同温度或不同电池的数据以增加鲁棒性
    sources = [
        {'file': '../Battery Data Set/1/B0005.mat', 'cycle_idx': 22}, 
        {'file': '../Battery Data Set/1/B0005.mat', 'cycle_idx': 50}, # 增加一个老化程度不同的
        {'file': '../Battery Data Set/1/B0006.mat', 'cycle_idx': 20}, 
    ] 
    
    loaded_files = {}
    combined_I, combined_V, combined_T, combined_OCV = [], [], [], []
    validation_list = [] 
    
    print(">>> 正在加载并清洗数据...")
    
    # === 2. 读取与清洗 ===
    for src in sources:
        fpath = src['file']
        c_idx = src['cycle_idx']
        
        if fpath not in loaded_files:
            try:
                loaded_files[fpath] = load_battery_data(fpath)
            except Exception as e:
                print(f"    [Error] 无法读取 {fpath}: {e}")
                continue
        
        all_cycles = loaded_files[fpath]
        discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
        
        if c_idx >= len(discharge_cycles): continue
            
        target = discharge_cycles[c_idx]
        clean = get_clean_data_from_cycle(target, target['cycle_id'])
        
        if len(clean['V']) == 0: continue
            
        # 计算 OCV
        ocv_segment = get_OCV(clean['SOC'])
        clean['OCV'] = ocv_segment 
        
        combined_I.append(clean['I'])
        combined_V.append(clean['V'])
        combined_T.append(clean['T'])
        combined_OCV.append(ocv_segment)
        validation_list.append(clean)

    if not combined_V: return

    train_I = np.concatenate(combined_I)
    train_V = np.concatenate(combined_V)
    train_T = np.concatenate(combined_T)
    train_OCV = np.concatenate(combined_OCV)
    
    print(f"\n>>> 训练数据: {len(train_V)} 点")
    
    # === 3. 约束拟合 (核心修改部分) ===
    x_data = (train_I, train_OCV, train_T)
    
    # --- 设置强约束 ---
    # R_base: 0.0 ~ 1.0 欧姆 (通常 18650 在 0.02-0.1 之间)
    # I0:     0.1 ~ 2.0 A (强制限定在经验范围内)
    
    # lower_bounds = [R_min, I0_min]
    lower_bounds = [0.0, 0.01]
    # upper_bounds = [R_max, I0_max]
    upper_bounds = [1.0, 2.0]
    
    bounds = (lower_bounds, upper_bounds)
    
    # 初始猜测 (确保在 Bounds 内部)
    p0 = [0.1, 0.5] 
    
    print("\n>>> 开始拟合参数 (R_base, I0)...")
    print(f"    约束范围: I0 ∈ [{lower_bounds[1]}, {upper_bounds[1]}] A")
    
    try:
        popt, pcov = curve_fit(total_voltage_model, x_data, train_V, p0=p0, bounds=bounds)
        R_base_fit, I0_fit = popt
        
        print(f"Fit Success!")
        print(f"  R_base (25C) = {R_base_fit:.6f} Ohm")
        print(f"  I0           = {I0_fit:.6f} A")
        
        # 检查边界触碰
        if np.isclose(I0_fit, lower_bounds[1]) or np.isclose(I0_fit, upper_bounds[1]):
            print("  [警告] I0 触碰到了边界值！这说明数据可能无法有效区分 Tafel 项。")
            print("         模型可能将 I0 当作一个纯粹的截距调整参数。")
            
    except Exception as e:
        print(f"拟合失败: {e}")
        return

    # === 4. 验证画图 ===
    print("\n>>> 生成验证图...")
    num_plots = len(validation_list)
    plt.figure(figsize=(10, 4 * num_plots))
    
    for i, data in enumerate(validation_list):
        plt.subplot(num_plots, 1, i+1)
        seg_inputs = (data['I'], data['OCV'], data['T'])
        V_seg_pred = total_voltage_model(seg_inputs, R_base_fit, I0_fit)
        seg_rmse = np.sqrt(np.mean((data['V'] - V_seg_pred)**2))
        
        plt.plot(data['SOC'], data['V'], 'k-', label='Measured', linewidth=1.5)
        plt.plot(data['SOC'], V_seg_pred, 'r--', label=f'Model (RMSE={seg_rmse:.4f})', linewidth=1.5)
        # 增加显示 OCV-IR (纯电阻压降) 以展示 Tafel 的贡献
        # 纯电阻电压 = OCV - I*R(T)
        R_T = R_base_fit * np.exp((31930.0 / 8.3145) * (1.0/(data['T']+273.15) - 1.0/298.15))
        V_ohm_only = data['OCV'] - data['I'] * R_T
        plt.plot(data['SOC'], V_ohm_only, 'b:', label='OCV - IR (No Tafel)', alpha=0.4)

        plt.gca().invert_xaxis()
        plt.title(f"Cycle {data['cycle_id']} (RMSE={seg_rmse:.4f}V)")
        plt.xlabel('SOC')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
