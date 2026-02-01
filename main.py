#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nasa_battery_reader import load_battery_data
from battery_physics import cul_SOC, get_OCV, total_voltage_model

def get_clean_data_from_cycle(cycle_data, cycle_id):
    I_raw = np.abs(cycle_data['data']['current_measured'])
    V_raw = cycle_data['data']['voltage_measured']
    t_raw = cycle_data['data']['time']
    
    if np.isscalar(cycle_data['data'].get('temperature_measured', 0)):
         T_raw = np.full_like(I_raw, cycle_data['ambient_temperature'])
    else:
         T_raw = cycle_data['data']['temperature_measured']
         
    Q = cycle_data['data']['capacity']
    soc_raw = cul_SOC(I_raw, t_raw, Q)
    
    # 筛选: 放电 (I>0.05) 且 SOC [0.1, 0.95]
    # 稍微放宽一点上限，看能不能拟合得更好
    mask = (I_raw > 0.05) & (soc_raw >= 0.0) & (soc_raw <= 0.9)
    
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
    # 使用 B0038 数据 (正如你图中所示)
    sources = [
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 1}, 
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 37},
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 13}, 
    ]
    
    loaded_files = {}
    combined_data = {'I': [], 'V': [], 'T': [], 'OCV': [], 'SOC': []}
    validation_list = [] 
    
    print(">>> 正在加载并清洗数据...")
    
    for src in sources:
        fpath = src['file']
        c_idx = src['cycle_idx']
        
        if fpath not in loaded_files:
            try:
                loaded_files[fpath] = load_battery_data(fpath)
            except Exception as e:
                print(f"    [Error] Skip {fpath}: {e}")
                continue
        
        all_cycles = loaded_files[fpath]
        discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
        
        if c_idx >= len(discharge_cycles): continue
        target = discharge_cycles[c_idx]
        
        clean = get_clean_data_from_cycle(target, target['cycle_id'])
        if len(clean['V']) == 0: continue
            
        ocv_segment = get_OCV(clean['SOC'])
        clean['OCV'] = ocv_segment
        
        combined_data['I'].append(clean['I'])
        combined_data['V'].append(clean['V'])
        combined_data['T'].append(clean['T'])
        combined_data['OCV'].append(ocv_segment)
        combined_data['SOC'].append(clean['SOC'])
        
        validation_list.append(clean)

    if not combined_data['V']: return

    # 拼接
    train_I = np.concatenate(combined_data['I'])
    train_V = np.concatenate(combined_data['V'])
    train_T = np.concatenate(combined_data['T'])
    train_OCV = np.concatenate(combined_data['OCV'])
    train_SOC = np.concatenate(combined_data['SOC'])
    
    print(f"\n>>> 训练数据: {len(train_V)} 点")
    print(f"    电流均值: {np.mean(train_I):.2f} A")
    
    # === 拟合参数 ===
    x_data = (train_I, train_OCV, train_T, train_SOC)
    
    # 待拟合参数: [R_base, k_low_soc]
    # R_base: 基础内阻，大概 0.05 - 0.2 之间
    # k_low_soc: 低电量内阻增加系数，通常很小
    p0 = [0.1, 0.01]
    bounds = ([0.0, 0.0], [1.0, 1.0])
    
    print("\n>>> 开始拟合参数 (R_base, k_low_soc)...")
    try:
        popt, pcov = curve_fit(total_voltage_model, x_data, train_V, p0=p0, bounds=bounds)
        R_base_fit, k_fit = popt
        
        print(f"Fit Success!")
        print(f"  R_base    = {R_base_fit:.6f} Ohm")
        print(f"  k_low_soc = {k_fit:.6f} Ohm (低SOC影响系数)")
    except Exception as e:
        print(f"拟合失败: {e}")
        return

    # === 可视化 ===
    print("\n>>> 生成验证图...")
    num_plots = len(validation_list)
    plt.figure(figsize=(10, 4 * num_plots))
    
    for i, data in enumerate(validation_list):
        plt.subplot(num_plots, 1, i+1)
        
        seg_inputs = (data['I'], data['OCV'], data['T'], data['SOC'])
        V_pred = total_voltage_model(seg_inputs, R_base_fit, k_fit)
        rmse = np.sqrt(np.mean((data['V'] - V_pred)**2))
        
        # 计算内阻分量用于展示
        temp_corr = np.exp((31930.0 / 8.3145) * (1.0/(data['T']+273.15) - 1.0/298.15))
        
        # 基础压降 (Base Ohmic)
        V_drop_base = data['I'] * R_base_fit * temp_corr
        # 极化压降 (Low SOC effect)
        V_drop_soc = data['I'] * (k_fit * (1.0/data['SOC'] - 1.0)) * temp_corr
        
        plt.plot(data['SOC'], data['V'], 'k-', label='Measured', linewidth=1.5)
        plt.plot(data['SOC'], V_pred, 'r--', label=f'Model (RMSE={rmse:.3f})', linewidth=1.5)
        
        # 绘制 OCV
        plt.plot(data['SOC'], data['OCV'], 'g:', label='OCV', alpha=0.4)
        
        # 堆叠填充图
        plt.fill_between(data['SOC'], data['OCV'], data['OCV'] - V_drop_base, 
                         color='blue', alpha=0.1, label='Base Resistance Drop')
        plt.fill_between(data['SOC'], data['OCV'] - V_drop_base, V_pred, 
                         color='orange', alpha=0.2, label='Low-SOC Polarization')
        
        plt.gca().invert_xaxis()
        plt.title(f"Cycle {data['cycle_id']} - I_avg={np.mean(data['I']):.1f}A")
        plt.xlabel('SOC')
        plt.ylabel('Voltage (V)')
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
