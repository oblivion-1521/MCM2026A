#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# 引入之前的模块
from nasa_battery_reader import load_battery_data
from battery_physics import (
    cul_SOC, get_OCV, total_voltage_model, 
    get_internal_resistance, thermal_model_simulation, simulate_power_depletion
)
# 引入新的功率计算模块
from Power_options import read_and_compute_power

def get_clean_data_from_cycle(cycle_data, cycle_id):
    I_raw = np.abs(cycle_data['data']['current_measured'])
    V_raw = cycle_data['data']['voltage_measured']
    t_raw = cycle_data['data']['time']
    
    if np.isscalar(cycle_data['data'].get('temperature_measured', 0)):
         T_raw = np.full_like(I_raw, cycle_data['ambient_temperature'])
    else:
         T_raw = cycle_data['data']['temperature_measured']
    
    T_env_val = cycle_data['ambient_temperature']
    T_env_raw = np.full_like(t_raw, T_env_val)
         
    Q = cycle_data['data']['capacity']
    soc_raw = cul_SOC(I_raw, t_raw, Q)
    
    # 筛选数据用于拟合
    mask = (I_raw > 0.05) & (soc_raw >= 0.0) & (soc_raw <= 0.9)
    
    clean_data = {
        'cycle_id': cycle_id,
        'I': I_raw[mask],
        'V': V_raw[mask],
        'T': T_raw[mask],
        'T_env': T_env_raw[mask],
        'SOC': soc_raw[mask],
        't': t_raw[mask]
    }
    return clean_data

# === 包装函数：用于热参数拟合 ===
def thermal_fit_wrapper(packed_inputs, mCp, c):
    R_base, k_low_soc, data_list = packed_inputs
    all_preds = []
    
    for data in data_list:
        sim_inputs = (
            data['t'], 
            data['I'], 
            data['T_env'], 
            data['SOC'], 
            data['T'][0] 
        )
        T_pred = thermal_model_simulation(sim_inputs, mCp, c, R_base, k_low_soc)
        all_preds.append(T_pred)
        
    return np.concatenate(all_preds)


def main(device_id='DEV_0013', capacity_sim=3.497):
    # ==========================================
    # Step 1: 加载 NASA 电池数据 (用于训练参数)
    # ==========================================
    # 请确保路径正确
    sources = [
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 0}, 
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 1}, 
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 2},
    ]
    
    loaded_files = {}
    combined_data = {'I': [], 'V': [], 'T': [], 'OCV': [], 'SOC': []}
    validation_list = [] 
    
    print(">>> [Step 1] Loading Training Data (NASA B0038)...")
    
    for src in sources:
        fpath = src['file']
        c_idx = src['cycle_idx']
        
        try:
            if fpath not in loaded_files:
                loaded_files[fpath] = load_battery_data(fpath)
            
            all_cycles = loaded_files[fpath]
            discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
            
            if c_idx >= len(discharge_cycles): continue
            target = discharge_cycles[c_idx]
            
            clean = get_clean_data_from_cycle(target, target['cycle_id'])
            if len(clean['V']) == 0: continue
                
            clean['OCV'] = get_OCV(clean['SOC'])
            
            combined_data['I'].append(clean['I'])
            combined_data['V'].append(clean['V'])
            combined_data['T'].append(clean['T'])
            combined_data['OCV'].append(clean['OCV'])
            combined_data['SOC'].append(clean['SOC'])
            validation_list.append(clean)
            
        except FileNotFoundError:
            print(f"Warning: File {fpath} not found. Skipping.")
            continue

    if not combined_data['V']: 
        print("Error: No valid data loaded. Check paths.")
        return

    # 拼接训练数据
    train_I = np.concatenate(combined_data['I'])
    train_V = np.concatenate(combined_data['V'])
    train_T = np.concatenate(combined_data['T'])
    train_OCV = np.concatenate(combined_data['OCV'])
    train_SOC = np.concatenate(combined_data['SOC'])
    
    # ==========================================
    # Step 2: 拟合电气参数 (R_base, k_low_soc)
    # ==========================================
    print("\n>>> [Step 2] Fitting Electrical Parameters...")
    
    x_data_elec = (train_I, train_OCV, train_T, train_SOC)
    p0_elec = [0.1, 0.01]
    bounds_elec = ([0.0, 0.0], [1.0, 1.0])
    
    try:
        popt_elec, _ = curve_fit(total_voltage_model, x_data_elec, train_V, p0=p0_elec, bounds=bounds_elec)
        R_base_fit, k_fit = popt_elec
        print(f"  R_base    = {R_base_fit:.6f} Ohm")
        print(f"  k_low_soc = {k_fit:.6f} Ohm")
    except Exception as e:
        print(f"  Voltage Fit Failed: {e}")
        return

    # ==========================================
    # Step 3: 拟合热参数 (mCp, c)
    # ==========================================
    print("\n>>> [Step 3] Fitting Thermal Parameters...")
    
    packed_inputs = (R_base_fit, k_fit, validation_list)
    train_T_all = np.concatenate([d['T'] for d in validation_list])
    
    def fit_func_thermal(dummy_x, mCp, c):
        return thermal_fit_wrapper(packed_inputs, mCp, c)
    
    p0_therm = [45.0, 0.04] 
    bounds_therm = ([1.0, 0.001], [500.0, 10.0])
    
    try:
        dummy_x = np.zeros_like(train_T_all)
        popt_therm, _ = curve_fit(fit_func_thermal, dummy_x, train_T_all, p0=p0_therm, bounds=bounds_therm)
        mCp_fit, c_fit = popt_therm
        print(f"  mCp (Thermal Mass) = {mCp_fit:.4f} J/K")
        print(f"  c   (Heat Transfer)= {c_fit:.4f} W/K")
    except Exception as e:
        print(f"  Thermal Fit Failed: {e}")
        # 如果拟合失败，使用默认值防止程序崩溃
        mCp_fit, c_fit = 45.0, 0.05
        print("  Using default thermal parameters.")

    # ==========================================
    # Step 4: 读取真实功率数据并仿真
    # ==========================================
    print("\n>>> [Step 4] Loading Real-World Power Profile from Power_options...")
    
    # 1. 读取 CSV 数据并计算功率
    try:
        # 假设 CSV 路径如下，如果不在这里请修改
        csv_path = '../battery_dataset_sample.csv'
        df_power = read_and_compute_power(csv_path, device_id=device_id)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    print(f"  Loaded {len(df_power)} rows for {device_id}.")
    
    # 2. 数据预处理
    # 提取时间并转换为相对秒数 (t=0 at start)
    t_start = df_power['timestamp'].iloc[0]
    df_power['time_sec'] = (df_power['timestamp'] - t_start).dt.total_seconds()
    
    # 提取模型计算的总功率 (mW) 并转换为 (W)
    # 注意：battery_physics 模型内部使用的是标准单位 (Amps, Volts, Ohms, Watts)
    P_source_watts = df_power['P_total_model_mW'].values / 1000.0
    t_source_sec = df_power['time_sec'].values
    
    # 3. 创建仿真时间轴 (Resampling)
    # 原始数据是稀疏的（每几分钟一个点），我们需要 1秒步长的连续数据进行积分
    sim_dt = 1.0 
    t_max = t_source_sec[-1]
    t_sim = np.arange(0, t_max, sim_dt)
    
    # 阶跃插值生成 P(t)
    # 使用阶跃插值（零阶保持），假设两个日志点之间功率保持不变
    # kind='zero' 或 'previous' 表示取前一个点的值
    interp_func = interp1d(t_source_sec, P_source_watts, kind='zero', 
                           bounds_error=False, fill_value=(P_source_watts[0], P_source_watts[-1]))
    P_sim = interp_func(t_sim)


    
    # 环境参数
    T_env_sim = 25.0 
    T_init_sim = 25.0
    SOC_init_sim = 1.0 # 假设满电开始
    V_cutoff_sim = 3.0 # 截止电压
    
    print(f"  Starting Simulation: Duration={t_max/3600:.2f} hours, Capacity={capacity_sim} Ah")
    
    # 5. 运行仿真
    sim_result = simulate_power_depletion(
        t_sim, P_sim, 
        SOC_init_sim, T_init_sim, T_env_sim, capacity_sim,
        R_base_fit, k_fit, mCp_fit, c_fit,
        V_cutoff=V_cutoff_sim
    )
    
  # ==========================================
    # Step 5: 结果可视化 (包含模型拟合验证 和 现实功率仿真)
    # ==========================================
    print("\n>>> Generating Visualizations...")

    # 获取仿真结束的具体时间
    t_end = sim_result['t'][-1]

    # --- 1. 模型拟合验证图 (略，保持你之前的代码不变) ---
    # ... (这部分代码建议保留在你原本的位置) ...

    # --- 2. 现实功耗耗尽仿真图 (针对 Power_options 的 CSV 数据) ---
    plt.figure(figsize=(14, 12))
    
    # 子图 A: 输入功率与计算出的电流
    plt.subplot(3, 1, 1)
    
    # 【核心修改点】：过滤原始日志点，只保留仿真运行时间范围内的数据
    mask = t_source_sec <= t_end
    t_log_filtered = t_source_sec[mask]
    P_log_filtered = P_source_watts[mask]

    # 绘制过滤后的原始采样点
    plt.scatter(t_log_filtered, P_log_filtered, color='black', marker='x', label='Original Log Points (W)', alpha=0.6)
    
    # 绘制插值后的连续功率曲线 (使用 sim_result 中的数据，它是已经根据耗尽时间截断好的)
    plt.plot(sim_result['t'], sim_result['P'], color='green', alpha=0.4, label='Interpolated P_in (W)')
    
    # 绘制计算出的响应电流
    plt.plot(sim_result['t'], sim_result['I'], color='blue', linestyle='--', label='Calculated I_out (A)')
    
    plt.ylabel('Power (W) / Current (A)')
    plt.title('Simulation Part 1: Real-world Power Profile & Battery Current Response')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 设置 X 轴范围，使三个图的横轴对齐到仿真结束时刻
    plt.xlim(-1000, t_end + 1000) 
    
    # 子图 B: 电压下降与 SOC 消耗
    plt.subplot(3, 1, 2)
    ax_v = plt.gca()
    ax_soc = ax_v.twinx()
    
    ax_v.plot(sim_result['t'], sim_result['V'], 'red', label='Terminal Voltage (V)', linewidth=1.5)
    ax_v.axhline(y=V_cutoff_sim, color='red', linestyle=':', label='Cutoff Voltage')
    ax_soc.plot(sim_result['t'], sim_result['SOC']*100, 'black', label='SOC (%)', linewidth=2)
    
    ax_v.set_ylabel('Voltage (V)', color='red')
    ax_soc.set_ylabel('SOC (%)', color='black')
    ax_v.set_title(f'Simulation Part 2: Voltage and SOC Depletion (Capacity: {capacity_sim} Ah)')
    ax_v.set_xlim(-1000, t_end + 1000)
    
    h1, l1 = ax_v.get_legend_handles_labels()
    h2, l2 = ax_soc.get_legend_handles_labels()
    ax_v.legend(h1+h2, l1+l2, loc='lower left')
    ax_v.grid(True, alpha=0.3)
    
    # 子图 C: 电池发热情况
    plt.subplot(3, 1, 3)
    plt.plot(sim_result['t'], sim_result['T'], color='orange', linewidth=2, label='Simulated Battery Temp')
    plt.axhline(y=T_env_sim, color='gray', linestyle='--', label='Ambient Temperature')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.title('Simulation Part 3: Battery Thermal Profile during Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1000, t_end + 1000)
    
    plt.tight_layout()
    plt.show()

    # 打印仿真统计信息 (保持不变)
    duration_hrs = t_end / 3600.0
    print(f"\n[Simulation Summary]")
    print(f"Total simulated time: {t_end:.1f} s ({duration_hrs:.2f} hours)")
    print(f"Final SOC: {sim_result['SOC'][-1]*100:.2f} %") 

if __name__ == "__main__":
    device_id = 'DEV_0030'  # 可以修改为其他设备ID
    capacity_sim = 3.951
    main(device_id=device_id, capacity_sim=capacity_sim)
# %%
