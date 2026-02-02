#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nasa_battery_reader import load_battery_data
from battery_physics import (
    cul_SOC, get_OCV, total_voltage_model, 
    get_internal_resistance, thermal_model_simulation, simulate_power_depletion
)

def get_clean_data_from_cycle(cycle_data, cycle_id):
    I_raw = np.abs(cycle_data['data']['current_measured'])
    V_raw = cycle_data['data']['voltage_measured']
    t_raw = cycle_data['data']['time']
    
    if np.isscalar(cycle_data['data'].get('temperature_measured', 0)):
         T_raw = np.full_like(I_raw, cycle_data['ambient_temperature'])
    else:
         T_raw = cycle_data['data']['temperature_measured']
    
    # 确保 T_env 存在，NASA数据集中 ambient_temperature 通常是标量
    T_env_val = cycle_data['ambient_temperature']
    T_env_raw = np.full_like(t_raw, T_env_val)
         
    Q = cycle_data['data']['capacity']
    soc_raw = cul_SOC(I_raw, t_raw, Q)
    
    # 筛选: 放电 (I>0.05) 且 SOC [0.1, 1.0] 
    # 热模型建议保留到放电开始，否则初始温度不好定
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
# 因为 curve_fit 只接受一个输入数组 x，我们需要把所有片段的数据打包
def thermal_fit_wrapper(packed_inputs, mCp, c):
    """
    这是一个中间层，用于处理多段数据的拼接预测。
    packed_inputs 包含: (R_base, k_low_soc, list_of_cycle_data)
    """
    # 解包固定参数和数据
    R_base, k_low_soc, data_list = packed_inputs
    
    all_preds = []
    
    for data in data_list:
        # 准备单个片段的仿真输入
        # inputs: (time_arr, I_arr, T_env_arr, SOC_arr, T_init)
        sim_inputs = (
            data['t'], 
            data['I'], 
            data['T_env'], 
            data['SOC'], 
            data['T'][0] # 使用真实测量的初始温度作为仿真起点
        )
        
        # 运行仿真
        T_pred = thermal_model_simulation(sim_inputs, mCp, c, R_base, k_low_soc)
        all_preds.append(T_pred)
        
    return np.concatenate(all_preds)


def main():
    # 1. 数据源
    sources = [
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 0}, 
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 1}, 
        {'file': '../Battery Data Set/3/B0038.mat', 'cycle_idx': 2},
    ]
    
    loaded_files = {}
    combined_data = {'I': [], 'V': [], 'T': [], 'OCV': [], 'SOC': []}
    validation_list = [] 
    
    print(">>> [Step 1] Loading Data...")
    
    for src in sources:
        fpath = src['file']
        c_idx = src['cycle_idx']
        
        if fpath not in loaded_files:
            loaded_files[fpath] = load_battery_data(fpath)
        
        all_cycles = loaded_files[fpath]
        discharge_cycles = [c for c in all_cycles if c['type'] == 'discharge']
        
        if c_idx >= len(discharge_cycles): continue
        target = discharge_cycles[c_idx]
        
        clean = get_clean_data_from_cycle(target, target['cycle_id'])
        if len(clean['V']) == 0: continue
            
        clean['OCV'] = get_OCV(clean['SOC'])
        
        # 收集用于电压拟合
        combined_data['I'].append(clean['I'])
        combined_data['V'].append(clean['V'])
        combined_data['T'].append(clean['T'])
        combined_data['OCV'].append(clean['OCV'])
        combined_data['SOC'].append(clean['SOC'])
        
        # 收集用于热拟合 (保存列表结构)
        validation_list.append(clean)

    if not combined_data['V']: return

    # 拼接训练数据 (电压部分)
    train_I = np.concatenate(combined_data['I'])
    train_V = np.concatenate(combined_data['V'])
    train_T = np.concatenate(combined_data['T'])
    train_OCV = np.concatenate(combined_data['OCV'])
    train_SOC = np.concatenate(combined_data['SOC'])
    
    # ==========================================
    # Step 2: 拟合电压/内阻参数 (R_base, k_low_soc)
    # ==========================================
    print("\n>>> [Step 2] Fitting Electrical Parameters (R_base, k_low_soc)...")
    
    x_data_elec = (train_I, train_OCV, train_T, train_SOC)
    p0_elec = [0.1, 0.01]
    bounds_elec = ([0.0, 0.0], [1.0, 1.0])
    
    try:
        popt_elec, _ = curve_fit(total_voltage_model, x_data_elec, train_V, p0=p0_elec, bounds=bounds_elec)
        R_base_fit, k_fit = popt_elec
        print(f"  Fit Success!")
        print(f"  R_base    = {R_base_fit:.6f} Ohm")
        print(f"  k_low_soc = {k_fit:.6f} Ohm")
    except Exception as e:
        print(f"  Voltage Fit Failed: {e}")
        return

    # ==========================================
    # Step 3: 拟合热参数 (mCp, c)
    # ==========================================
    print("\n>>> [Step 3] Fitting Thermal Parameters (mCp, c)...")
    
    # 准备热拟合的数据包
    # 我们不直接传递 huge array，而是传递 cycle 列表，以便 wrapper 分别积分
    # curve_fit 要求 xdata 必须是 array-like 且长度与 ydata 一致，这里我们用个 trick
    # 这里的 x_data 实际上不被 func 直接使用，我们通过 partial 或 lambda 闭包传递真实结构
    # 但最标准的做法是利用 wrapper
    
    packed_inputs = (R_base_fit, k_fit, validation_list)
    
    # 目标 Y 值：所有片段的真实温度拼在一起
    train_T_all = np.concatenate([d['T'] for d in validation_list])
    
    # 定义 lambda 函数，固定 inputs，只暴露 (x, mCp, c) 给 curve_fit
    # 注意：curve_fit 调用的函数形式 func(x, p1, p2)，x 必须存在
    # 这里我们传一个伪造的 x，在 wrapper 里直接忽略 x，使用 closed-over 的 packed_inputs
    # 为了兼容性，wrapper 还是设计成接收 packed_inputs
    
    # 修正：scipy 的 curve_fit 不支持传递复杂对象作为 x。
    # 我们改用 lambda 闭包，让 curve_fit 认为 x 是 dummy
    def fit_func_thermal(dummy_x, mCp, c):
        return thermal_fit_wrapper(packed_inputs, mCp, c)
    
    # 初始猜测
    # mCp: 18650电池约 45g, Cp~1000 J/kgK -> mCp ~ 45 J/K
    # c: h*A, A~0.004 m2, h~10 W/m2K -> c ~ 0.04 W/K
    p0_therm = [45.0, 0.04] 
    bounds_therm = ([1.0, 0.001], [500.0, 10.0])
    
    try:
        # dummy_x 长度必须和 train_T_all 一致
        dummy_x = np.zeros_like(train_T_all)
        popt_therm, _ = curve_fit(fit_func_thermal, dummy_x, train_T_all, p0=p0_therm, bounds=bounds_therm)
        mCp_fit, c_fit = popt_therm
        
        print(f"  Fit Success!")
        print(f"  mCp (Thermal Mass) = {mCp_fit:.4f} J/K")
        print(f"  c   (Heat Transfer)= {c_fit:.4f} W/K")
        
    except Exception as e:
        print(f"  Thermal Fit Failed: {e}")
        return

    # ==========================================
    # Step 4: 结果可视化
    # ==========================================
    print("\n>>> Generating Plots...")
    num_plots = len(validation_list)
    plt.figure(figsize=(12, 5 * num_plots))
    
    for i, data in enumerate(validation_list):
        # 1. 预测电压
        v_inputs = (data['I'], data['OCV'], data['T'], data['SOC'])
        V_pred = total_voltage_model(v_inputs, R_base_fit, k_fit)
        
        # 2. 预测温度 (使用拟合出的热参数)
        sim_inputs = (data['t'], data['I'], data['T_env'], data['SOC'], data['T'][0])
        T_pred = thermal_model_simulation(sim_inputs, mCp_fit, c_fit, R_base_fit, k_fit)
        
        # --- 子图 1: 电压 ---
        ax1 = plt.subplot(num_plots, 2, i*2 + 1)
        ax1.plot(data['SOC'], data['V'], 'k-', label='Measured V')
        ax1.plot(data['SOC'], V_pred, 'r--', label='Model V')
        ax1.set_title(f"Cycle {data['cycle_id']} - Voltage")
        ax1.set_xlabel('SOC')
        ax1.set_ylabel('Voltage (V)')
        ax1.invert_xaxis()
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # --- 子图 2: 温度 ---
        ax2 = plt.subplot(num_plots, 2, i*2 + 2)
        # 用时间轴画温度更直观
        t_norm = data['t'] - data['t'][0]
        ax2.plot(t_norm, data['T'], 'k-', label='Measured T')
        ax2.plot(t_norm, T_pred, 'r--', label='Model T')
        ax2.set_title(f"Cycle {data['cycle_id']} - Temperature Fit")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 添加参数文本
        info_text = (f"mCp={mCp_fit:.1f}\nc={c_fit:.3f}")
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    print("\n>>> [Step 5] Running Constant Power Depletion Simulation...")
    
    # 1. 定义仿真条件
    # 假设一个恒定功率负载，例如 4W (手机高负载游戏场景)
    # 或者构造一个随时间变化的 P(t)
    sim_duration = 3600 * 2  # 模拟 2 小时
    dt = 1.0                 # 时间步长 1s
    t_sim = np.arange(0, sim_duration, dt)
    
    # 构造 P(t): 前10分钟 1W (待机), 之后 4.5W (打游戏)
    P_sim = np.ones_like(t_sim) * 1.0 
    P_sim[600:] = 4.5 
    
    # 电池参数 (从数据集中读取容量，或者假设一个值)
    # NASA B0038 容量约为 1.1 Ah 左右 (根据数据)
    # 这里我们取第一个 cycle 的容量
    real_capacity = validation_list[0]['I'].sum() * (validation_list[0]['t'][1]-validation_list[0]['t'][0]) / 3600 / (1-validation_list[0]['SOC'][-1])
    # 粗略估算一下，或者直接给 1.1
    capacity_sim = 1.1 
    
    T_env_sim = 24.0 # 环境温度
    T_init_sim = 24.0
    SOC_init_sim = 1.0
    
    # 2. 运行仿真
    sim_result = simulate_power_depletion(
        t_sim, P_sim, 
        SOC_init_sim, T_init_sim, T_env_sim, capacity_sim,
        R_base_fit, k_fit, mCp_fit, c_fit,
        V_cutoff=2.7
    )
    
    # 3. 绘图验证
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Power & Current
    plt.subplot(3, 1, 1)
    plt.plot(sim_result['t'], sim_result['P'], 'g--', label='Input Power (W)')
    plt.plot(sim_result['t'], sim_result['I'], 'b-', label='Calculated Current (A)')
    plt.ylabel('Magnitude')
    plt.title('Simulation: Power Input -> Current Response')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Plot 2: Voltage & SOC
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(sim_result['t'], sim_result['V'], 'r-', label='Voltage (V)')
    ax2.plot(sim_result['t'], sim_result['SOC'], 'k:', label='SOC')
    ax1.set_ylabel('Voltage (V)', color='r')
    ax2.set_ylabel('SOC', color='k')
    ax1.set_title('Simulation: Voltage and SOC Drop')
    ax1.grid(True)
    
    # Plot 3: Temperature
    plt.subplot(3, 1, 3)
    plt.plot(sim_result['t'], sim_result['T'], 'orange', label='Temperature (C)')
    plt.axhline(y=T_env_sim, color='gray', linestyle='--', label='Ambient T')
    plt.ylabel('Temp (°C)')
    plt.xlabel('Time (s)')
    plt.title('Simulation: Temperature Rise')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# %%
