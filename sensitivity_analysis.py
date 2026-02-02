#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import multiprocessing as mp
import os

# 引入你现有的模块
from battery_physics import simulate_power_depletion
from Power_options import read_and_compute_power
from visualization import plot_monte_carlo_results


def single_simulation(args):
    """单次蒙特卡洛模拟，用于多进程并行"""
    seed, t_sim, P_base_curve, params_base = args
    
    # 设置随机种子确保每个进程独立
    np.random.seed(seed)
    
    # --- A. 随机抽样 (改变参数值) ---
    sim_capacity = np.random.normal(params_base['capacity'], params_base['capacity'] * 0.02)
    sim_R_base = np.random.normal(params_base['R_base'], params_base['R_base'] * 0.1)
    sim_c = params_base['c'] * np.random.uniform(0.8, 1.1)
    sim_T_env = np.random.normal(params_base['T_env'], 3.0)
    
    # --- B. 随机抽样 (使用模式波动) ---
    sim_P_scale = np.random.normal(1.0, 0.15)
    noise = np.random.normal(0, 0.05, size=len(P_base_curve))
    
    # 合成新的功率曲线
    P_sim_input = (P_base_curve * sim_P_scale) + noise
    P_sim_input = np.maximum(P_sim_input, 0.001)
    
    # --- C. 运行仿真 ---
    res = simulate_power_depletion(
        t_sim, P_sim_input,
        initial_soc=1.0,
        initial_temp=sim_T_env,
        T_env=sim_T_env,
        capacity_Ah=sim_capacity,
        R_base=sim_R_base,
        k_low_soc=params_base['k_low_soc'],
        mCp=params_base['mCp'],
        c=sim_c
    )
    
    # --- D. 提取指标 ---
    runtime_hours = res['t'][-1] / 3600.0
    max_T = np.max(res['T'])
    
    return {
        'runtime': runtime_hours,
        'max_temp': max_T,
        'inputs': {
            'Capacity': sim_capacity,
            'R_base': sim_R_base,
            'c': sim_c,
            'T_env': sim_T_env,
            'P': sim_P_scale
        }
    }


def run_sensitivity_analysis(N_simulations=500):
    # ================= 1. 准备基准数据 =================
    # 读取基准功率曲线
    df_power = read_and_compute_power('../battery_dataset_sample.csv', device_id='DEV_0030')
    t_start = df_power['timestamp'].iloc[0]
    t_source_sec = (df_power['timestamp'] - t_start).dt.total_seconds().values
    P_source_watts = df_power['P_total_model_mW'].values / 1000.0
    
    # 建立插值函数，方便重采样
    t_max = t_source_sec[-1]
    t_sim = np.arange(0, t_max, 1.0) # 1秒步长
    interp_func = interp1d(t_source_sec, P_source_watts, kind='zero', 
                           bounds_error=False, fill_value=(P_source_watts[0], P_source_watts[-1]))
    P_base_curve = interp_func(t_sim)
    
    # 基准参数 (来自 main.py 的拟合结果，这里手动填入示例值，实际应用你之前拟合的值)
    params_base = {
        'capacity': 3.951,
        'R_base': 0.28626,    # 假设值，请替换为你 fit 出的值
        'k_low_soc': 0.003092,
        'mCp': 94.0829,
        'c': 0.0492,
        'T_env': 24.0,
        'P': 1.0    # 功率缩放因子基准
    }

    # 获取CPU线程数
    num_workers = os.cpu_count()
    print(f"Detected {num_workers} CPU threads")
    print(f"Running Monte Carlo Simulation ({N_simulations} runs) with {num_workers} processes...")

    # ================= 2. 蒙特卡洛并行计算 =================
    # 准备参数列表，每个任务带不同的随机种子
    args_list = [(i, t_sim, P_base_curve, params_base) for i in range(N_simulations)]
    
    # 计算最优 chunksize 减少进程间通信开销
    chunksize = max(1, N_simulations // (num_workers * 4))
    
    # 使用进程池并行执行 (imap_unordered 更快，因为不需要保持顺序)
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(single_simulation, args_list, chunksize=chunksize),
            total=N_simulations,
            desc="Monte Carlo Simulation",
            unit="sim"
        ))
    
    # 提取结果
    results_runtime = [r['runtime'] for r in results]
    results_max_temp = [r['max_temp'] for r in results]
    input_records = [r['inputs'] for r in results]

    # ================= 3. 结果分析与绘图 =================
    df_inputs = pd.DataFrame(input_records)
    runtimes = np.array(results_runtime)
    
    # 调用 visualization 模块绘图
    plot_monte_carlo_results(runtimes, df_inputs, N_simulations)

if __name__ == "__main__":
    run_sensitivity_analysis(10000)
# %%
