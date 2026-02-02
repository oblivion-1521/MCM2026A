import numpy as np
from battery_physics import (
    get_OCV, get_internal_resistance, solve_current_from_power
)


def simulate_constant_power_depletion(
    P_const,
    initial_soc, initial_temp, T_env, capacity_Ah,
    R_base, k_low_soc, mCp, c,
    V_cutoff=3.0,
    dt=1.0,
    max_time=24*3600  # 默认最大仿真24小时
):
    """
    在恒定功率下仿真电池耗尽过程
    
    参数:
        P_const: 恒定功率 (W)
        initial_soc: 初始SOC
        initial_temp: 初始温度 (°C)
        T_env: 环境温度 (°C)
        capacity_Ah: 电池容量 (Ah)
        R_base, k_low_soc: 内阻参数
        mCp, c: 热参数
        V_cutoff: 截止电压 (V)
        dt: 时间步长 (s)
        max_time: 最大仿真时间 (s)
    
    返回:
        dict: 包含 t, SOC, V, T, I 的仿真结果
    """
    N_max = int(max_time / dt)
    
    res_t = []
    res_SOC = []
    res_V = []
    res_T = []
    res_I = []
    
    curr_SOC = initial_soc
    curr_T = initial_temp
    t_curr = 0.0
    
    for i in range(N_max):
        curr_OCV = get_OCV(curr_SOC)
        curr_R = get_internal_resistance(curr_T, curr_SOC, R_base, k_low_soc)
        
        I_val = solve_current_from_power(P_const, curr_OCV, curr_R)
        
        if I_val is None:
            break
            
        V_val = curr_OCV - I_val * curr_R
        
        res_t.append(t_curr)
        res_SOC.append(curr_SOC)
        res_V.append(V_val)
        res_T.append(curr_T)
        res_I.append(I_val)
        
        if V_val <= V_cutoff or curr_SOC <= 0.05:
            break
        
        # 更新状态
        dSOC = - I_val * dt / (capacity_Ah * 3600.0)
        curr_SOC = curr_SOC + dSOC
        
        P_heat = (I_val ** 2) * curr_R
        P_cool = c * (curr_T - T_env)
        dT = (P_heat - P_cool) / mCp * dt
        curr_T = curr_T + dT
        
        t_curr += dt
    
    return {
        't': np.array(res_t),
        'SOC': np.array(res_SOC),
        'V': np.array(res_V),
        'T': np.array(res_T),
        'I': np.array(res_I),
        'P_const': P_const
    }


def analyze_time_uncertainty(
    P_min, P_max,
    initial_soc, initial_temp, T_env, capacity_Ah,
    R_base, k_low_soc, mCp, c,
    V_cutoff=3.0,
    dt=1.0,
    n_levels=5
):
    """
    分析在不同恒定功率下的电池续航时间不确定性
    
    参数:
        P_min: 最小功率 (W)
        P_max: 最大功率 (W)
        n_levels: 功率等级数量（包含P_min和P_max）
        其他参数同 simulate_constant_power_depletion
    
    返回:
        list: 每个功率等级的仿真结果列表
    """
    P_levels = np.linspace(P_min, P_max, n_levels)
    results = []
    
    for P in P_levels:
        result = simulate_constant_power_depletion(
            P,
            initial_soc, initial_temp, T_env, capacity_Ah,
            R_base, k_low_soc, mCp, c,
            V_cutoff=V_cutoff,
            dt=dt
        )
        results.append(result)
        
        # 打印每个功率等级的续航时间
        runtime_hrs = result['t'][-1] / 3600.0
        print(f"  P = {P*1000:.1f} mW -> Runtime: {runtime_hrs:.2f} hours")
    
    return results


def get_runtime_bounds(results):
    """
    从不确定性分析结果中提取续航时间边界
    
    返回:
        tuple: (t_min, t_max, t_avg) 单位为小时
    """
    runtimes = [r['t'][-1] / 3600.0 for r in results]
    return min(runtimes), max(runtimes), np.mean(runtimes)
