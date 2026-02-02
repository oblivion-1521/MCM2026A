import numpy as np
from battery_physics import (
    get_OCV, get_internal_resistance, solve_current_from_power
)


def simulate_power_depletion_with_aging(
    time_arr, P_arr,
    initial_soc, initial_temp, T_env, capacity_Ah,
    R_base, k_low_soc, mCp, c,
    lambda_aging=0.2,
    D_aging=0.0,
    V_cutoff=3.0
):
    """
    考虑老化的耗尽仿真: 
        - 内阻增加: R_int_new = R_int * (1 + lambda_aging)
        - 容量衰减: C_eff = C_ini * (1 - D_aging)
    
    参数:
        lambda_aging: 老化系数(内阻)，默认0.2表示内阻增加20%
        D_aging: 容量衰减系数，默认0.0，取值范围[0,1]，例如0.2表示容量变为原来的80%
        其他参数同 simulate_power_depletion
    """
    N = len(time_arr)
    
    # 计算有效容量
    capacity_eff = capacity_Ah * (1 - D_aging)
    
    res_I = np.zeros(N)
    res_V = np.zeros(N)
    res_SOC = np.zeros(N)
    res_T = np.zeros(N)
    
    curr_SOC = initial_soc
    curr_T = initial_temp
    end_idx = N
    
    for i in range(N):
        t_curr = time_arr[i]
        P_curr = P_arr[i]
        
        curr_OCV = get_OCV(curr_SOC)
        # 老化后的内阻: R_int_new = R_int * (1 + lambda)
        curr_R = get_internal_resistance(curr_T, curr_SOC, R_base, k_low_soc) * (1 + lambda_aging)
        
        I_val = solve_current_from_power(P_curr, curr_OCV, curr_R)
        
        if I_val is None:
            print(f"[Aging] Simulation stopped at t={t_curr:.1f}s: Power too high.")
            end_idx = i
            break
            
        V_val = curr_OCV - I_val * curr_R
        
        res_I[i] = I_val
        res_V[i] = V_val
        res_SOC[i] = curr_SOC
        res_T[i] = curr_T
        
        if V_val <= V_cutoff or curr_SOC <= 0.05:
            end_idx = i + 1
            break
            
        if i < N - 1:
            dt = time_arr[i+1] - time_arr[i]
            # 使用衰减后的有效容量计算SOC变化
            dSOC = - I_val * dt / (capacity_eff * 3600.0)
            curr_SOC = curr_SOC + dSOC
            
            P_heat = (I_val ** 2) * curr_R
            P_cool = c * (curr_T - T_env)
            dT = (P_heat - P_cool) / mCp * dt
            curr_T = curr_T + dT

    return {
        't': time_arr[:end_idx],
        'P': P_arr[:end_idx],
        'I': res_I[:end_idx],
        'V': res_V[:end_idx],
        'SOC': res_SOC[:end_idx],
        'T': res_T[:end_idx],
        'capacity_eff': capacity_eff  # 返回有效容量供参考
    }