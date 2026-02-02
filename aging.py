import numpy as np
import matplotlib.pyplot as plt
from battery_physics import (
    get_OCV, get_internal_resistance, solve_current_from_power
)


def simulate_power_depletion_with_aging(
    time_arr, P_arr,
    initial_soc, initial_temp, T_env, capacity_Ah,
    R_base, k_low_soc, mCp, c,
    lambda_aging=0.2,
    V_cutoff=3.0
):
    """
    考虑老化的耗尽仿真: R_int_new = R_int * (1 + lambda_aging)
    
    参数:
        lambda_aging: 老化系数，默认0.2表示内阻增加20%
        其他参数同 simulate_power_depletion
    """
    N = len(time_arr)
    
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
        
        if V_val <= V_cutoff or curr_SOC <= 0.005:
            end_idx = i + 1
            break
            
        if i < N - 1:
            dt = time_arr[i+1] - time_arr[i]
            dSOC = - I_val * dt / (capacity_Ah * 3600.0)
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
        'T': res_T[:end_idx]
    }


def plot_aging_comparison(sim_result_normal, sim_result_aging, lambda_aging, V_cutoff=3.0):
    """
    对比正常电池和老化电池的耗尽曲线
    """
    plt.figure(figsize=(14, 10))
    
    # 子图1: SOC对比
    plt.subplot(3, 1, 1)
    plt.plot(sim_result_normal['t'], sim_result_normal['SOC']*100, 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['SOC']*100, 'r--', 
             label=f'Aged Battery (λ={lambda_aging})', linewidth=2)
    plt.ylabel('SOC (%)')
    plt.title('Battery Aging Comparison: SOC Depletion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 电压对比
    plt.subplot(3, 1, 2)
    plt.plot(sim_result_normal['t'], sim_result_normal['V'], 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['V'], 'r--', 
             label=f'Aged Battery (λ={lambda_aging})', linewidth=2)
    plt.axhline(y=V_cutoff, color='gray', linestyle=':', label='Cutoff Voltage')
    plt.ylabel('Terminal Voltage (V)')
    plt.title('Battery Aging Comparison: Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 温度对比
    plt.subplot(3, 1, 3)
    plt.plot(sim_result_normal['t'], sim_result_normal['T'], 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['T'], 'r--', 
             label=f'Aged Battery (λ={lambda_aging})', linewidth=2)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.title('Battery Aging Comparison: Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印对比统计
    t_end_normal = sim_result_normal['t'][-1]
    t_end_aging = sim_result_aging['t'][-1]
    print(f"\n[Aging Comparison Summary]")
    print(f"Normal battery runtime: {t_end_normal/3600:.2f} hours")
    print(f"Aged battery runtime:   {t_end_aging/3600:.2f} hours")
    print(f"Runtime reduction:      {(1 - t_end_aging/t_end_normal)*100:.1f}%")