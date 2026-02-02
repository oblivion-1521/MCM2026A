import numpy as np
import matplotlib.pyplot as plt


def plot_power_depletion_simulation(sim_result, t_source_sec, P_source_watts, 
                                     capacity_sim, T_env_sim, V_cutoff_sim):
    """
    绘制功率耗尽仿真结果的可视化图
    
    参数:
        sim_result: simulate_power_depletion 返回的结果字典
        t_source_sec: 原始日志时间点 (秒)
        P_source_watts: 原始日志功率点 (W)
        capacity_sim: 电池容量 (Ah)
        T_env_sim: 环境温度 (°C)
        V_cutoff_sim: 截止电压 (V)
    """
    # 获取仿真结束的具体时间
    t_end = sim_result['t'][-1]

    plt.figure(figsize=(14, 12))
    
    # 子图 A: 输入功率与计算出的电流
    plt.subplot(3, 1, 1)
    
    # 过滤原始日志点，只保留仿真运行时间范围内的数据
    mask = t_source_sec <= t_end
    t_log_filtered = t_source_sec[mask]
    P_log_filtered = P_source_watts[mask]

    # 绘制过滤后的原始采样点
    plt.scatter(t_log_filtered, P_log_filtered, color='black', marker='x', 
                label='Original Log Points (W)', alpha=0.6)
    
    # 绘制插值后的连续功率曲线
    plt.plot(sim_result['t'], sim_result['P'], color='green', alpha=0.4, 
             label='Interpolated P_in (W)')
    
    # 绘制计算出的响应电流
    plt.plot(sim_result['t'], sim_result['I'], color='blue', linestyle='--', 
             label='Calculated I_out (A)')
    
    plt.ylabel('Power (W) / Current (A)')
    plt.title('Simulation Part 1: Real-world Power Profile & Battery Current Response')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
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
    plt.plot(sim_result['t'], sim_result['T'], color='orange', linewidth=2, 
             label='Simulated Battery Temp')
    plt.axhline(y=T_env_sim, color='gray', linestyle='--', label='Ambient Temperature')
    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Temperature (°C)')
    plt.title('Simulation Part 3: Battery Thermal Profile during Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1000, t_end + 1000)
    
    plt.tight_layout()
    plt.show()

    # 打印仿真统计信息
    duration_hrs = t_end / 3600.0
    print(f"\n[Simulation Summary]")
    print(f"Total simulated time: {t_end:.1f} s ({duration_hrs:.2f} hours)")
    print(f"Final SOC: {sim_result['SOC'][-1]*100:.2f} %")


def plot_aging_comparison(sim_result_normal, sim_result_aging, lambda_aging, 
                          D_aging=0.0, V_cutoff=3.0):
    """
    对比正常电池和老化电池的耗尽曲线
    
    参数:
        sim_result_normal: 正常电池仿真结果
        sim_result_aging: 老化电池仿真结果
        lambda_aging: 内阻增加系数
        D_aging: 容量衰减系数
        V_cutoff: 截止电压 (V)
    """
    plt.figure(figsize=(14, 10))
    
    # 子图1: SOC对比
    plt.subplot(3, 1, 1)
    plt.plot(sim_result_normal['t'], sim_result_normal['SOC']*100, 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['SOC']*100, 'r--', 
             label=f'Aged Battery (λ={lambda_aging}, D={D_aging})', linewidth=2)
    plt.ylabel('SOC (%)')
    plt.title('Battery Aging Comparison: SOC Depletion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 电压对比
    plt.subplot(3, 1, 2)
    plt.plot(sim_result_normal['t'], sim_result_normal['V'], 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['V'], 'r--', 
             label=f'Aged Battery (λ={lambda_aging}, D={D_aging})', linewidth=2)
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
             label=f'Aged Battery (λ={lambda_aging}, D={D_aging})', linewidth=2)
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
    capacity_eff = sim_result_aging.get('capacity_eff', 'N/A')
    
    print(f"\n[Aging Comparison Summary]")
    print(f"Normal battery runtime:   {t_end_normal/3600:.2f} hours")
    print(f"Aged battery runtime:     {t_end_aging/3600:.2f} hours")
    print(f"Runtime reduction:        {(1 - t_end_aging/t_end_normal)*100:.1f}%")
    if isinstance(capacity_eff, float):
        print(f"Effective capacity (aged): {capacity_eff:.3f} Ah")
