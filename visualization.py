import numpy as np
import matplotlib.pyplot as plt


def plot_power_depletion_simulation(sim_result, t_source_sec, P_source_watts, 
                                     capacity_sim, T_env_sim, V_cutoff_sim,
                                     phone_model='Unknown'):
    """
    绘制功率耗尽仿真结果的可视化图
    
    参数:
        sim_result: simulate_power_depletion 返回的结果字典
        t_source_sec: 原始日志时间点 (秒)
        P_source_watts: 原始日志功率点 (W)
        capacity_sim: 电池容量 (Ah)
        T_env_sim: 环境温度 (°C)
        V_cutoff_sim: 截止电压 (V)
        phone_model: 手机型号名称
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
    plt.title(f'[{phone_model}] Simulation Part 1: Real-world Power Profile & Battery Current Response')
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
    ax_v.set_title(f'[{phone_model}] Simulation Part 2: Voltage and SOC Depletion (Capacity: {capacity_sim} Ah)')
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
    plt.title(f'[{phone_model}] Simulation Part 3: Battery Thermal Profile during Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1000, t_end + 1000)
    
    plt.tight_layout()
    plt.show()

    # 打印仿真统计信息
    duration_hrs = t_end / 3600.0
    print(f"\n[Simulation Summary - {phone_model}]")
    print(f"Total simulated time: {t_end:.1f} s ({duration_hrs:.2f} hours)")
    print(f"Final SOC: {sim_result['SOC'][-1]*100:.2f} %")


def plot_aging_comparison(sim_result_normal, sim_result_aging, lambda_aging, 
                          D_aging=0.0, V_cutoff=3.0, phone_model='Unknown'):
    """
    对比正常电池和老化电池的耗尽曲线
    
    参数:
        sim_result_normal: 正常电池仿真结果
        sim_result_aging: 老化电池仿真结果
        lambda_aging: 内阻增加系数
        D_aging: 容量衰减系数
        V_cutoff: 截止电压 (V)
        phone_model: 手机型号名称
    """
    plt.figure(figsize=(14, 10))
    
    # 子图1: SOC对比
    plt.subplot(3, 1, 1)
    plt.plot(sim_result_normal['t'], sim_result_normal['SOC']*100, 'b-', 
             label='Normal Battery', linewidth=2)
    plt.plot(sim_result_aging['t'], sim_result_aging['SOC']*100, 'r--', 
             label=f'Aged Battery (λ={lambda_aging}, D={D_aging})', linewidth=2)
    plt.ylabel('SOC (%)')
    plt.title(f'[{phone_model}] Battery Aging Comparison: SOC Depletion')
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
    plt.title(f'[{phone_model}] Battery Aging Comparison: Voltage')
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
    plt.title(f'[{phone_model}] Battery Aging Comparison: Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印对比统计
    t_end_normal = sim_result_normal['t'][-1]
    t_end_aging = sim_result_aging['t'][-1]
    capacity_eff = sim_result_aging.get('capacity_eff', 'N/A')
    
    print(f"\n[Aging Comparison Summary - {phone_model}]")
    print(f"Normal battery runtime:   {t_end_normal/3600:.2f} hours")
    print(f"Aged battery runtime:     {t_end_aging/3600:.2f} hours")
    print(f"Runtime reduction:        {(1 - t_end_aging/t_end_normal)*100:.1f}%")
    if isinstance(capacity_eff, float):
        print(f"Effective capacity (aged): {capacity_eff:.3f} Ah")


def plot_time_uncertainty(uncertainty_results, capacity_sim, phone_model='Unknown'):
    """
    绘制时间不确定性分析结果 - 不同恒定功率下的 SOC 随时间变化
    
    参数:
        uncertainty_results: analyze_time_uncertainty 返回的结果列表
        capacity_sim: 电池容量 (Ah)
        phone_model: 手机型号名称
    """
    plt.figure(figsize=(14, 10))
    
    # 使用 colormap 生成颜色
    n_levels = len(uncertainty_results)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_levels))
    
    # 子图1: SOC 随时间变化
    plt.subplot(2, 1, 1)
    for i, result in enumerate(uncertainty_results):
        P_mW = result['P_const'] * 1000  # 转换为 mW
        runtime_hrs = result['t'][-1] / 3600.0
        plt.plot(result['t'] / 3600, result['SOC'] * 100, 
                 color=colors[i], linewidth=2,
                 label=f'P={P_mW:.0f}mW (Runtime: {runtime_hrs:.2f}h)')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('SOC (%)')
    plt.title(f'[{phone_model}] Time Uncertainty Analysis: SOC vs Time at Different Power Levels')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # 子图2: 电压随时间变化
    plt.subplot(2, 1, 2)
    for i, result in enumerate(uncertainty_results):
        P_mW = result['P_const'] * 1000
        plt.plot(result['t'] / 3600, result['V'], 
                 color=colors[i], linewidth=2,
                 label=f'P={P_mW:.0f}mW')
    
    plt.axhline(y=3.0, color='gray', linestyle=':', label='Cutoff Voltage (3.0V)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Terminal Voltage (V)')
    plt.title(f'[{phone_model}] Time Uncertainty Analysis: Voltage vs Time at Different Power Levels')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    runtimes = [r['t'][-1] / 3600.0 for r in uncertainty_results]
    powers = [r['P_const'] * 1000 for r in uncertainty_results]
    
    print(f"\n[Time Uncertainty Summary - {phone_model}]")
    print(f"Capacity: {capacity_sim} Ah")
    print(f"Power range: {min(powers):.0f} mW ~ {max(powers):.0f} mW")
    print(f"Runtime range: {max(runtimes):.2f} h (at P_min) ~ {min(runtimes):.2f} h (at P_max)")
    print(f"Runtime uncertainty: ±{(max(runtimes) - min(runtimes))/2:.2f} hours around mean")


def plot_monte_carlo_results(runtimes, df_inputs, N_simulations):
    """
    绘制蒙特卡洛敏感性分析结果
    
    参数:
        runtimes: 续航时间数组 (hours)
        df_inputs: 输入参数 DataFrame
        N_simulations: 仿真次数
    """
    import pandas as pd
    
    runtimes = np.array(runtimes)
    
    # 增加图片尺寸，给右侧图更多空间
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 1. 续航时间分布直方图
    ax1 = axes[0]
    
    # 根据数据范围自动计算 bin 数量
    data_range = np.max(runtimes) - np.min(runtimes)
    n_bins = max(50, int(data_range / 0.2))
    
    ax1.hist(runtimes, bins=n_bins, color='skyblue', edgecolor='black', alpha=0.7, linewidth=0.5)
    ax1.axvline(np.mean(runtimes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(runtimes):.2f}h')
    ax1.axvline(np.percentile(runtimes, 2.5), color='orange', linestyle=':', linewidth=1.5, label=f'2.5%: {np.percentile(runtimes, 2.5):.2f}h')
    ax1.axvline(np.percentile(runtimes, 97.5), color='orange', linestyle=':', linewidth=1.5, label=f'97.5%: {np.percentile(runtimes, 97.5):.2f}h')
    
    ax1.set_xlabel('Runtime (Hours)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Monte Carlo Analysis: Battery Runtime Distribution\n(N = {N_simulations:,} simulations)', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息文本框 - 放在右上角
    stats_text = f'Mean: {np.mean(runtimes):.3f}h\nStd: {np.std(runtimes):.3f}h\n95% CI: [{np.percentile(runtimes, 2.5):.3f}, {np.percentile(runtimes, 97.5):.3f}]h'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 龙卷风图 (Tornado Plot) - 相关性系数
    ax2 = axes[1]
    correlations = df_inputs.corrwith(pd.Series(runtimes))
    correlations_sorted = correlations.sort_values()
    
    # 更美观的颜色
    colors = ['#E57373' if x < 0 else '#64B5F6' for x in correlations_sorted]
    bars = ax2.barh(correlations_sorted.index, correlations_sorted.values, 
                    color=colors, alpha=0.85, edgecolor='black', height=0.6)
    
    ax2.set_title(f'Sensitivity Analysis: Correlation with Runtime\n(N = {N_simulations:,} simulations)', fontsize=13)
    ax2.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 设置 x 轴范围
    x_min = min(correlations_sorted.values.min() * 1.1, -0.15)
    x_max = max(correlations_sorted.values.max() * 1.3, 0.25)
    ax2.set_xlim(x_min, x_max)
    
    # 在条形图上添加数值标签 - 放在条形内部或紧贴末端
    for bar, val in zip(bars, correlations_sorted.values):
        # 对于较长的条（|val| > 0.3），标签放在条形内部；否则放在外部
        if abs(val) > 0.3:
            # 标签放在条形内部
            if val >= 0:
                x_pos = val - 0.02
                ha = 'right'
            else:
                x_pos = val + 0.02
                ha = 'left'
            text_color = 'white'
        else:
            # 标签放在条形外部
            if val >= 0:
                x_pos = val + 0.02
                ha = 'left'
            else:
                x_pos = val - 0.02
                ha = 'right'
            text_color = 'black'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold', color=text_color)
    
    # 美化 Y 轴标签
    ax2.set_yticklabels(correlations_sorted.index, fontsize=11)
    
    plt.tight_layout(pad=2.0)
    plt.show()
    
    # 打印统计
    print(f"\n=== Monte Carlo Results (N={N_simulations:,}) ===")
    print(f"Mean Runtime: {np.mean(runtimes):.3f} hours")
    print(f"Std Dev:      {np.std(runtimes):.3f} hours")
    print(f"95% CI:       [{np.percentile(runtimes, 2.5):.3f}, {np.percentile(runtimes, 97.5):.3f}] hours")
    print(f"Min:          {np.min(runtimes):.3f} hours")
    print(f"Max:          {np.max(runtimes):.3f} hours")
