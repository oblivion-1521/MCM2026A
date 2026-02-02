import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def read_and_compute_power(csv_file='../battery_dataset_sample.csv', device_id='DEV_0013'):
    """
    读取 CSV 文件，只保留指定 device_id 的数据，
    计算 PDF 中描述的各部分功耗，并输出总估算功耗。
    
    参数:
        csv_file: CSV 文件路径
        device_id: 设备ID，默认为 'DEV_0013'
    """
    # 1. 读取数据
    df = pd.read_csv(csv_file)
    
    # 2. 只保留指定设备
    df = df[df['device_id'] == device_id].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for device_id: {device_id}")
    
    # 3. 转换时间戳为 datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按时间排序（虽然数据看起来已排序，但以防万一）
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 计算相邻记录的时间差（分钟），用于后续可能分析耗电率
    df['time_diff_min'] = df['timestamp'].diff().dt.total_seconds() / 60.0
    df['time_diff_min'] = df['time_diff_min'].fillna(0)
    
    # 参数（来自 PDF）
    A_ref       = 100
    A           = 110
    beta_br     = 612      # mW
    
    P_CPUstatic = 121.46   # mW
    beta_f1     = 4.34     # mW / %
    beta_f2     = 3.42     # mW / %
    a           = 15
    u0          = 0.4
    
    beta_Gon    = 429.55   # mW
    
    P_static    = 100      # mW
    
    # sigmoid 函数 w(u)，u 是 0~1 的 cpu_load
    def w(u):
        return 1 / (1 + np.exp(-a * (u - u0)))
    
    # 屏幕功耗
    def p_screen(row):
        b = row['brightness']
        return beta_br * (A / A_ref) * b
    
    # CPU 功耗（注意：beta_f1 / beta_f2 是按百分比给的，所以 cpu_load 需要 ×100）
    def p_cpu(row):
        u = row['cpu_load']               # 0~1
        u_percent = u * 100               # 转为 0~100%
        
        wu = w(u)                         # sigmoid 用原始 0~1 的 u
        
        p_low  = P_CPUstatic + beta_f1 * u_percent
        p_high = P_CPUstatic + beta_f2 * u_percent
        
        return (1 - wu) * p_low + wu * p_high
    
    # GPS 功耗
    def p_gps(row):
        m_gon = 1 if row['gps_enabled'] else 0
        # 第二项 β_Gsl × M_Gsl × 3 = 0（按要求取0）
        return beta_Gon * m_gon
    
    # 网络功耗（WiFi / 4G / 5G）
    def p_network(row):
        if not row['network_active']:
            return 0.0
        nt = str(row['network_type']).lower()
        if nt == 'wifi':
            return 3.0
        elif nt in ['4g', '5g']:
            return 10.0
        else:
            return 0.0
    
    # 计算各部分功率
    df['P_screen_mW']    = df.apply(p_screen, axis=1)
    df['P_CPU_mW']       = df.apply(p_cpu, axis=1)
    df['P_GPS_mW']       = df.apply(p_gps, axis=1)
    df['P_network_mW']   = df.apply(p_network, axis=1)
    df['P_static_mW']    = P_static
    
    # 总估算功率（不含其他未建模的部分）
    df['P_total_model_mW'] = (
        df['P_screen_mW'] +
        df['P_CPU_mW'] +
        df['P_GPS_mW'] +
        df['P_network_mW'] +
        df['P_static_mW']
    )
    
    # 选择要展示的列，方便对比
    cols_to_show = [
        'timestamp', 'power_consumption_mw', 'P_total_model_mW',
        'P_screen_mW', 'P_CPU_mW', 'P_GPS_mW', 'P_network_mW', 'P_static_mW',
        'cpu_load', 'brightness', 'network_type', 'network_active', 'gps_enabled'
    ]
    
    return df[cols_to_show]


# ===================== 示例运行与输出 =====================
if __name__ == "__main__":
    df_result = read_and_compute_power()
    
    # print("前 5 行结果（DEV_0013）：")
    # print(df_result.head(10).to_string(index=False))
    
    # print("\n第 0 行详细输出：")
    # print(df_result.iloc[0])
    
    # print("\n第 1 行详细输出：")
    # print(df_result.iloc[1])

    # 计算第一到10行power_consumption_mw与P_total_model_mW的差值之和除以10
    discra = df_result['power_consumption_mw'][:10] - df_result['P_total_model_mW'][:10]
    ans = discra.sum()/10
    print(ans)
    # 输出前82列的时间戳，不要换行，用空格隔开
    print("\n前 82 行的时间戳：")
    for i in range(85):
        print(df_result.iloc[i]['timestamp'], end=' ')