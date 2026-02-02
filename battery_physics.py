import numpy as np
from scipy.integrate import cumulative_trapezoid

# === 常数定义 ===
FARADAY = 96485.3       
R_GAS = 8.3145          
E_ACTIVATION = 31930.0  
T_REF = 298.15          

def cul_SOC(I, t, capacity_Ah):
    """ 安时积分法 """
    discharged_charge_As = cumulative_trapezoid(I, t, initial=0)
    discharged_charge_Ah = discharged_charge_As / 3600.0
    soc = 1.0 - (discharged_charge_Ah / capacity_Ah)
    return np.clip(soc, 0.01, 1)

def get_OCV(soc):
    """ 
    OCV 模型 (用户指定更新) 
    3.4 + (4.0 - 3.4) * soc = 3.4 + 0.6 * soc
    """
    # 这里稍微更正一下参数。
    return 3.4 + 0.6 * soc

def get_internal_resistance(T_celsius, SOC, R_base, k_low_soc):
    """
    计算特定时刻的内阻 R_int(T, SOC)
    供电压模型和热模型共用
    """
    T_kelvin = T_celsius + 273.15
    
    # 1. Arrhenius 温度修正项
    temp_correction = np.exp((E_ACTIVATION / R_GAS) * (1.0/T_kelvin - 1.0/T_REF))
    
    # 2. SOC 修正项 (R = R_base + k * (1/SOC - 1))
    R_soc_part = R_base + k_low_soc * (1.0/SOC - 1.0)
    
    # 3. 总内阻
    return R_soc_part * temp_correction

def total_voltage_model(inputs, R_base, k_low_soc):
    """
    电压模型 (用于第一步拟合)
    """
    I, OCV_val, T_celsius, SOC_val = inputs
    
    # 调用提取出来的电阻计算函数
    R_total = get_internal_resistance(T_celsius, SOC_val, R_base, k_low_soc)
    
    return OCV_val - I * R_total

def thermal_model_simulation(inputs, mCp, c, R_base, k_low_soc):
    """
    热模型仿真求解器 (用于第二步拟合)
    
    方程: mCp * dT/dt = I^2 * R(T) - c * (T - T_env)
    离散化: T[i+1] = T[i] + dt * (I^2*R - c(T-T_env)) / mCp
    
    参数:
        inputs: (time_arr, I_arr, T_env_arr, SOC_arr, T_init)
        mCp: 热容 (J/K)
        c:   散热系数 (W/K)
        R_base, k_low_soc: 已知的电阻参数
    """
    t_arr, I_arr, T_env_arr, SOC_arr, T_init = inputs
    
    N = len(t_arr)
    T_pred = np.zeros(N)
    T_pred[0] = T_init  # 初始温度
    
    # 欧拉法积分 (Euler Integration)
    for i in range(N - 1):
        dt = t_arr[i+1] - t_arr[i]
        
        # 当前状态
        Curr_T = T_pred[i]
        Curr_I = I_arr[i]
        Curr_SOC = SOC_arr[i]
        Curr_T_env = T_env_arr[i]
        
        # 1. 计算当前电阻 (电阻随当前温度变化)
        R_curr = get_internal_resistance(Curr_T, Curr_SOC, R_base, k_low_soc)
        
        # 2. 计算产热功率 P_heat
        P_heat = (Curr_I ** 2) * R_curr
        
        # 3. 计算散热功率 P_cool
        P_cool = c * (Curr_T - Curr_T_env)
        
        # 4. 更新温度
        dT = (P_heat - P_cool) / mCp * dt
        T_pred[i+1] = Curr_T + dT
        
    return T_pred

# === battery_physics.py 追加内容 ===

# === 功率模型系数 (来自 PDF 第 6 节) ===
ETA_A = 0.0453
ETA_B = 0.0558
ETA_C = 0.0018

def get_power_efficiency(P):
    """
    计算功率转换效率 eta(P)
    公式: eta(P) = P / (aP^2 + (1+b)P + c)
    注意: 当 P=0 时，效率无意义(或为0)，需要避免除以0
    """
    if P < 1e-6:
        return 0.0 # 待机或极小功率
    
    denom = ETA_A * (P**2) + (1 + ETA_B) * P + ETA_C
    return P / denom

def solve_current_from_power(P, OCV, R_total):
    """
    根据功率求解电流 I
    方程: I = P / (eta * (OCV - I * R))
    推导: eta * R * I^2 - eta * OCV * I + P = 0
    
    返回:
        I (float): 计算出的电流。如果无解（功率过大电压崩溃），返回 None
    """
    if P <= 1e-6:
        return 0.0
    
    eta = get_power_efficiency(P)
    
    # 二次方程系数 A*I^2 + B*I + C = 0
    A = eta * R_total
    B = -eta * OCV
    C = P
    
    # 判别式 delta = B^2 - 4AC
    delta = B**2 - 4 * A * C
    
    if delta < 0:
        return None # 无法支持该功率输出
    
    # 求解: I = (-B ± sqrt(delta)) / 2A
    # 通常取较小的根（对应高电压、低电流的稳定工作点）
    # 较大的根对应电压崩溃区域
    I = (-B - np.sqrt(delta)) / (2 * A)
    
    return I

def simulate_power_depletion(
    time_arr, P_arr, 
    initial_soc, initial_temp, T_env, capacity_Ah,
    R_base, k_low_soc, mCp, c,
    V_cutoff=3.0
):
    """
    全过程仿真：给定功率曲线 P(t)，模拟直到耗尽
    
    参数:
        time_arr: 时间数组 (s)
        P_arr: 对应的功率数组 (W)
        initial_soc: 初始 SOC (0~1)
        initial_temp: 初始电池温度 (C)
        T_env: 环境温度 (C)
        capacity_Ah: 电池容量
        R_base, k_low_soc: 内阻模型参数
        mCp, c: 热模型参数
        V_cutoff: 截止电压
        
    返回:
        results (dict): 包含 t, I, V, SOC, T 的数组
    """
    N = len(time_arr)
    
    # 初始化结果数组
    res_I = np.zeros(N)
    res_V = np.zeros(N)
    res_SOC = np.zeros(N)
    res_T = np.zeros(N)
    
    # 设置初始状态
    curr_SOC = initial_soc
    curr_T = initial_temp
    
    # 记录是否提前终止
    end_idx = N
    
    for i in range(N):
        t_curr = time_arr[i]
        P_curr = P_arr[i]
        
        # 1. 计算当前时刻的 OCV 和 内阻
        curr_OCV = get_OCV(curr_SOC)
        curr_R = get_internal_resistance(curr_T, curr_SOC, R_base, k_low_soc)
        
        # 2. 求解电流 I
        I_val = solve_current_from_power(P_curr, curr_OCV, curr_R)
        
        if I_val is None:
            print(f"Simulation stopped at t={t_curr:.1f}s: Power too high for battery state.")
            end_idx = i
            break
            
        # 3. 计算端电压 V
        V_val = curr_OCV - I_val * curr_R
        
        # 4. 记录数据
        res_I[i] = I_val
        res_V[i] = V_val
        res_SOC[i] = curr_SOC
        res_T[i] = curr_T
        
        # 5. 检查截止条件
        if V_val <= V_cutoff or curr_SOC <= 0.05: # 留一点余量防止除零
            # print(f"Simulation stopped at t={t_curr:.1f}s: Cutoff reached (V={V_val:.2f}, SOC={curr_SOC:.2f})")
            end_idx = i + 1
            break
            
        # 6. 状态更新 (为下一步做准备)
        if i < N - 1:
            dt = time_arr[i+1] - time_arr[i]
            
            # 更新 SOC (安时积分)
            # capacity_Ah * 3600 = Total Coulombs
            # dSOC = - I * dt / Total_Coulombs
            dSOC = - I_val * dt / (capacity_Ah * 3600.0)
            curr_SOC = curr_SOC + dSOC
            
            # 更新 温度 (欧拉法)
            # dT = (P_heat - P_cool) / mCp * dt
            P_heat = (I_val ** 2) * curr_R
            P_cool = c * (curr_T - T_env)
            dT = (P_heat - P_cool) / mCp * dt
            curr_T = curr_T + dT

    # 截断数组返回有效部分
    return {
        't': time_arr[:end_idx],
        'P': P_arr[:end_idx],
        'I': res_I[:end_idx],
        'V': res_V[:end_idx],
        'SOC': res_SOC[:end_idx],
        'T': res_T[:end_idx]
    }