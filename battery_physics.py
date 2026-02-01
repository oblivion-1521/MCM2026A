import numpy as np
from scipy.integrate import cumulative_trapezoid

# 常数定义
FARADAY = 96485.3  # 法拉第常数 (C/mol)
R_GAS = 8.3145    # 气体常数 (J/(mol·K))
E_ACTIVATION = 31930  
T_REF = 298.15  # 参考温度 (K)

def cul_SOC(I, t, capacity_Ah):
    """
    根据安时积分法计算放电过程中的 SOC 序列。
    公式: d(SOC)/dt = -I(t) / Q_capacity
    积分形式: SOC(t) = SOC(0) - (1/Q_capacity) * \int_{0}^{t} I(\tau) d\tau
    
    参数:
        I (np.array): 电流序列 (单位: 安培 A)。
                      在 NASA 数据集中，放电电流通常记录为正值，
                      配合公式中的负号，SOC 会随时间下降。
        t (np.array): 时间序列 (单位: 秒 s)。
        capacity_Ah (float): 该循环的总容量 (单位: 安培小时 Ah)。
        
    返回:
        soc (np.array): 与时间序列等长的 SOC 序列 (0 到 1 之间)。
    """
    
    # 1. 计算电流对时间的积分 (单位: A*s, 即库仑)
    # 使用复合梯形法则进行累积积分
    # initial=0 表示积分从0开始，输出数组长度与输入一致
    discharged_charge_As = cumulative_trapezoid(I, t, initial=0)
    
    # 2. 单位换算: A*s 转换为 A*h (1小时 = 3600秒)
    discharged_charge_Ah = discharged_charge_As / 3600.0
    
    # 3. 计算 SOC
    # 假设放电开始时是满电状态 (SOC = 1.0)
    # 如果电流 I 为正值，则 SOC = 1 - (已放电量 / 总容量)
    soc = 1.0 - (discharged_charge_Ah / capacity_Ah)
    
    # 4. 数据清洗：限制在 [0, 1] 范围内，防止由于积分误差导致的微小越界
    soc = np.clip(soc, 0, 1)
    
    return soc

def get_OCV(soc):
    """
    参数:
        soc (np.array or float): 荷电状态 [0, 1]
        
    返回:
        ocv (np.array or float): 开路电压 (V)
    """
    # V_max = 4.2
    # V_min = 3.0
    
    # ocv = V_min + (V_max - V_min) * soc
    # return ocv
    # 根据图片提供的参数
    return (2.85 + 4.80*soc - 17.8*soc**2 + 38.59*soc**3 + 4.91*soc**4
             - 210.52*soc**5 + 433.98*soc**6 - 368.45*soc**7 + 115.81*soc**8)

def get_thermal_voltage(T_celsius):
    """
    计算 Tafel 公式中的热电压项系数 (RT/F)。
    注意：Tafel 公式中 U_p = (2RT/F) * arcsinh(...)
    这里只处理温度转换和常数部分。
    """
    T_kelvin = T_celsius + 273.15
    # 返回 RT/F 的值
    return (R_GAS * T_kelvin) / FARADAY
def total_voltage_model(inputs, R_base, I0):
    """
    包含 Arrhenius 温度修正的总电压模型。
    
    公式: 
    1. R_int(T) = R_base * exp( (Ea/R) * (1/T - 1/T_ref) )
    2. V(t) = OCV(SOC) - I(t)*R_int(T) - (2RT/F)*arcsinh(I(t)/(2*I0))
    
    参数:
        inputs: tuple (I, OCV, T_celsius)
            - I: 电流 (A)
            - OCV: 开路电压 (V)
            - T_celsius: 电池实际温度 (℃) **注意这里必须是随时间变化的数组**
        R_base: 待拟合参数，参考温度下的内阻 (Ohm)
        I0: 待拟合参数，交换电流 (A)
    """
    I, OCV_val, T_celsius = inputs
    
    # 转换温度为开尔文
    T_kelvin = T_celsius + 273.15
    
    # --- 1. 计算随温度变化的内阻 (Arrhenius Equation) ---
    # 指数项: (Ea / R) * (1/T - 1/T_ref)
    arrhenius_term = (E_ACTIVATION / R_GAS) * (1.0 / T_kelvin - 1.0 / T_REF)
    
    # 当前温度下的内阻
    R_T = R_base * np.exp(arrhenius_term)
    
    # 欧姆压降
    U_ohm = I * R_T
    
    # --- 2. 计算极化电压 (Tafel / Butler-Volmer 简化) ---
    # 系数 2RT/F
    tafel_coeff = (2.0 * R_GAS * T_kelvin) / FARADAY
    
    # 防止 I0 为 0 导致除法错误
    U_p = tafel_coeff * np.arcsinh(I / (2.0 * I0 + 1e-12))
    
    # --- 3. 总电压 ---
    V_pred = OCV_val - U_ohm - U_p
    
    return V_pred