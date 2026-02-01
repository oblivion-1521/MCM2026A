import numpy as np
from scipy.integrate import cumulative_trapezoid

# === 物理常数定义 ===
FARADAY = 96485.3       # 法拉第常数 (C/mol)
R_GAS = 8.3145          # 气体常数 (J/(mol·K))

# === 题目/截图提供的模型参数 (Page 2) ===
# Ea = 31.93 kJ/mol = 31930 J/mol
E_ACTIVATION = 31930.0  
# T_ref = 25 degC = 298.15 K
T_REF = 298.15          

def cul_SOC(I, t, capacity_Ah):
    """
    根据安时积分法计算放电过程中的 SOC 序列。
    公式: SOC(t) = SOC(0) - (1/Q) * ∫ I(τ) dτ
    
    参数:
        I (np.array): 电流序列 (A)。
        t (np.array): 时间序列 (s)。
        capacity_Ah (float): 当前循环的总容量 (Ah)。
    """
    # 计算电流对时间的累积积分 (单位: A*s)
    # initial=0 保证输出长度与输入一致
    discharged_charge_As = cumulative_trapezoid(I, t, initial=0)
    
    # 单位换算: A*s -> A*h
    discharged_charge_Ah = discharged_charge_As / 3600.0
    
    # 假设放电开始时为满电 (1.0)
    soc = 1.0 - (discharged_charge_Ah / capacity_Ah)
    
    # 限制在 [0, 1] 范围内
    return np.clip(soc, 0, 1)

def get_OCV(soc):
    """
    使用截图中的八次多项式拟合模型计算开路电压 (OCV)。
    OCV(SOC) = k0 + k1*SOC + k2*SOC^2 + ... + k8*SOC^8
    """
    return (3.5+(4.0-3.5)*soc)
    # return (2.85 + 4.80*soc - 17.8*soc**2 + 38.59*soc**3 + 4.91*soc**4
    #          - 210.52*soc**5 + 433.98*soc**6 - 368.45*soc**7 + 115.81*soc**8)

def total_voltage_model(inputs, R_base, I0):
    """
    综合电压模型 (用于参数拟合)。
    
    公式: V(t) = OCV(SOC) - I(t)*R_int(T) - Up(I, T)
    
    其中:
    1. R_int(T) 使用 Arrhenius 模型:
       R_int(T) = R_base * exp( (Ea/R) * (1/T - 1/T_ref) )
       (此处 R_base 即为参考温度 T_ref 下的内阻)
       
    2. Up 使用 Tafel 公式 (Butler-Volmer 在高电位下的简化):
       Up = (2*R*T/F) * arcsinh( I / (2*I0) )
    
    参数:
        inputs: tuple (I, OCV_val, T_celsius)
        R_base: 待拟合参数，参考温度下的欧姆内阻 (Ohm)
        I0:     待拟合参数，交换电流 (A)
        
    返回:
        V_pred: 预测的端电压序列 (V)
    """
    I, OCV_val, T_celsius = inputs
    
    # 转换温度为开尔文
    T_kelvin = T_celsius + 273.15
    
    # --- 1. 计算受温度影响的欧姆内阻 ---
    # exponent = (Ea / R) * (1/T - 1/T_ref)
    exponent = (E_ACTIVATION / R_GAS) * (1.0 / T_kelvin - 1.0 / T_REF)
    R_int_T = R_base * np.exp(exponent)
    
    # 欧姆压降
    U_ohm = I * R_int_T
    
    # --- 2. 计算极化电压 Up (Tafel) ---
    # 系数 A = 2RT / F
    coeff = (2 * R_GAS * T_kelvin) / FARADAY
    
    # 使用 arcsinh 计算极化压降。
    # 加一个极小的 offset (1e-12) 防止 I0 被拟合到 0 时发生除零错误
    U_p = coeff * np.arcsinh(I / (2 * I0 + 1e-12))
    
    # --- 3. 计算端电压 ---
    V_pred = OCV_val - U_ohm - U_p
    
    return V_pred