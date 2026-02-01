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
    # 限制 SOC 不小于 0.01，防止 1/SOC 计算时除以零
    return np.clip(soc, 0.01, 1)

def get_OCV(soc):
    """ OCV 模型 """
    return (3.5+(4.0-3.5)*soc)
    return (2.85 + 4.80*soc - 17.8*soc**2 + 38.59*soc**3 + 4.91*soc**4
             - 210.52*soc**5 + 433.98*soc**6 - 368.45*soc**7 + 115.81*soc**8)

def total_voltage_model(inputs, R_base, k_low_soc):
    """
    自适应内阻模型
    
    公式: V = OCV - I * R_total(SOC, T)
    
    其中 R_total = (R_base + k_low_soc / SOC) * Arrhenius(T)
    
    参数:
      R_base:    基础欧姆内阻 (Ohm)
      k_low_soc: 低SOC区域的阻抗上升系数 (Ohm)
    """
    I, OCV_val, T_celsius, SOC_val = inputs
    
    T_kelvin = T_celsius + 273.15
    
    # 1. 计算温度修正系数 (Arrhenius)
    temp_correction = np.exp((E_ACTIVATION / R_GAS) * (1.0/T_kelvin - 1.0/T_REF))
    
    # 2. 计算与 SOC 相关的总内阻
    #    模型假设：电阻由基础值 + 低电量时的极化增加值组成
    #    使用 0.1/SOC 是为了让系数 k 的量级在数学上更可控，或者直接用 1/SOC
    #    这里使用简单的 R = R_base + k * (1/SOC - 1) 
    #    (当 SOC=1时 R=R_base; 当 SOC->0时 R增加)
    
    # 也就是 R_soc = R_base + k_low_soc * (1.0/SOC_val - 1.0)
    # 这样定义 k_low_soc 物理意义更明确：SOC越低，增加的电阻越多
    R_soc_part = R_base + k_low_soc * (1.0/SOC_val - 1.0)
    
    # 3. 综合内阻
    R_total = R_soc_part * temp_correction
    
    # 4. 计算电压
    V_pred = OCV_val - I * R_total
    
    return V_pred