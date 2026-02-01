#%%
import matplotlib.pyplot as plt
from nasa_battery_reader import load_battery_data  # 导入刚才写的接口

# 1. 读取数据
filename = '../Battery Data Set/1/B0005.mat' # 替换为实际路径
cycles = load_battery_data(filename)

# 2. 提取容量曲线 (复现你 MATLAB 的逻辑)
discharge_capacities = []
cycle_indices = []

for cycle in cycles:
    if cycle['type'] == 'discharge':
        cap = cycle['data']['capacity']
        discharge_capacities.append(cap)
        cycle_indices.append(cycle['cycle_id'])

# 3. 提取第一个放电周期的电压曲线
first_discharge = [c for c in cycles if c['type'] == 'discharge'][0]
time_vec = first_discharge['data']['time']
volt_vec = first_discharge['data']['voltage_measured']

# 4. 提取阻抗 Re 的变化
re_values = []
re_indices = []
for cycle in cycles:
    if cycle['type'] == 'impedance':
        re_values.append(cycle['data']['Re'])
        re_indices.append(cycle['cycle_id'])

# --- 绘图 (Python matplotlib) ---
plt.figure(figsize=(12, 8))

# 子图1: 容量衰减
plt.subplot(2, 2, 1)
plt.plot(range(1, len(discharge_capacities)+1), discharge_capacities, 'b.-')
plt.title('Capacity Degradation')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.grid(True)

# 子图2: 第1次放电电压
plt.subplot(2, 2, 2)
plt.plot(time_vec, volt_vec, 'r-')
plt.title('Voltage Profile (Cycle 1)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)

# 子图3: 阻抗 Re 变化
plt.subplot(2, 2, 3)
plt.plot(re_indices, re_values, 'g.-')
plt.title('Impedance (Re) Change')
plt.xlabel('Cycle ID')
plt.ylabel('Re (Ohms)')
plt.grid(True)

plt.tight_layout()
plt.show()
# %%
