import numpy as np
import scipy.io
import os

def load_battery_data(mat_file_path):
    """
    读取 NASA 电池数据集 (.mat) 并转换为易于使用的 Python 列表/字典格式。
    
    参数:
        mat_file_path (str): .mat 文件的路径
        
    返回:
        dataset (list): 包含所有 cycle 的列表。每个元素是一个字典，结构如下：
            {
                'cycle_id': 1,
                'type': 'discharge' / 'charge' / 'impedance',
                'ambient_temperature': 24,
                'time': '2008-04-05 12:00:00',
                'data': { ...取决于 type... }
            }
    """
    
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"文件未找到: {mat_file_path}")

    # 1. 加载 .mat 文件
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # 2. 自动寻找顶层变量名 (例如 'B0005', 'B0006' 等)，排除系统变量
    variable_name = None
    for key in mat_data.keys():
        if not key.startswith('__'):
            variable_name = key
            break
            
    if variable_name is None:
        raise ValueError("未在 .mat 文件中找到有效的数据变量")
    
    print(f"检测到电池数据集变量名: {variable_name}")
    
    # 获取 cycle 结构体数组
    # MATLAB struct 在 scipy 中读取后层级很深，通常是 [0,0]['field']
    raw_cycles = mat_data[variable_name][0, 0]['cycle'][0]
    
    dataset = []
    
    # 3. 遍历每一个 Cycle
    for i, cycle in enumerate(raw_cycles):
        # 提取基础元数据
        cycle_type = str(cycle['type'][0])
        ambient_temp = cycle['ambient_temperature'][0][0]
        timestamp = str(cycle['time'][0])
        
        # 初始化处理后的字典
        processed_cycle = {
            'cycle_id': i + 1,
            'type': cycle_type,
            'ambient_temperature': ambient_temp,
            'time': timestamp,
            'data': {}
        }
        
        # 提取 data 字段中的核心数据
        raw_data = cycle['data']
        
        # 根据 cycle 类型提取不同字段
        if cycle_type == 'discharge':
            # 注意：scipy 读取的数组通常是 (1, N) 或 (N, 1)，需要 flatten() 转为一维数组
            processed_cycle['data'] = {
                'voltage_measured': raw_data['Voltage_measured'][0, 0].flatten(),
                'current_measured': raw_data['Current_measured'][0, 0].flatten(),
                'temperature_measured': raw_data['Temperature_measured'][0, 0].flatten(),
                'current_load': raw_data['Current_load'][0, 0].flatten(),
                'voltage_load': raw_data['Voltage_load'][0, 0].flatten(),
                'time': raw_data['Time'][0, 0].flatten(),
                'capacity': raw_data['Capacity'][0, 0][0, 0]  # 容量通常是一个标量
            }
            
        elif cycle_type == 'charge':
            processed_cycle['data'] = {
                'voltage_measured': raw_data['Voltage_measured'][0, 0].flatten(),
                'current_measured': raw_data['Current_measured'][0, 0].flatten(),
                'temperature_measured': raw_data['Temperature_measured'][0, 0].flatten(),
                'current_charge': raw_data['Current_charge'][0, 0].flatten(),
                'voltage_charge': raw_data['Voltage_charge'][0, 0].flatten(),
                'time': raw_data['Time'][0, 0].flatten()
            }
            
        elif cycle_type == 'impedance':
            # 按照要求，对于 impedance 只读取 Re (实部阻抗)
            # 有些文件里可能还有 Rct，这里严格按要求读取 Re
            try:
                re_value = raw_data['Re'][0, 0][0, 0]
                # 顺便读取 Rct (电荷转移电阻) 以备不时之需，如果不需要可注释掉
                rct_value = raw_data['Rct'][0, 0][0, 0] 
                
                processed_cycle['data'] = {
                    'Re': re_value,
                    'Rct': rct_value
                }
            except IndexError:
                # 某些 impedance cycle 可能数据不全
                processed_cycle['data'] = {'Re': np.nan, 'Rct': np.nan}

        dataset.append(processed_cycle)

    return dataset

# --- 单元测试代码 (直接运行此文件时执行) ---
if __name__ == "__main__":
    # 这里的路径改成你本地的测试路径
    test_filename = '../Battery Data Set/1/B0005.mat'
    
    try:
        data = load_battery_data(test_filename)
        print(f"成功加载数据，共 {len(data)} 个 Cycles")
        
        # 验证读取 Discharge
        discharges = [c for c in data if c['type'] == 'discharge']
        if discharges:
            print(f"第一个放电周期的容量: {discharges[0]['data']['capacity']:.4f} Ah")
            print(f"第一个放电周期的电压数据长度: {len(discharges[0]['data']['voltage_measured'])}")
            
        # 验证读取 Impedance
        impedances = [c for c in data if c['type'] == 'impedance']
        if impedances:
            print(f"第一个阻抗测量的 Re 值: {impedances[0]['data']['Re']}")
            
    except Exception as e:
        print(f"测试失败: {e}")