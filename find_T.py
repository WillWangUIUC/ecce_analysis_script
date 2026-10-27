import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
from pathlib import Path

from fitting_script import SCRIPT_DIR, data_folder

def extract_T_merge(output_text):
    """
    从拟合脚本的输出中提取 T_merge 的值
    """
    pattern = r'T_merge\s*=\s*([0-9.]+)'
    match = re.search(pattern, output_text)
    if match:
        return float(match.group(1))
    return None

def run_fitting_script(t_min, t_max, script_path="fitting_script.py"):
    """
    调用拟合脚本，传入 t_min 和 t_max
    只返回 T_merge 值
    """
    script_path = Path(script_path).expanduser()
    if not script_path.is_absolute():
        script_path = SCRIPT_DIR / script_path
    cmd = [sys.executable, str(script_path.resolve()), str(t_min), str(t_max)]
    
    try:
        result = subprocess.run(
            cmd, cwd=data_folder,
            capture_output=True, 
            text=True, 
            check=True,
            timeout=60
        )
        
        T_merge = extract_T_merge(result.stdout)
        return T_merge
        
    except Exception as e:
        print(f"t_min={t_min}, t_max={t_max} 拟合失败: {e}")
        return None

def main():
    # ========== 修改这里 ==========
    fit_length = 4000  # 固定的拟合时间长度
    t_min_start = 1000  # 起始 t_min
    t_min_end = 2000   # 结束 t_min
    step = 50         # 步长
    
    # 生成 t_min 列表
    t_initial_list = range(t_min_start, t_min_end + 1, step)
    # =============================
    
    script_path = SCRIPT_DIR / "fitting_script.py"
    
    if not os.path.exists(script_path):
        print(f"错误：找不到拟合脚本 '{script_path}'")
        return
    
    # 存储结果
    t_min_values = []
    t_max_values = []
    T_values = []
    
    print("开始批量拟合...")
    print(f"固定拟合时间长度 = {fit_length}")
    print(f"t_min 范围: {t_min_start} 到 {t_min_end}, 步长 {step}")
    print(f"对应的 t_max = t_min + {fit_length}")
    print("-" * 60)
    
    for t_min in t_initial_list:
        t_max = t_min + fit_length  # t_max = t_min + 4000
        print(f"拟合: t_min = {t_min}, t_max = {t_max}", end=" ")
        
        T_merge = run_fitting_script(t_min, t_max, script_path)
        
        if T_merge is not None:
            t_min_values.append(t_min)
            t_max_values.append(t_max)
            T_values.append(T_merge)
            print(f"-> T_merge = {T_merge:.2f}")
        else:
            print(f"-> 失败")
    
    if len(t_min_values) == 0:
        print("没有有效的拟合结果！")
        return
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：T_merge vs t_min
    ax1 = axes[0]
    ax1.plot(t_min_values, T_values, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=np.average(T_values), color='r', linestyle='--', alpha=0.7, 
                label=f'average = {np.average(T_values):.2f}')
    ax1.set_xlabel('t_min', fontsize=12)
    ax1.set_ylabel('T_merge', fontsize=12)
    ax1.set_title(f'T_merge vs t_min (fit length = {fit_length})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：T_merge vs t_max
    ax2 = axes[1]
    ax2.plot(t_max_values, T_values, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=np.average(T_values), color='b', linestyle='--', alpha=0.7,
                label=f'average = {np.average(T_values):.2f}')
    ax2.set_xlabel('t_max (= t_min + 4000)', fontsize=12)
    ax2.set_ylabel('T_merge', fontsize=12)
    ax2.set_title(f'T_merge vs t_max', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(data_folder / 'T_merge_vs_tmin_fixed_length.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存数据到文件
    data = np.column_stack([t_min_values, t_max_values, T_values])
    np.savetxt(data_folder / 'T_merge_data_fixed_length.txt', data,
               header='t_min    t_max    T_merge', fmt='%.2f')
    
    print(f"\n完成！有效拟合: {len(t_min_values)}/{len(t_initial_list)}")
    print(f"图表已保存为 'T_merge_vs_tmin_fixed_length.png'")
    print(f"数据已保存为 'T_merge_data_fixed_length.txt'")
    print(f"T_merge 平均值: {np.average(T_values):.2f}")
    print(f"T_merge 标准差: {np.std(T_values):.2f}")

if __name__ == "__main__":
    main()
