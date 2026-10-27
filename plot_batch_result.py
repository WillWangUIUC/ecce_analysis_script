import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from fitting_script import data_folder, resolve_data_path

def plot_batch_results(filename):
    """
    讀取批次擬合結果並繪圖（包含所有拟合参数）
    """
    # 讀取數據
    filename = resolve_data_path(filename)
    data = np.loadtxt(filename)
    
    # 提取各列數據（根据新格式）
    t_min = data[:, 0]
    a1 = data[:, 1]
    a2 = data[:, 2]
    T_merge = data[:, 3]
    B = data[:, 4]
    omega = data[:, 5]
    phi = data[:, 6]
    e = data[:, 7]
    e_err = data[:, 8]
    delta_vr = data[:, 9]
    delta_vr_err = data[:, 10]
    delta_e = data[:, 11]
    delta_e_err = data[:, 12]
    R_squared = data[:, 13]
    
    # 找到有效数据（过滤掉 NaN）
    valid = ~np.isnan(R_squared)
    t_min_valid = t_min[valid]
    a1_valid = a1[valid]
    a2_valid = a2[valid]
    T_merge_valid = T_merge[valid]
    B_valid = B[valid]
    omega_valid = omega[valid]
    phi_valid = phi[valid]
    e_valid = e[valid]
    e_err_valid = e_err[valid]
    delta_vr_valid = delta_vr[valid]
    delta_vr_err_valid = delta_vr_err[valid]
    delta_e_valid = delta_e[valid]
    delta_e_err_valid = delta_e_err[valid]
    R_squared_valid = R_squared[valid]
    
    # 创建 3x4 的子图布局（12个子图）
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    fig.suptitle('Batch Fitting Results - All Parameters vs t_min', fontsize=16)
    
    # ========== 第一行：幂律参数 ==========
    # 图1: a1
    ax = axes[0, 0]
    ax.plot(t_min_valid, a1_valid, 'o-', color='blue', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('a1')
    ax.set_title('a1 (Power Law Coefficient)')
    ax.grid(True, alpha=0.3)
    
    # 图2: a2
    ax = axes[0, 1]
    ax.plot(t_min_valid, a2_valid, 's-', color='red', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('a2')
    ax.set_title('a2 (Power Law Coefficient)')
    ax.grid(True, alpha=0.3)
    
    # 图3: T_merge
    ax = axes[0, 2]
    ax.plot(t_min_valid, T_merge_valid, '^-', color='green', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('T_merge')
    ax.set_title('T_merge (Coalescence Time)')
    ax.grid(True, alpha=0.3)
    
    # 图4: R²
    ax = axes[0, 3]
    ax.plot(t_min_valid, R_squared_valid, 'o-', color='black', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('R²')
    ax.set_title('Goodness of Fit')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.01])
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    
    # ========== 第二行：余弦参数 ==========
    # 图5: B
    ax = axes[1, 0]
    ax.plot(t_min_valid, B_valid, 'd-', color='purple', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('B')
    ax.set_title('B (Cosine Amplitude)')
    ax.grid(True, alpha=0.3)
    
    # 图6: omega
    ax = axes[1, 1]
    ax.plot(t_min_valid, omega_valid, 'v-', color='orange', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('ω')
    ax.set_title('ω (Angular Frequency)')
    ax.grid(True, alpha=0.3)
    
    # 图7: phi
    ax = axes[1, 2]
    ax.plot(t_min_valid, phi_valid, '*-', color='brown', markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('φ')
    ax.set_title('φ (Phase)')
    ax.grid(True, alpha=0.3)
    
    
    # ========== 第三行：物理参数 ==========
    # 图9: Eccentricity e with error bars
    ax = axes[2, 0]
    ax.errorbar(t_min_valid, e_valid, yerr=e_err_valid, fmt='o-', 
                color='darkblue', capsize=3, markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('e')
    ax.set_title('Eccentricity e')
    ax.grid(True, alpha=0.3)
    
    # 图10: delta_vr with error bars
    ax = axes[2, 1]
    ax.errorbar(t_min_valid, delta_vr_valid, yerr=delta_vr_err_valid, fmt='s-', 
                color='darkred', capsize=3, markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('Δv_r')
    ax.set_title('Radial Velocity Difference')
    ax.grid(True, alpha=0.3)
    
    # 图11: delta_e with error bars
    ax = axes[2, 2]
    ax.errorbar(t_min_valid, delta_e_valid, yerr=delta_e_err_valid, fmt='^-', 
                color='darkgreen', capsize=3, markersize=4, linewidth=1.5)
    ax.set_xlabel('t_min')
    ax.set_ylabel('ΔΩ')
    ax.set_title('Angular Velocity Difference')
    ax.grid(True, alpha=0.3)
    
    # 图12: 综合信息（或留空）
    ax = axes[2, 3]
    # 可以画 R² 作为参考
    ax.plot(t_min_valid, R_squared_valid, 'o-', color='black', markersize=4, linewidth=1.5, label='R²')
    ax.set_xlabel('t_min')
    ax.set_ylabel('R²')
    ax.set_title('Goodness of Fit')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.01])
    ax.legend(fontsize='small')
    
    plt.tight_layout()
    
    # 保存图片
    base_name = os.path.splitext(filename)[0]
    output_file = f"{base_name}_all_params_plots.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")
    
    # 显示统计信息
    print(f"\n=== 统计信息 ===")
    print(f"e 的范围: {np.min(e_valid):.6e} 到 {np.max(e_valid):.6e}")
    print(f"e 的平均值: {np.mean(e_valid):.6e} ± {np.std(e_valid):.6e}")
    print(f"delta_vr 的范围: {np.min(delta_vr_valid):.6e} 到 {np.max(delta_vr_valid):.6e}")
    print(f"delta_e 的范围: {np.min(delta_e_valid):.6e} 到 {np.max(delta_e_valid):.6e}")
    print(f"R² 的范围: {np.min(R_squared_valid):.6f} 到 {np.max(R_squared_valid):.6f}")
    
    # 找出最佳拟合窗口
    best_idx = np.argmax(R_squared_valid)
    print(f"\n最佳拟合窗口: t_min = {t_min_valid[best_idx]:.2f}")
    print(f"  对应 R² = {R_squared_valid[best_idx]:.6f}")
    print(f"  对应 e = {e_valid[best_idx]:.6e} ± {e_err_valid[best_idx]:.6e}")
    
    plt.show()
    return t_min_valid, e_valid, e_err_valid, R_squared_valid

def plot_simple_summary(filename):
    """
    简化版绘图（只画 e, delta_vr, delta_e, R²）
    """
    filename = resolve_data_path(filename)
    data = np.loadtxt(filename)
    
    t_min = data[:, 0]
    e = data[:, 7]
    e_err = data[:, 8]
    delta_vr = data[:, 9]
    delta_vr_err = data[:, 10]
    delta_e = data[:, 11]
    delta_e_err = data[:, 12]
    R_squared = data[:, 13]
    
    # 过滤 NaN
    valid = ~np.isnan(R_squared)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # Eccentricity
    ax = axes[0]
    ax.errorbar(t_min[valid], e[valid], yerr=e_err[valid], fmt='o-', 
                color='blue', capsize=3, markersize=4)
    ax.set_xlabel('t_min')
    ax.set_ylabel('e')
    ax.set_title('Eccentricity vs Time Window')
    ax.grid(True, alpha=0.3)
    
    # delta_vr
    ax = axes[1]
    ax.errorbar(t_min[valid], delta_vr[valid], yerr=delta_vr_err[valid], fmt='s-', 
                color='red', capsize=3, markersize=4)
    ax.set_xlabel('t_min')
    ax.set_ylabel('Δv_r')
    ax.set_title('Delta v_r vs Time Window')
    ax.grid(True, alpha=0.3)
    
    # delta_e
    ax = axes[2]
    ax.errorbar(t_min[valid], delta_e[valid], yerr=delta_e_err[valid], fmt='^-', 
                color='green', capsize=3, markersize=4)
    ax.set_xlabel('t_min')
    ax.set_ylabel('Δe')
    ax.set_title('Delta e vs Time Window')
    ax.grid(True, alpha=0.3)
    
    # R²
    ax = axes[3]
    ax.plot(t_min[valid], R_squared[valid], 'o-', color='magenta', markersize=4)
    ax.set_xlabel('t_min')
    ax.set_ylabel('R²')
    ax.set_title('R² vs Time Window')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, 1.01])
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    base_name = os.path.splitext(filename)[0]
    output_file = f"{base_name}_summary.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"摘要图已保存到: {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = resolve_data_path(sys.argv[1])
    else:
        batch_files = list(data_folder.glob("batch_fitting_results_w*.txt"))
        if batch_files:
            filename = max(batch_files, key=os.path.getmtime)
            print(f"自动选择档案: {filename}")
        else:
            print("错误: 找不到批次拟合结果档案")
            sys.exit(1)
    
    # 绘制完整参数图
    plot_batch_results(filename)
    
    # 也绘制简化摘要图
    plot_simple_summary(filename)
