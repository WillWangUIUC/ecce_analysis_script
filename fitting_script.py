MERGE = False  # Set to False to keep a1_fit and a2_fit at zero.

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
import sys
import argparse
import json
from pathlib import Path
import time as clock
import traceback
import warnings


SCRIPT_DIR = Path(__file__).resolve().parent
# Case folder containing output_all_vol_integral/time_separation.txt.
# Change this one setting for all scripts in analysis_script. Absolute paths
# are supported; relative paths are anchored to this script, never the shell.
data_folder = Path("../A_2/ecce_fit")
data_folder = data_folder.expanduser()
if not data_folder.is_absolute():
    data_folder = SCRIPT_DIR / data_folder
data_folder = data_folder.resolve()


def resolve_data_path(filename):
    """Resolve input/output paths relative to the configured case folder."""
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = data_folder / path
    return path.resolve()


def fit_with_diagnostics(stage, model, time, values, names, p0, bounds, maxfev=1000):
    """Report the optimizer inputs, warnings, and outcome for one fitting stage."""
    print(f"\n[FIT {stage}] Starting: points={len(time)}, maxfev={maxfev}", flush=True)
    for name, value, lower, upper in zip(names, p0, bounds[0], bounds[1]):
        print(f"  {name}: initial={value:.12g}, bounds=[{lower:.12g}, {upper:.12g}]")
    started = clock.perf_counter()
    try:
        for name, value, lower, upper in zip(names, p0, bounds[0], bounds[1]):
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"Initial {name}={value!r} is not finite or is outside "
                    f"its bounds [{lower}, {upper}]."
                )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                popt, pcov = curve_fit(
                    model, time, values, p0=p0, bounds=bounds, maxfev=maxfev
                )
            finally:
                for warning in caught:
                    print(
                        f"[WARN {stage}] {warning.category.__name__}: {warning.message}",
                        file=sys.stderr, flush=True,
                    )
    except Exception as exc:
        print(
            f"[ERROR {stage}] Failed after {clock.perf_counter() - started:.3f}s: "
            f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True,
        )
        traceback.print_exc()
        raise

    print(f"[FIT {stage}] Optimizer returned after {clock.perf_counter() - started:.3f}s")
    print("  " + ", ".join(f"{name}={value:.12g}" for name, value in zip(names, popt)))
    if not np.all(np.isfinite(popt)):
        raise ValueError(f"{stage}: optimizer returned non-finite parameters: {popt}")
    residual = values - model(time, *popt)
    print(f"  residual RMS={np.sqrt(np.mean(residual**2)):.6e}")
    if not np.all(np.isfinite(residual)):
        raise ValueError(f"{stage}: fitted model produces NaN/Inf residuals.")
    for name, value, lower, upper in zip(names, popt, bounds[0], bounds[1]):
        if ((np.isfinite(lower) and np.isclose(value, lower, rtol=1e-6, atol=1e-10))
                or (np.isfinite(upper) and np.isclose(value, upper, rtol=1e-6, atol=1e-10))):
            print(f"[WARN {stage}] {name}={value:.12g} is at/near a parameter bound [{lower}, {upper}].")
    if not np.all(np.isfinite(pcov)):
        print(f"[WARN {stage}] Covariance contains NaN/Inf; parameter errors are unreliable.")
    else:
        try:
            condition = np.linalg.cond(pcov)
            print(f"  covariance condition number={condition:.6e} (sensitive to parameter scaling)")
            if condition > 1e12:
                print(f"[WARN {stage}] Poorly conditioned covariance; inspect parameter degeneracy/scales.")
        except np.linalg.LinAlgError as exc:
            print(f"[WARN {stage}] Cannot assess covariance: {exc}")
    return popt, pcov


def print_batch_result(results):
    """Emit an unambiguous summary independent of human-readable progress messages."""
    values = {
        'a1': results['power_law']['a1'], 'a2': results['power_law']['a2'],
        'T_merge': results['power_law']['T'],
        **results['cosine'],
        'e': results['e_val'], 'e_err': results['e_err'],
        'delta_vr': results['delta_vr_val'], 'delta_vr_err': results['delta_vr_err'],
        'delta_e': results['delta_e_val'], 'delta_e_err': results['delta_e_err'],
        'R_squared': results['r_squared'],
        'average_separation': results['average_separation'],
        'average_separation_err': results['average_separation_err'],
        'max_derivative_time': results['max_derivative_time'],
    }
    payload = {
        key: float(value) if value is not None and np.isfinite(value) else None
        for key, value in values.items()
    }
    payload['fit_status'] = 'ok' if results['stage_status']['full_model'] else 'fallback'
    payload['stage_status'] = results['stage_status']
    print('FIT_RESULT_JSON: ' + json.dumps(payload, allow_nan=False), flush=True)

def read_data(filename):
    """读取数据文件并计算两颗星球的距离/2"""
    filename = resolve_data_path(filename)
    print(f"[DATA] Reading {Path(filename).resolve()}", flush=True)
    data = np.loadtxt(filename, ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 2:
        raise ValueError(f"Expected nonempty time/separation columns; got shape {data.shape}.")
    if not np.all(np.isfinite(data[:, :2])):
        raise ValueError("Input time/separation columns contain NaN or Inf.")
    
    # 提取时间和位置数据
    time = data[:, 0]    
    distance = data[:, 1]
    # 測試數據
    #time = np.linspace(0, 6000, 6000)
    #distance = 0.5*np.sin((2 * np.pi / 1000) * time + 1.1) + 2 * (22000 - time)**0.25 + 0.0001*2 * (22000 - time)**1.25
    
    return time, distance, distance[0]  # 返回初始距离作为 dini

def power_law_model(t, a1, a2, T):
    """幂律模型: a1 * (T - t)^{1/4} + a2 * (T - t)^{5/4}"""
    #T = 1000000
    T = 10000
    diff = T -  t
    valid_mask = diff > 0
    result = np.zeros_like(t, dtype=float)
    result[valid_mask] = a1 * np.power(diff[valid_mask], 0.25) + 0.0001*a2 * np.power(diff[valid_mask], 1.25)
    result[~valid_mask] = 0
    return result

def cosine_model_simple(t, B, omega, phi):
    """简单余弦模型: B/omega * cos(omega*t + phi)"""
    return B / omega * np.cos(omega * t + phi)

def cosine_model_with_constant(t, B, omega, phi, C):
    """Non-merging separation: C + B/omega * cos(omega*t + phi)."""
    return C + cosine_model_simple(t, B, omega, phi)


def estimate_period_autocorrelation(time, separation):
    """Estimate recurrence time from the mean-subtracted autocorrelation.

    Use the strongest positive local peak after the first nonpositive lag,
    excluding the zero-lag lobe. Normalize by the zero-lag correlation; do
    not divide by overlap count, which would amplify sparsely sampled tails.
    Nonuniform samples are linearly interpolated onto a uniform grid only
    for initialization; the fit still uses the original samples.
    Return None if a recurrence peak cannot be calculated.
    """
    time = np.asarray(time, dtype=float)
    separation = np.asarray(separation, dtype=float)
    if (time.ndim != 1 or separation.shape != time.shape or time.size < 3
            or not np.all(np.isfinite(time)) or not np.all(np.isfinite(separation))):
        return None
    order = np.argsort(time)
    time, separation = time[order], separation[order]
    if np.any(np.diff(time) <= 0):
        return None

    sample_dt = (time[-1] - time[0]) / (time.size - 1)
    if not np.allclose(np.diff(time), sample_dt, rtol=1e-6, atol=0.0):
        uniform_time = np.linspace(time[0], time[-1], time.size)
        separation = np.interp(uniform_time, time, separation)
    centered = separation - np.mean(separation)
    autocorrelation = correlate(centered, centered, mode='full', method='fft')[time.size - 1:]
    if not np.isfinite(autocorrelation[0]) or autocorrelation[0] <= 0.0:
        return None
    autocorrelation /= autocorrelation[0]
    nonpositive_lags = np.flatnonzero(autocorrelation[1:] <= 0.0) + 1
    if nonpositive_lags.size == 0:
        return None
    peaks, _ = find_peaks(autocorrelation)
    peaks = peaks[(peaks > nonpositive_lags[0]) & (autocorrelation[peaks] > 0.0)]
    if peaks.size == 0:
        return None
    peak_lag = peaks[np.argmax(autocorrelation[peaks])]
    return float(peak_lag * sample_dt)


def full_model(t, a1, a2, T, B, omega, phi, C=0.0):
    """完整模型: 常数 + 幂律 + 余弦 (C=0 when MERGE=True)."""
    return C + power_law_model(t, a1, a2, T) + cosine_model_simple(t, B, omega, phi)

def compute_derived_errors(B, omega, phi, dini, e_val, pcov):
    """
    计算导出量的误差
    
    e = B / (omega * dini)
    delta_vr = B * sin(phi)
    delta_e = -B * cos(phi) / (omega * dini) * (1 - e_val)
    """
    # 提取协方差矩阵元素
    B_idx = 3
    omega_idx = 4
    phi_idx = 5
    
    var_B = pcov[B_idx, B_idx]
    var_omega = pcov[omega_idx, omega_idx]
    var_phi = pcov[phi_idx, phi_idx]
    cov_B_omega = pcov[B_idx, omega_idx]
    cov_B_phi = pcov[B_idx, phi_idx]
    cov_omega_phi = pcov[omega_idx, phi_idx]
    
    # ========== e 的误差 ==========
    de_dB = 1.0 / (omega * dini)
    de_domega = -B / (omega**2 * dini)
    de_dphi = 0.0
    
    e_err = np.sqrt(
        de_dB**2 * var_B + 
        de_domega**2 * var_omega + 
        2 * de_dB * de_domega * cov_B_omega
    )
    
    # ========== delta_vr 的误差 ==========
    dvr_dB = np.sin(phi)
    dvr_dphi = B * np.cos(phi)
    dvr_domega = 0.0
    
    delta_vr_err = np.sqrt(
        dvr_dB**2 * var_B + 
        dvr_dphi**2 * var_phi + 
        2 * dvr_dB * dvr_dphi * cov_B_phi
    )
    
    # ========== delta_e 的误差 ==========
    # delta_e = -B * cos(phi) / (omega * dini) * (1 - e_val)
    # 其中 e_val = B / (omega * dini)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    omega_dini = omega * dini
    
    dde_dB = -cos_phi / omega_dini + 2 * B * cos_phi / (omega_dini**2)
    dde_domega = B * cos_phi / (omega**2 * dini) - 2 * B**2 * cos_phi / (omega**3 * dini**2)
    dde_dphi = B * sin_phi / omega_dini - B**2 * sin_phi / (omega_dini**2)
    
    delta_e_err = np.sqrt(
        dde_dB**2 * var_B + 
        dde_domega**2 * var_omega + 
        dde_dphi**2 * var_phi +
        2 * dde_dB * dde_domega * cov_B_omega +
        2 * dde_dB * dde_dphi * cov_B_phi +
        2 * dde_domega * dde_dphi * cov_omega_phi
    )
    
    return e_err, delta_vr_err, delta_e_err

def fit_data(filename, t_min=None, t_max=None, m=1.0):
    """
    主拟合函数
    
    参数:
    - filename: 数据文件名
    - t_min: 拟合起始时间（默认None，使用数据起始时间）
    - t_max: 拟合结束时间（默认None，使用数据结束时间）
    """
    # 读取全部数据
    time_all, separation_all, dini = read_data(filename)
    stage_status = {'power_law': False, 'cosine': False, 'full_model': False}
    if dini <= 0 or not np.isfinite(m) or m <= 0:
        raise ValueError(f"Initial separation and mass must be positive: dini={dini}, m={m}.")
    
    # 应用时间范围过滤
    if t_min is None:
        t_min = time_all[0]
    if t_max is None:
        t_max = time_all[-1]
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
        raise ValueError(f"Invalid fitting interval [{t_min}, {t_max}]; require finite t_min < t_max.")
    
    mask = (time_all >= t_min) & (time_all <= t_max)
    time = time_all[mask]
    separation = separation_all[mask]
    print(f"[DATA] Available times: [{time_all.min():.6f}, {time_all.max():.6f}]")
    print(f"[DATA] Requested interval: [{t_min:.6f}, {t_max:.6f}], selected points={len(time)}")
    if len(time) <= 6:
        raise ValueError(f"Need more than 6 points for the full fit and covariance; selected {len(time)}.")
    
    print(f"总数据点数: {len(time_all)}")
    print(f"拟合数据点数: {len(time)}")
    print(f"拟合时间范围: {time[0]:.6f} 到 {time[-1]:.6f}")



    
    C_guess = C_fit = 0.0
    average_separation = average_separation_err = max_derivative_time = None

    # ========== 第一步：拟合幂律部分 ==========
    if MERGE:
        print("\n=== 第一步：拟合幂律模型 power_law_model ===")
        T_guess = 5*dini**4/512/m**3*2
        a_guess = 4*(m**2*2*m/5)**0.25/2
        #T_guess = 22000.0
        #a_guess = 0.1
        print(f"初始估计 T_guess = {T_guess:.6f}")
        print(f"初始估计 a_guess = {a_guess:.6f}")
        #T_guess = time[-1] * 1.1
        #a_guess = separation[0] / np.power(T_guess - time[0], 0.25)

        try:
            popt_power, pcov_power = fit_with_diagnostics(
                '1 power_law', power_law_model, time, separation,
                names=['a1', 'a2', 'T'],
                p0=[a_guess, 0.0, T_guess],
                bounds=([0.0,    0.0,    time[-1]*1.01],
                        [np.inf, np.inf, time[-1]*10]),
                maxfev=1000
            )
            a1_fit, a2_fit, T_fit = popt_power
            stage_status['power_law'] = True
            print(f"拟合参数: a1 = {a1_fit:.6f}, a2 = {a2_fit:.6f}, T = {T_fit:.6f}")

            power_fit = power_law_model(time, a1_fit, a2_fit, T_fit)
            residuals = separation - power_fit

        except Exception as e:
            stage_status['power_law'] = False
            print(f"[WARN step 1] {type(e).__name__}: {e}; continuing with initial guesses.")
            a1_fit = a_guess
            a2_fit = 0.0
            T_fit = T_guess
            power_fit = power_law_model(time, a1_fit, a2_fit, T_fit)
            residuals = separation - power_fit
            print(f"使用估计参数: a1 = {a1_fit:.6f}, a2 = 0.0, T = {T_fit:.6f}")
    else:
        print("\n[FIT 1 power_law] Skipped: MERGE=False; a1=a2=0.")
        a1_fit = a2_fit = 0.0
        T_fit = 0.0  # Unused when the power-law contribution is disabled.
        power_fit = np.zeros_like(time, dtype=float)
        # The all-time sample mean is only an initial guess; stage 3 fits C.
        C_guess = float(np.mean(separation_all))
        C_fit = C_guess
        average_separation = C_fit
        print(f"[INIT] All-time mean separation: C_guess = {C_guess:.12g}; C will be fitted.")
        residuals = separation - C_guess
    
    # ========== 第二步：简单余弦拟合 ==========
    print("\n=== 第二步：简单余弦拟合 B/omega*cos(omega*t+phi) ===")
        
    # Estimate the delay at which the selected signal resembles itself.
    measured_period = estimate_period_autocorrelation(time, separation)
    period_guess = 1000.0  # Previous default, used only if estimation fails.
    if measured_period is not None:
        if np.isfinite(measured_period) and measured_period > 0:
            period_guess = measured_period
            print(
                f"[INIT] Autocorrelation recurrence delay: period_guess={period_guess:.12g}"
            )
        else:
            print(
                f"[WARN init] Invalid autocorrelation delay {measured_period!r}; "
                "using fallback period_guess=1000."
            )
    else:
        print(
            "[WARN init] No autocorrelation recurrence peak available; "
            "using fallback period_guess=1000."
        )
    omega_guess = 2*np.pi / period_guess
    print(f"初始估计 omega_guess = {omega_guess:.6f}")
    B_guess = np.max(np.abs(residuals)) * omega_guess
    phi_guess = 0.0
    
    # Keep phase unbounded in every optimizer call: equivalent phases separated
    # by 2*pi must not become artificial boundaries in the search.
    try:
        popt_cosine_simple, pcov_cosine_simple = fit_with_diagnostics(
            '2 cosine', cosine_model_simple, time, residuals,
            names=['B', 'omega', 'phi'],
            p0=[B_guess, omega_guess, phi_guess],
            bounds=([0.0,    0.0,    -np.inf],
                    [np.inf, np.inf, np.inf]),
            maxfev=1000
        )
        B_simple, omega_simple, phi_simple = popt_cosine_simple
        stage_status['cosine'] = True
        print(f"简单拟合参数: B = {B_simple:.6f}, omega = {omega_simple:.6f}, phi = {phi_simple:.6f}")
        print(f"对应周期: T = {2*np.pi/omega_simple:.6f}")
        print(f"eccentricity e = {B_simple / omega_simple/dini:.6f}, delta vr = {B_simple * np.sin(phi_simple):.6f}, delta Omega = {B_simple * np.cos(phi_simple)/2/dini:.6f}")
        
    except Exception as e:
        stage_status['cosine'] = False
        print(f"[WARN step 2] {type(e).__name__}: {e}; continuing with initial guesses.")
        B_simple = B_guess
        omega_simple = omega_guess
        phi_simple = phi_guess
        print(f"使用估计参数: B = {B_simple:.6f}, omega = {omega_simple:.6f}, phi = {phi_simple:.6f}")
    
    
    # ========== 第三步：完整模型拟合（幂律 + 完整余弦）==========
    print("\n=== 第三步：完整模型拟合 power_law_model + B/omega*cos(omega*t+phi) ===")
    phi_fit = phi_simple  # Carry the continuous, unwrapped phase into step 3.
    pcov_full = None  # 初始化协方差矩阵
    B_fit = B_simple
    omega_fit = omega_simple

    
    try:
        if MERGE:
            popt_full, pcov_full = fit_with_diagnostics(
                '3 full_model', full_model, time, separation,
                names=['a1', 'a2', 'T', 'B', 'omega', 'phi'],
                p0=     [a1_fit, a2_fit, T_fit,       B_fit, omega_fit,phi_fit],  # 使用前两步结果作为初始值
                bounds=([0.0,    0.0,   time[-1]*1.01, 0.0,    0.0,   -np.inf],
                        [np.inf, np.inf,time[-1]*10  , np.inf, np.inf, np.inf]),
                maxfev=1000
            )
            a1_fit, a2_fit, T_fit, B_fit, omega_fit, phi_fit = popt_full
        else:
            popt_cosine, pcov_cosine = fit_with_diagnostics(
                '3 cosine + fitted C (MERGE=False)', cosine_model_with_constant, time, separation,
                names=['B', 'omega', 'phi', 'C'],
                p0=[B_fit, omega_fit, phi_fit, C_guess],
                bounds=([0.0, np.finfo(float).eps, -np.inf, -np.inf],
                        [np.inf, np.inf, np.inf, np.inf]),
                maxfev=1000
            )
            B_fit, omega_fit, phi_fit, C_fit = popt_cosine
            average_separation = C_fit
            average_separation_err = np.sqrt(pcov_cosine[3, 3])
            print(f"  Fitted C = {C_fit:.12g} +/- {average_separation_err:.6e}")
            # Append C, preserving the existing indices of B, omega, and phi.
            # Layout: [a1, a2, T, B, omega, phi, C], including C cross-covariances.
            pcov_full = np.zeros((7, 7), dtype=float)
            pcov_full[3:, 3:] = pcov_cosine
        # Normalize only after optimization, for reporting and derived values.
        # A 2*pi shift preserves the model and its local covariance.
        phi_fit = np.mod(phi_fit + np.pi, 2*np.pi) - np.pi
        print(f"完整模型拟合参数:")
        print(f"  幂律部分: a1 = {a1_fit:.6f}, a2 = {a2_fit:.6f}, T = {T_fit:.6f}")
        print(f"  余弦部分: B = {B_fit:.6f}, omega = {omega_fit:.6f}, phi = {phi_fit:.6f}")
        print(f"对应周期: T = {2*np.pi/omega_fit:.6f}")
        
        # 计算导出量
        e_val = B_fit / omega_fit / dini
        delta_vr_val = B_fit * np.sin(phi_fit)
        delta_e_val = -B_fit * np.cos(phi_fit) / (omega_fit * dini) * (1 - e_val)
        
        if pcov_full is not None:
            e_err, delta_vr_err, delta_e_err = compute_derived_errors(B_fit, omega_fit, phi_fit, dini, e_val, pcov_full)
            print(f"eccentricity e = {e_val:.6f} ± {e_err:.6e}")
            print(f"delta vr = {delta_vr_val:.6f} ± {delta_vr_err:.6e}")
            print(f"delta e = {delta_e_val:.6f} ± {delta_e_err:.6e}")
        else:
            print(f"eccentricity e = {e_val:.6f}")
            print(f"delta vr = {delta_vr_val:.6f}")
            print(f"delta e = {delta_e_val:.6f}")
        
        power_fit = power_law_model(time, a1_fit, a2_fit, T_fit)
        cosine_fit = cosine_model_simple(time, B_fit, omega_fit, phi_fit)
        full_fit = full_model(time, a1_fit, a2_fit, T_fit, B_fit, omega_fit, phi_fit, C_fit)
        stage_status['full_model'] = True
        
    except Exception as e:
        print(f"[ERROR step 3] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        print("[WARN] Full fit failed. Plotting fallback parameters; this window is NOT a successful batch result.")
        pcov_full = None
        average_separation_err = None
        phi_fit = np.mod(phi_fit + np.pi, 2*np.pi) - np.pi
        power_fit = power_law_model(time, a1_fit, a2_fit, T_fit)
        cosine_fit = cosine_model_simple(time, B_fit, omega_fit, phi_fit)
        full_fit = C_fit + power_fit + cosine_fit
        e_val = B_fit / omega_fit / dini
        delta_vr_val = B_fit * np.sin(phi_fit)
        delta_e_val = -B_fit * np.cos(phi_fit) / (omega_fit * dini) * (1 - e_val)
        e_err = delta_vr_err = delta_e_err = None
    
    # ========== 完整拟合结果 ==========
    final_residuals = separation - full_fit
    
    ss_res = np.sum(final_residuals**2)
    ss_tot = np.sum((separation - np.mean(separation))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    if not np.isfinite(r_squared):
        print("[WARN] R_squared is undefined/non-finite; check constant data or non-finite model output.")
    for name, error in [('e_err', e_err), ('delta_vr_err', delta_vr_err), ('delta_e_err', delta_e_err)]:
        if error is None or not np.isfinite(error):
            print(f"[WARN] {name} is unavailable/non-finite; inspect the covariance and fit status above.")
    
    print(f"\n=== 拟合结果 ===")
    print(f"R² = {r_squared:.6f}")
    print(f"残差均方根: {np.sqrt(np.mean(final_residuals**2)):.6e}")
    
    # ========== 可视化（5行1列）==========
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    
    # 第一张图：原始数据和四步拟合结果
    ax1 = axes[0]
    ax1.scatter(time_all, separation_all, s=3, alpha=0.3, color='gray', label='All data')
    ax1.scatter(time, separation, s=5, alpha=0.7, color='blue', label='Fitted data')
    ax1.plot(time, full_fit, 'r-', label='Step 4: Full model fit', linewidth=2)
    ax1.plot(time, power_fit + C_fit, 'g--',
             label='Step 1: Power law fit' if MERGE else 'Fitted average separation C', linewidth=1.5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Separation')
    
    # 处理误差为 None 的情况
    if delta_vr_err is not None and delta_e_err is not None:
        ax1.set_title(f'Data and All Fits (fitted range: [{t_min:.2f}, {t_max:.2f}])\n delta vr = {delta_vr_val:.6f} ± {delta_vr_err:.6e}, delta e = {delta_e_val:.6f} ± {delta_e_err:.6e}')
    else:
        ax1.set_title(f'Data and All Fits (fitted range: [{t_min:.2f}, {t_max:.2f}])\n delta vr = {delta_vr_val:.6f}, delta e = {delta_e_val:.6f}')
    
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # 第二张图：第一步残差 + 简单余弦拟合（第二步）
    ax2 = axes[1]
    
    residuals_step1 = separation - power_fit - C_guess
    cosine_simple_fit = cosine_model_simple(time, B_simple, omega_simple, phi_simple)
    ax2.scatter(time, residuals_step1, s=5, alpha=0.5, label='Residuals after Step 1')
    ax2.plot(time, cosine_simple_fit, 'r-', label='Step 2: Simple cosine fit', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Step 2: Simple Cosine Fit to Residuals')
    ax2.set_ylim(-0.4, 0.4)
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # 第三张图：第一步残差 + 完整余弦拟合（第三步）
    ax3 = axes[2]
    ax3.scatter(time, separation - power_fit - C_fit, s=5, alpha=0.5,
                label='Residuals after power law and fitted C')
    ax3.plot(time, cosine_fit, 'r-', label='Step 3: Full cosine fit', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Step 3: Full Cosine Fit to Residuals')
    ax3.set_ylim(-0.4, 0.4)
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    # 第四张图：第四步拟合与原始数据对比（放大拟合区域）
    ax4 = axes[3]
    ax4.scatter(time, separation, s=8, alpha=0.8, color='blue', label='Data (fitted range)')
    ax4.plot(time, full_fit, 'r--', label='Step 4: Full model fit', linewidth=2)
    #ax4.plot(time, power_fit, 'g--', label='Power law component', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Separation')
    ax4.set_title('Step 4: Full Model Fit vs Data (Fitted Range Only)')
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.3)

    # 第五张图：完整模型拟合的最终残差
    ax5 = axes[4]
    ax5.scatter(time, final_residuals, s=5, alpha=0.5)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Final Residuals')
    ax5.set_title(f'Step 4: Final Residuals (Full Model Fit)\nRMS = {np.sqrt(np.mean(final_residuals**2)):.6e}')
    ax5.set_ylim(-0.2, 0.2)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 生成带有时间范围的档名
    suffix = f"_tmin{int(t_min):04d}"
    plot_path = data_folder / f'fitting_results{suffix}.png'
    print(f"[PLOT] Saving {plot_path}", flush=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 返回拟合结果
    results = {
        'stage_status': stage_status,
        'power_law': {'a1': a1_fit, 'a2': a2_fit, 'T': T_fit},
        'cosine': {'B': B_fit, 'omega': omega_fit, 'phi': phi_fit},
        'constant_separation': C_fit,
        'average_separation': average_separation,
        'average_separation_err': average_separation_err,
        'max_derivative_time': max_derivative_time,
        'r_squared': r_squared,
        'time': time,
        'data': separation,
        'power_fit': power_fit,
        'cosine_fit': cosine_fit,
        'full_fit': full_fit,
        'residuals': final_residuals,
        'pcov_full': pcov_full,
        'dini': dini,
        'e_val': e_val,
        'delta_vr_val': delta_vr_val,
        'delta_e_val': delta_e_val,
        'e_err': e_err if pcov_full is not None else None,
        'delta_vr_err': delta_vr_err if pcov_full is not None else None,
        'delta_e_err': delta_e_err if pcov_full is not None else None
    }
    # 输出结果
    print(f"a1 = {a1_fit}")
    print(f"a2 = {a2_fit}")
    print(f"T_merge = {T_fit}")
    print(f"B = {B_fit}")
    print(f"omega = {omega_fit}")
    print(f"phi = {phi_fit}")
    if e_err is not None:
        print(f"eccentricity e = {e_val} ± {e_err}")
        print(f"delta vr = {delta_vr_val} ± {delta_vr_err}")
        print(f"delta e = {delta_e_val} ± {delta_e_err}")
    else:
        print(f"eccentricity e = {e_val}")
        print(f"delta vr = {delta_vr_val}")
        print(f"delta e = {delta_e_val}")
    print(f"R² = {r_squared}")
    return results

def save_results(results, t_min, t_max):
    """保存拟合结果到文件"""
    # 生成带有时间范围的档名
    suffix = f"_tmin{int(t_min):04d}"
    
    results_filename = data_folder / f'fit_results{suffix}.txt'
    data_filename = data_folder / f'fit_data{suffix}.txt'
    
    #with open(results_filename, 'w') as f:
    #    f.write("=== 拟合结果 ===\n\n")
    #    f.write(f"拟合数据点数: {len(results['time'])}\n")
    #    f.write(f"拟合时间范围: {results['time'][0]:.10f} 到 {results['time'][-1]:.10f}\n\n")
    #    
    #    f.write("幂律模型: a1*(T-t)^(1/4) + a2*(T-t)^(5/4)\n")
    #    f.write(f"  a1     = {results['power_law']['a1']:.10f}\n")
    #    f.write(f"  a2     = {results['power_law']['a2']:.10f}\n")
    #    f.write(f"  T      = {results['power_law']['T']:.10f}\n\n")
        
    #    f.write("余弦模型: B/omega*(1+ke*t)*cos(omega*t*(1+k_ome*t)+phi)\n")
    #    f.write(f"  B      = {results['cosine']['B']:.10f}\n")
    #    f.write(f"  omega  = {results['cosine']['omega']:.10f}\n")
    #    f.write(f"  phi    = {results['cosine']['phi']:.10f}\n")
    #    f.write(f"  ke     = {results['cosine']['ke']:.10f}\n")
    #    f.write(f"  k_ome = {results['cosine']['k_ome']:.10f}\n")
    #    f.write(f"  period = {2*np.pi/results['cosine']['omega']:.10f}\n\n")
        
    #    f.write(f"拟合优度 R² = {results['r_squared']:.10f}\n\n")

    #    f.write(f"eccentricity e = {results['e_val']:.10f} ± {results['e_err']:.10e}\n")
    #    f.write(f"delta vr = {results['delta_vr_val']:.10f} ± {results['delta_vr_err']:.10e}\n")
    #    f.write(f"delta e = {results['delta_e_val']:.10f} ± {results['delta_e_err']:.10e}\n")

    # 保存拟合数据
    #np.savetxt(data_filename, 
    #           np.column_stack([results['time'], 
    #                           results['data'],
    #                           results['full_fit'],
    #                           results['residuals']]),
    #           header='Time    Data    Full_Fit    Residuals')

# ========== 使用示例 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit separation data with stage diagnostics.")
    parser.add_argument('t_min', type=float)
    parser.add_argument('t_max', type=float)
    parser.add_argument(
        'data_file', nargs='?',
        default='output_all_vol_integral/time_separation.txt',
        help='Input file; relative paths are resolved under data_folder.',
    )
    args = parser.parse_args()
    filename = resolve_data_path(args.data_file)
    t_min, t_max = args.t_min, args.t_max
    m0 = 1.4
    print(f"[RUN] Script: {Path(__file__).resolve()}", flush=True)
    print(f"[RUN] Data/output directory: {data_folder}")
    try:
        results = fit_data(filename, t_min=t_min, t_max=t_max, m=m0)
        save_results(results, t_min, t_max)
        
        suffix = f"_tmin{int(t_min):04d}"
        print(f"图表已保存为 '{data_folder / f'fitting_results{suffix}.png'}'")
        print("[OUTPUT] Parameter/data text saving is disabled in save_results(); only the plot was saved.")
        print_batch_result(results)
        if not results['stage_status']['full_model']:
            sys.exit(2)
        
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
