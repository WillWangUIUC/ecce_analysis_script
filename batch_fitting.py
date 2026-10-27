import subprocess
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
import time as clock
import traceback

from fitting_script import data_folder, resolve_data_path


SCRIPT_DIR = Path(__file__).resolve().parent


def show_output(label, output):
    """TimeoutExpired may return bytes even when text=True."""
    if isinstance(output, bytes):
        output = output.decode('utf-8', errors='replace')
    if output and output.strip():
        print(f"\n--- {label} ---", flush=True)
        print(output.rstrip(), flush=True)

def run_fitting_for_range(t_min, t_max, data_file, script_name="fitting_script.py"):
    """Run one window, display diagnostics, and validate its structured result."""
    script_path = Path(script_name).expanduser()
    if not script_path.is_absolute():
        script_path = SCRIPT_DIR / script_path
    script_path = script_path.resolve()
    data_path = resolve_data_path(data_file)
    cmd = [
        sys.executable,
        '-u',
        str(script_path),
        str(t_min),
        str(t_max),
        str(data_path),
    ]
    print(f"[BATCH] Window [{t_min}, {t_max}]", flush=True)
    print(f"[BATCH] Command: {subprocess.list2cmdline(cmd)}", flush=True)
    print(f"[BATCH] Data/output directory: {data_folder}", flush=True)
    print("[BATCH] Child output is captured and displayed on completion or the 300s timeout.", flush=True)
    started = clock.perf_counter()
    try:
        if not script_path.is_file():
            raise FileNotFoundError(f"Fitting script does not exist: {script_path}")
        if not data_path.is_file():
            raise FileNotFoundError(f"Input data does not exist: {data_path}")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            cmd, cwd=data_folder, env=env,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
        )
        print(f"[BATCH] Child exited with code {result.returncode} after {clock.perf_counter() - started:.3f}s")
        show_output('fitting_script stdout', result.stdout)
        show_output('fitting_script stderr / warnings / traceback', result.stderr)
        if result.returncode != 0:
            print(
                f"[ERROR] Window [{t_min}, {t_max}]: fitting process failed "
                f"(exit code {result.returncode}). See the stage error and traceback above."
            )
            return None

        prefix = 'FIT_RESULT_JSON: '
        summaries = [line[len(prefix):] for line in result.stdout.splitlines() if line.startswith(prefix)]
        if not summaries:
            print(
                f"[ERROR parse] Window [{t_min}, {t_max}]: no FIT_RESULT_JSON summary was emitted. "
                "The fitter may have stopped before its final summary or be an older script. "
                f"Check the output above and script path: {script_path}"
            )
            return None
        try:
            payload = json.loads(summaries[-1])
        except json.JSONDecodeError as exc:
            print(f"[ERROR parse] Invalid summary JSON at column {exc.colno}: {exc.msg}")
            print(f"[ERROR parse] Offending summary: {summaries[-1]!r}")
            return None
        if not isinstance(payload, dict):
            print(f"[ERROR parse] Expected a result object, got {payload!r}")
            return None
        if payload.get('fit_status') != 'ok':
            print(f"[ERROR fit] Fit status={payload.get('fit_status')!r}, stages={payload.get('stage_status')!r}")
            return None

        required = ['a1', 'a2', 'T_merge', 'B', 'omega', 'phi', 'e', 'delta_vr', 'delta_e', 'R_squared']
        params = {'t_min': t_min}
        invalid = []
        for key in required:
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                invalid.append(f"{key}={value!r}")
            else:
                params[key] = float(value)
        if invalid:
            print(f"[ERROR parse] Window [{t_min}, {t_max}]: missing/invalid final fields: {', '.join(invalid)}")
            print(f"[ERROR parse] Received keys: {', '.join(sorted(payload))}")
            return None

        for key in ['e_err', 'delta_vr_err', 'delta_e_err']:
            value = payload.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                params[key] = np.nan
                print(f"[WARN] {key}={value!r}: storing NaN; inspect covariance warnings above.")
            else:
                params[key] = float(value)
        print(f"[BATCH] Parsed final result successfully; stage status={payload.get('stage_status')}")
        return params

    except subprocess.TimeoutExpired as exc:
        print(f"[ERROR timeout] Window [{t_min}, {t_max}] exceeded {exc.timeout}s; child was stopped.")
        show_output('partial stdout before timeout', exc.stdout)
        show_output('partial stderr before timeout', exc.stderr)
        return None
    except Exception as e:
        print(f"[ERROR batch] Window [{t_min}, {t_max}]: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None

def batch_fitting(data_file, window_size=2000, step=25, script_name="fitting_script.py"):
    """
    批次擬合主函數
    """
    
    # 讀取數據以獲取時間範圍
    if not np.isfinite(window_size) or not np.isfinite(step) or window_size <= 0 or step <= 0:
        raise ValueError(f"window_size and step must be finite and positive: {window_size}, {step}")
    data_file = resolve_data_path(data_file)
    print(f"[BATCH] Reading {data_file}", flush=True)
    data = np.loadtxt(data_file, ndmin=2)
    if data.shape[0] == 0 or data.shape[1] < 2 or not np.all(np.isfinite(data[:, :2])):
        raise ValueError(f"Expected finite time/separation columns; got shape {data.shape}.")
    time = data[:, 0]
    t_min_data = 1000
    t_max_data = time[-1]
    
    print(f"數據時間範圍: {t_min_data:.2f} 到 {t_max_data:.2f}")
    print(f"窗口大小: {window_size}, 步長: {step}")
    
    # 生成所有擬合範圍
    results = []
    failed_windows = []
    attempted = 0
    current_t_min = t_min_data
    
    while current_t_min + window_size <= t_max_data:
        t_max = current_t_min + window_size
        attempted += 1
        
        print(f"\n處理: t_min={current_t_min:.2f}, t_max={t_max:.2f} "
              f"(窗口: {attempted}, 成功: {len(results)}, 失败: {len(failed_windows)})", flush=True)
        
        result = run_fitting_for_range(current_t_min, t_max, data_file, script_name)
        if result is not None:
            results.append(result)
            print(f"  成功: e={result['e']:.6f}, R²={result['R_squared']:.6f}")
        else:
            failed_windows.append((current_t_min, t_max))
            print(f"[BATCH] Skipping failed window [{current_t_min}, {t_max}]. See diagnostics above.")
        
        current_t_min += step
    
    # 保存所有结果
    if results:
        output_file = data_folder / f"batch_fitting_results_w{window_size}_s{step}.txt"
        
        # 保存完整参数
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# t_min    a1    a2    T_merge    B    omega    phi    e    e_err    delta_vr    delta_vr_err    delta_e    delta_e_err    R_squared\n")
            for r in results:
                f.write(f"{r['t_min']:.10f}    ")
                f.write(f"{r['a1']:.8e}    " if r['a1'] is not None else "NaN    ")
                f.write(f"{r['a2']:.8e}    " if r['a2'] is not None else "NaN    ")
                f.write(f"{r['T_merge']:.8f}    " if r['T_merge'] is not None else "NaN    ")
                f.write(f"{r['B']:.8e}    " if r['B'] is not None else "NaN    ")
                f.write(f"{r['omega']:.8e}    " if r['omega'] is not None else "NaN    ")
                f.write(f"{r['phi']:.6f}    " if r['phi'] is not None else "NaN    ")
                f.write(f"{r['e']:.8e}    {r['e_err']:.8e}    ")
                f.write(f"{r['delta_vr']:.8e}    {r['delta_vr_err']:.8e}    ")
                f.write(f"{r['delta_e']:.8e}    {r['delta_e_err']:.8e}    ")
                f.write(f"{r['R_squared']:.8f}\n")
        
        print(f"\n完成！共 {len(results)} 組擬合結果")
        print(f"結果已保存到: {output_file}")
        
        # 输出摘要
        print(f"\n=== 結果摘要 ===")
        print(f"t_min 範圍: {results[0]['t_min']:.2f} 到 {results[-1]['t_min']:.2f}")
        e_vals = [r['e'] for r in results]
        print(f"e 範圍: {np.min(e_vals):.6f} 到 {np.max(e_vals):.6f}")
        r2_vals = [r['R_squared'] for r in results]
        print(f"R² 範圍: {np.min(r2_vals):.6f} 到 {np.max(r2_vals):.6f}")
        print(f"平均 R²: {np.mean(r2_vals):.6f}")
        
    else:
        print("沒有成功完成任何擬合！")
    print(f"[BATCH SUMMARY] Attempted={attempted}, successful={len(results)}, failed={len(failed_windows)}")
    if not attempted:
        print(f"[ERROR] No complete window fits: start={t_min_data}, end={t_max_data}, window_size={window_size}.")
    if failed_windows:
        print("[BATCH SUMMARY] Failed intervals: " + ', '.join(f'[{lo}, {hi}]' for lo, hi in failed_windows))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch separation fits with subprocess and parsing diagnostics.")
    parser.add_argument('--data-file', default='output_all_vol_integral/time_separation.txt',
                        help='Input file; relative paths are resolved under data_folder.')
    parser.add_argument('--script', default=str(SCRIPT_DIR / 'fitting_script.py'))
    parser.add_argument('--window-size', type=int, default=2000)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--single-window', nargs=2, type=float, metavar=('T_MIN', 'T_MAX'),
                        help='Diagnose one interval instead of running the full batch.')
    args = parser.parse_args()
    try:
        if args.single_window is not None:
            result = run_fitting_for_range(*args.single_window, args.data_file, args.script)
            sys.exit(0 if result is not None else 1)
        results = batch_fitting(args.data_file, args.window_size, args.step, args.script)
        sys.exit(0 if results else 1)
    except Exception as exc:
        print(f"[ERROR batch setup/output] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
