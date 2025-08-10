#!/usr/bin/env python3
"""
Paper-style plots for SHARP (Fig.1 & Fig.2 alike), with robust loaders that
handle different .mat layouts, including real/imag split as (..., 2).

What it draws
-------------
Fig.1-style (default):
  Left  = empty room (E)
  Right = a person activity (e.g., W/R/J1/J2/...)
  Rows  = amplitude (dB), raw phase (unwrapped), sanitized phase (unwrapped)

Fig.2-style (optional):
  Same empty room (E) in two different subdirs/days; amplitude only.

Inputs it expects
-----------------
RAW  : datasets/<subdir>/<subdir>_<ACT>.mat
PRO  : results/processed_phase/<subdir>/<subdir>_<ACT>_stream_<i>.mat

Usage examples (run from project root)
--------------------------------------
python plot_phase_paper_style.py --subdir S1a --empty_act E --person_act W --stream 0 --tc_ms 6 --save
python plot_phase_paper_style.py --subdir S1a --empty_act E --person_act R --stream 2 --tc_ms 6 --save
# Fig.2-style (two days of the same empty room)
python plot_phase_paper_style.py --subdir S1a --empty_act E --day2_subdir S1b --tc_ms 6 --save
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---------------- Utilities: shape & complex handling ----------------

def _debug_shape(tag, arr):
    try:
        print(f"[DEBUG] {tag}: shape={np.shape(arr)}, dtype={getattr(arr, 'dtype', type(arr))}")
    except Exception:
        pass


def _find_first_csi_array(mat_dict):
    """Return the first 2D/3D ndarray likely to be CSI. Prefer 'csi_buff'."""
    if 'csi_buff' in mat_dict and hasattr(mat_dict['csi_buff'], 'ndim'):
        arr = mat_dict['csi_buff']
        if arr.ndim in (2, 3):
            return arr
    for k, v in mat_dict.items():
        if k.startswith('__'):
            continue
        if hasattr(v, 'ndim') and v.ndim in (2, 3):
            return v
    raise RuntimeError('No 2D/3D CSI-like array in .mat file')


def _to_complex_if_split_real_imag(arr):
    """If last dim==2 and real dtype, assume [real, imag] and merge to complex."""
    if isinstance(arr, np.ndarray) and arr.ndim >= 3 and arr.shape[-1] == 2 and np.isrealobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    return arr


def _standardize_2d(arr2d):
    """
    标准化为 (subcarriers, frames)。
    - 优先把“接近 242/245/256 的维度”当作 subcarriers 维
    - 否则退化为“较小的一维是 subcarriers”
    """
    if arr2d.ndim != 2:
        raise ValueError(
            f"Expected 2D after squeeze, got {arr2d.ndim}D with shape {arr2d.shape}")

    h, w = arr2d.shape
    subc_candidates = {242, 243, 244, 245, 256}

    if w in subc_candidates:
        # 当前是 (frames, subc) → 需要转置为 (subc, frames)
        return arr2d.T
    if h in subc_candidates:
        # 已经是 (subc, frames)
        return arr2d

    # 容错：一般 subcarriers < frames（例如 245 vs 5809）
    return arr2d if h < w else arr2d.T



def _reduce_3d(arr3d):
    """Reduce 3D arrays to 2D (frames, subc) or (subc, frames).
    - If last dim is streams/etc, take mean along it (safer than picking index 0).
    - If shape looks like (frames, subc, 1), squeeze it.
    """
    if arr3d.shape[-1] != 2 or not np.isrealobj(arr3d):
        # not real/imag split; assume third dim is channels/streams and average
        arr2d = arr3d.mean(axis=2)
    else:
        # real/imag handled before by _to_complex_if_split_real_imag; this branch rarely used
        arr2d = arr3d[..., 0]
    return arr2d


def _drop_center_dc_subcarriers(H):
    # H: (subc, frames). 若 subc=245，去掉中间3个得到242
    if H.shape[0] == 245:
        mid = H.shape[0] // 2  # 122
        keep = np.r_[0:mid-1, mid+2:H.shape[0]]  # 去掉 [mid-1, mid, mid+1]
        H = H[keep, :]
    return H



def _load_mat_generic_as_subc_time(path_mat: Path) -> np.ndarray:
    """Load .mat, merge real/imag if needed, reduce to 2D, standardize to (subc, frames)."""
    md = loadmat(str(path_mat))
    arr = _find_first_csi_array(md)
    arr = _to_complex_if_split_real_imag(arr)
    _debug_shape(f"load {path_mat.name} raw", arr)
    if arr.ndim == 3:
        arr = _reduce_3d(arr)
    arr = _standardize_2d(arr)
    arr = _drop_center_dc_subcarriers(arr)
    _debug_shape(f"load {path_mat.name} std2d", arr)
    
    return arr

# ---------------- Amplitude/phase helpers ----------------

def amplitude_db(H: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(H), 1e-8))


def unwrap_phase_time(H: np.ndarray) -> np.ndarray:
    """Unwrap along time (axis=1) and remove per-subcarrier median to center baseline."""
    ph = np.unwrap(np.angle(H), axis=1)
    ph = ph - np.median(ph, axis=1, keepdims=True)
    return ph


def show_heat(A: np.ndarray, title: str, tc: float, is_phase=False, vmin=None, vmax=None):
    """imshow helper with time in seconds and subcarrier index on y-axis."""
    # Expect A as (subc, frames)
    T = A.shape[1] * tc
    extent = [0, T, -122, 122]  # paper-style y-axis
    cmap = 'twilight' if is_phase else 'viridis'
    im = plt.imshow(A, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('time [s]')
    plt.ylabel('sub-channel')
    cbar = plt.colorbar(im)
    cbar.set_label('phase [rad]' if is_phase else 'power [dB]')

# ---------------- Plotters ----------------

def plot_fig1_style(root: Path, subdir: str, empty_act: str, person_act: str, stream: int, tc: float, save: bool):
    """Left: empty (E); Right: person activity. Rows: amplitude, raw phase, sanitized phase."""
    plots = root / 'results' / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    # RAW data
    raw_empty  = _load_mat_generic_as_subc_time(root / 'datasets' / subdir / f'{subdir}_{empty_act}.mat')
    raw_person = _load_mat_generic_as_subc_time(root / 'datasets' / subdir / f'{subdir}_{person_act}.mat')

    # PRO (sanitized) data
    pro_empty_path  = root / 'results' / 'phase_processing' / 'processed_phase' / subdir / f'{subdir}_{empty_act}_stream_{stream}.mat'
    pro_person_path = root / 'results' / 'phase_processing' / 'processed_phase' / subdir / f'{subdir}_{person_act}_stream_{stream}.mat'
    pro_empty  = _load_mat_generic_as_subc_time(pro_empty_path)
    pro_person = _load_mat_generic_as_subc_time(pro_person_path)

    # Amplitude dB
    A_empty  = amplitude_db(raw_empty)
    A_person = amplitude_db(raw_person)

    # Phases (unwrap along time + per-subcarrier baseline removal)
    P_raw_empty  = unwrap_phase_time(raw_empty)
    P_raw_person = unwrap_phase_time(raw_person)
    P_pro_empty  = unwrap_phase_time(pro_empty)
    P_pro_person = unwrap_phase_time(pro_person)

    # Plot 3x2
    fig = plt.figure(figsize=(10, 6))
    # Left column: empty
    plt.subplot(3, 2, 1); show_heat(A_empty, 'amplitude', tc, is_phase=False, vmin=-40, vmax=0)
    plt.subplot(3, 2, 3); show_heat(P_raw_empty, 'raw phase', tc, is_phase=True, vmin=-np.pi, vmax=np.pi)
    plt.subplot(3, 2, 5); show_heat(P_pro_empty, 'sanitized phase', tc, is_phase=True, vmin=-np.pi, vmax=np.pi)
    # Right column: person
    plt.subplot(3, 2, 2); show_heat(A_person, 'amplitude', tc, is_phase=False, vmin=-40, vmax=0)
    plt.subplot(3, 2, 4); show_heat(P_raw_person, 'raw phase', tc, is_phase=True, vmin=-np.pi, vmax=np.pi)
    plt.subplot(3, 2, 6); show_heat(P_pro_person, 'sanitized phase', tc, is_phase=True, vmin=-np.pi, vmax=np.pi)
    plt.tight_layout()

    tag = f'{subdir}_s{stream}_{empty_act}_vs_{person_act}'
    if save:
        out = plots / f'{tag}_fig1_style.png'
        fig.savefig(out, dpi=150)
        print('Saved:', out)
    plt.show()


essential_empty_msg = (
    "Fig.2-style requires two subdirs (days) that both contain RAW empty-room .mat files."
)

def plot_fig2_style(root: Path, day1_subdir: str, day2_subdir: str, empty_act: str, tc: float, save: bool):
    """Same empty-room activity E on two different subdirs (days), amplitude only."""
    plots = root / 'results' / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    path1 = root / 'datasets' / day1_subdir / f'{day1_subdir}_{empty_act}.mat'
    path2 = root / 'datasets' / day2_subdir / f'{day2_subdir}_{empty_act}.mat'
    if not path1.exists() or not path2.exists():
        raise FileNotFoundError(essential_empty_msg + f" Missing: {[p for p in [path1, path2] if not p.exists()]}")

    H_day1 = _load_mat_generic_as_subc_time(path1)
    H_day2 = _load_mat_generic_as_subc_time(path2)

    A1 = amplitude_db(H_day1)
    A2 = amplitude_db(H_day2)

    fig = plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1); show_heat(A1, f'{day1_subdir} amplitude', tc, is_phase=False, vmin=-40, vmax=0)
    plt.subplot(1, 2, 2); show_heat(A2, f'{day2_subdir} amplitude', tc, is_phase=False, vmin=-40, vmax=0)
    plt.tight_layout()

    tag = f'{day1_subdir}_vs_{day2_subdir}_{empty_act}'
    if save:
        out = plots / f'{tag}_fig2_style.png'
        fig.savefig(out, dpi=150)
        print('Saved:', out)
    plt.show()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description='Paper-style plots for SHARP: amplitude & (unwrapped) phases')
    ap.add_argument('--root', type=str, default='.', help='Project root with datasets/ and results/')
    ap.add_argument('--subdir', type=str, required=True, help='Dataset subdir (e.g., S1a)')
    ap.add_argument('--empty_act', type=str, default='E', help='Empty-room activity code (default: E)')
    ap.add_argument('--person_act', type=str, default='W', help='Person activity code (e.g., W/R/J1...)')
    ap.add_argument('--stream', type=int, default=0, help='Stream index for processed_phase (0..3)')
    ap.add_argument('--tc_ms', type=float, default=6.0, help='Sampling period in ms (default 6)')
    ap.add_argument('--day2_subdir', type=str, default='', help='If set, also produce Fig.2-style day1 vs day2 amplitude (empty room)')
    ap.add_argument('--save', action='store_true', help='Save figures to results/plots')
    args = ap.parse_args()

    root = Path(args.root)
    tc = float(args.tc_ms) / 1000.0

    # Fig.1-style
    plot_fig1_style(root, args.subdir, args.empty_act, args.person_act, args.stream, tc, args.save)

    # Fig.2-style (optional)
    if args.day2_subdir:
        plot_fig2_style(root, args.subdir, args.day2_subdir, args.empty_act, tc, args.save)


if __name__ == '__main__':
    main()
