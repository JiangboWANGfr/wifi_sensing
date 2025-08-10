#!/usr/bin/env python3
from utils import plots_utility as PU  # 只用这个库出图
import pickle
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# 如果你的 plots_utility 在 code/src/utils 下：
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "code" / "src"))

# ========== 配置 ==========
ROOT = Path(".")         # 项目根
SUBDIR = "S1a"             # 数据子目录
ACT = "E"               # 活动: E/L/W/R/J1/J2/S/H/C
STREAM = 0                 # 流索引
TC_MS = 6.0               # 采样周期(ms)
N = 31                # STFT窗长(帧)
ND = 100               # 多普勒bins
WIN_T = 340               # 约2秒窗口(=340帧)
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ========== 路径 ==========
RAW_MAT = ROOT / "datasets" / SUBDIR / f"{SUBDIR}_{ACT}.mat"
PRE_TXT = ROOT / "results" / "phase_processing" / \
    "signal_preprocessing" / f"signal_{SUBDIR}_{ACT}.txt"
PRO_MAT = ROOT / "results" / "processed_phase" / \
    SUBDIR / f"{SUBDIR}_{ACT}_stream_{STREAM}.mat"
tag = f"{SUBDIR}_{ACT}_s{STREAM}"

# ========== 读取 ==========


def _find_first(mat):
    if "csi_buff" in mat:
        return mat["csi_buff"]
    for k, v in mat.items():
        if not k.startswith("__") and hasattr(v, "ndim") and v.ndim in (2, 3):
            return v
    raise RuntimeError("No CSI-like array in .mat")


def to_subc_frames(arr):
    # 统一成 (subcarriers, frames)
    if arr.ndim == 3:
        if arr.shape[1] >= arr.shape[0]:
            arr = arr.mean(axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
            arr = arr.T
        else:
            arr = arr.mean(axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
        return arr
    return arr.T if arr.shape[0] < arr.shape[1] else arr


# RAW
H_raw = to_subc_frames(_find_first(loadmat(str(RAW_MAT))))
# PRE (预处理的pickle：形状通常为 (subc, frames, streams))
with open(PRE_TXT, "rb") as fp:
    pre_all = pickle.load(fp)
# (subc, frames)
H_pre = pre_all[:, :, STREAM] if pre_all.ndim == 3 else pre_all
# PRO
H_pro = to_subc_frames(_find_first(loadmat(str(PRO_MAT))))

# ========== 简化微多普勒 ==========


def doppler_spectrogram(H, n=N, nd=ND, win_frames=WIN_T):
    subc, T = H.shape
    H = np.where(np.isfinite(H), H, 0)
    if not np.iscomplexobj(H):
        H = H.astype(np.complex128)
    profiles = []
    for t0 in range(0, T - n + 1):
        W = H[:, t0:t0+n]
        F = np.fft.fft(W, n=nd, axis=1)
        profiles.append(np.mean(np.abs(F), axis=0))  # (nd,)
    if not profiles:
        return np.zeros((nd, 0))
    S = np.stack(profiles, axis=1)  # (nd, time)
    if S.shape[1] > win_frames:
        st = (S.shape[1] - win_frames)//2
        S = S[:, st:st+win_frames]
    return S.T  # 供 plots_utility: (time, bins)


D_raw = doppler_spectrogram(H_raw)
D_pre = doppler_spectrogram(H_pre)
D_pro = doppler_spectrogram(H_pro)

# ========== 出图（全部只用 plots_utility）==========

# 1) 三阶段 |H| 热图（横排/竖排你任选一个）
PU.plt_amplitude(np.abs(H_raw), str(PLOTS / f"{tag}_RAW_amp.png"))
PU.plt_amplitude(np.abs(H_pre), str(PLOTS / f"{tag}_PRE_amp.png"))
PU.plt_amplitude(np.abs(H_pro), str(PLOTS / f"{tag}_PRO_amp.png"))

# 2) 相位对比（最有意义的是 PRE vs PRO）
PU.plt_phase(np.angle(H_pre), np.angle(H_pro), str(
    PLOTS / f"{tag}_phase_PRE_vs_PRO.png"))
# 也可加 RAW vs PRO 的感觉
PU.plt_phase(np.angle(H_raw), np.angle(H_pro), str(
    PLOTS / f"{tag}_phase_RAW_vs_PRO.png"))

# 3) 三阶段 2 秒微多普勒（单天线单图版）
# 该函数要求输入是“按天线列表”的 time×bins 矩阵；我们就传 [D] 并用 antenna=0
PU.plt_doppler_activity_single([D_raw], antenna=0, sliding_lenght=N,  delta_v=1.0,
                               name_plot=str(PLOTS / f"{tag}_DOPPLER_RAW.png"))
PU.plt_doppler_activity_single([D_pre], antenna=0, sliding_lenght=N,  delta_v=1.0,
                               name_plot=str(PLOTS / f"{tag}_DOPPLER_PRE.png"))
PU.plt_doppler_activity_single([D_pro], antenna=0, sliding_lenght=N,  delta_v=1.0,
                               name_plot=str(PLOTS / f"{tag}_DOPPLER_PRO.png"))

print("Saved to:", PLOTS)
