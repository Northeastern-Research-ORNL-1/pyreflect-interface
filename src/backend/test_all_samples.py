#!/usr/bin/env python
"""
Benchmark: run inference on all 7 ORNL samples and compare R² against Krishna's values.

Usage:
    cd src/backend
    uv run python test_all_samples.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path("data/pretrained/trained_model.pth")
NORM_PATH = Path("data/pretrained/normalization_stat.npy")
CURVES_DIR = Path("data/curves")

SAMPLES = [194438, 194446, 194455, 194463, 194471, 194479, 194487]

KRISHNA_R2_SLD = {
    194438: 0.9709, 194446: 0.9673, 194455: 0.9540,
    194463: 0.9511, 194471: 0.9335, 194479: 0.9314, 194487: 0.9114,
}
KRISHNA_R2_NR = {
    194438: 0.9667, 194446: 0.9572, 194455: 0.9456,
    194463: 0.9332, 194471: 0.9130, 194479: 0.9235, 194487: 0.9296,
}

# ---------------------------------------------------------------------------
# Check files
# ---------------------------------------------------------------------------
missing = []
for sid in SAMPLES:
    nr = CURVES_DIR / f"np_out_REFL_{sid}_combined_data_auto.npy"
    sld = CURVES_DIR / f"sld_REF_L_{sid}.npy"
    if not nr.exists():
        missing.append(str(nr))
    if not sld.exists():
        missing.append(str(sld))
if not MODEL_PATH.exists():
    missing.append(str(MODEL_PATH))
if not NORM_PATH.exists():
    missing.append(str(NORM_PATH))
if missing:
    print("MISSING FILES:")
    for m in missing:
        print(f"  {m}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load model and norm stats
# ---------------------------------------------------------------------------
from pyreflect.models.cnn import CNN
from pyreflect.pipelines.reflectivity_pipeline import load_normalization_stat
from pyreflect.pipelines.helper import reverse_y_order

norm_stats = load_normalization_stat(str(NORM_PATH))
nr_stats = norm_stats["nr"]
sld_stats = norm_stats["sld"]

# Load model — layers=6, dropout=0.0873 from settings
model = CNN(layers=6, dropout_prob=0.0873)
sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(sd)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params\n")


# ---------------------------------------------------------------------------
# Helpers (same logic as _real_nr_sld_infer_core)
# ---------------------------------------------------------------------------

def run_inference(nr_path: Path) -> np.ndarray:
    """Returns denormalized predicted SLD as (2, 900) — no postprocessing."""
    nr_raw = np.load(nr_path, allow_pickle=True)
    # Handle (308, 2) → transpose to (2, 308), then add batch dim
    if nr_raw.ndim == 2 and nr_raw.shape[1] == 2:
        nr_raw = nr_raw.T
    if nr_raw.ndim == 2:
        nr_raw = nr_raw[np.newaxis, :, :]
    nr_arr = nr_raw.copy().astype(np.float64)

    # Normalize: log10 on y, then min-max
    nr_arr[:, 1, :] = np.log10(np.clip(nr_arr[:, 1, :], 1e-8, None))
    nr_y_norm = (nr_arr[:, 1, :] - nr_stats["y"]["min"]) / (nr_stats["y"]["max"] - nr_stats["y"]["min"])
    nr_input = torch.tensor(nr_y_norm[:, np.newaxis, :], dtype=torch.float32)

    # Forward pass — skip sigmoid (same as _real_nr_sld_infer_core)
    with torch.no_grad():
        x = nr_input
        for layer in model.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = model.linear1(x)
        x = x.reshape(-1, 2, 900)
        raw_output = x.cpu().numpy()

    # Denormalize
    pred_z = raw_output[0, 0, :] * (sld_stats["x"]["max"] - sld_stats["x"]["min"]) + sld_stats["x"]["min"]
    pred_y = raw_output[0, 1, :] * (sld_stats["y"]["max"] - sld_stats["y"]["min"]) + sld_stats["y"]["min"]
    return np.stack([pred_z, pred_y])


def preprocess_gt_sld(gt_path: Path, target_len: int = 900) -> np.ndarray:
    """Krishna's GT preprocessing: transpose, interp to 900, reverse_y_order, zero-shift z."""
    gt = np.load(gt_path, allow_pickle=True)
    if gt.ndim == 2 and gt.shape[1] == 2:
        gt = gt.T
    L = gt.shape[1]
    x_old = np.linspace(0, 1, L)
    x_new = np.linspace(0, 1, target_len)
    gt_interp = np.array([np.interp(x_new, x_old, gt[ch]) for ch in range(2)])
    gt_interp = reverse_y_order(gt_interp)
    gt_interp[0] -= gt_interp[0].min()
    return gt_interp


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def r2_overlap_masked(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    """R² using only GT points within predicted z-range (no flat-extrapolation)."""
    gt_z, gt_y = gt_arr[0], gt_arr[1]
    pred_z_min, pred_z_max = float(pred_arr[0].min()), float(pred_arr[0].max())
    pred_y_aligned = np.interp(gt_z, pred_arr[0], pred_arr[1])
    mask = (gt_z >= pred_z_min) & (gt_z <= pred_z_max)
    return r2_score(gt_y[mask], pred_y_aligned[mask])


def r2_krishna(gt_arr: np.ndarray, pred_arr: np.ndarray) -> float:
    """Krishna's method: interp pred onto GT z-grid, R² on ALL points (including extrapolated)."""
    pred_y_aligned = np.interp(gt_arr[0], pred_arr[0], pred_arr[1])
    return r2_score(gt_arr[1], pred_y_aligned)


# ---------------------------------------------------------------------------
# Run all samples
# ---------------------------------------------------------------------------
results = []

for sid in SAMPLES:
    nr_path = CURVES_DIR / f"np_out_REFL_{sid}_combined_data_auto.npy"
    gt_path = CURVES_DIR / f"sld_REF_L_{sid}.npy"

    pred = run_inference(nr_path)
    gt = preprocess_gt_sld(gt_path)

    r2_k = r2_krishna(gt, pred)       # Krishna's method (all points)
    r2_m = r2_overlap_masked(gt, pred)  # Overlap-masked method

    krishna_ref = KRISHNA_R2_SLD.get(sid, 0.0)
    delta_k = r2_k - krishna_ref
    delta_m = r2_m - krishna_ref

    results.append((sid, r2_k, r2_m, krishna_ref, delta_k, delta_m))

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"{'Sample':<10} {'R²(krishna)':>12} {'R²(masked)':>12} {'Krishna ref':>12} {'d(krishna)':>11} {'d(masked)':>10} {'Pass?':>6}")
print("-" * 80)

pass_count_k = 0
pass_count_m = 0
for sid, r2_k, r2_m, ref, dk, dm in results:
    pass_k = abs(dk) < 0.01
    pass_m = abs(dm) < 0.01
    if pass_k:
        pass_count_k += 1
    if pass_m:
        pass_count_m += 1
    flag_k = "PASS" if pass_k else "FAIL"
    flag_m = "PASS" if pass_m else "FAIL"
    print(f"{sid:<10} {r2_k:>12.4f} {r2_m:>12.4f} {ref:>12.4f} {dk:>+11.4f} {dm:>+10.4f}   {flag_k}/{flag_m}")

print("-" * 80)
avg_r2_k = np.mean([r[1] for r in results])
avg_r2_m = np.mean([r[2] for r in results])
avg_ref = np.mean([r[3] for r in results])
avg_dk = np.mean([abs(r[4]) for r in results])
avg_dm = np.mean([abs(r[5]) for r in results])
print(f"{'AVERAGE':<10} {avg_r2_k:>12.4f} {avg_r2_m:>12.4f} {avg_ref:>12.4f} {avg_dk:>11.4f} {avg_dm:>10.4f}   {pass_count_k}/{len(results)} / {pass_count_m}/{len(results)}")
print()
print("Column legend:")
print("  R²(krishna)  = Krishna's method: interp pred onto GT z-grid, all points")
print("  R²(masked)   = Overlap-masked: exclude GT points outside pred z-range")
print("  d(krishna)   = R²(krishna) - Krishna's reference value")
print("  d(masked)    = R²(masked)  - Krishna's reference value")
print("  Pass?        = krishna/masked (|d| < 0.01)")
