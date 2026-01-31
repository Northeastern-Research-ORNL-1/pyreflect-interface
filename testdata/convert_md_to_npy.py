#!/usr/bin/env python3
"""
Convert ORNL reflectometry .md (markdown/plaintext) files to numpy format for pyreflect.

This script handles experimental NR data where only Q, R, dR are available
(no SLD profile - that's what the model will predict!).

Usage:
    python convert_md_to_npy.py <input.md> [--plot] [--target-length 308]
    
Example:
    python convert_md_to_npy.py ../REFL_212592_reduced_noQb.md --plot
"""

import numpy as np
from pathlib import Path
import sys
import re
import argparse


def parse_md_reflectivity(md_path: Path) -> dict:
    """
    Parse an ORNL-style reflectometry .md file containing Q, R, dR, dQ data.
    
    Expected format (after header comments):
        Q [1/Angstrom]        R                     dR                    dQ [FWHM]
        0.0083469176538132    0.9350638820680347    0.0277800865883856    0.0002323386967239
        ...
    
    Returns dict with:
        - 'q': Q values (1/Angstrom)
        - 'r': Reflectivity values
        - 'dr': Reflectivity error
        - 'dq': Q resolution (FWHM)
        - 'metadata': dict of header info
    """
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    metadata = {}
    q_values, r_values, dr_values, dq_values = [], [], [], []
    data_started = False
    
    for line in lines:
        line = line.strip()
        
        # Parse header comments
        if line.startswith('#'):
            # Extract key-value pairs from comments
            match = re.match(r'#\s*(.+?):\s*(.+)', line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                metadata[key] = value
            continue
        
        # Skip empty lines and header row
        if not line or 'Q [' in line or 'R ' in line:
            continue
        
        # Parse data lines
        parts = line.split()
        if len(parts) >= 4:
            try:
                q = float(parts[0])
                r = float(parts[1])
                dr = float(parts[2])
                dq = float(parts[3])
                
                # Basic validity checks
                if q > 0 and r > -1e-6:  # Allow very small negative from noise
                    q_values.append(q)
                    r_values.append(max(0, r))  # Clip negative reflectivity
                    dr_values.append(dr)
                    dq_values.append(dq)
                    data_started = True
            except ValueError:
                continue
    
    if not data_started:
        raise ValueError(f"No valid data found in {md_path}")
    
    return {
        'q': np.array(q_values),
        'r': np.array(r_values),
        'dr': np.array(dr_values),
        'dq': np.array(dq_values),
        'metadata': metadata,
    }


def interpolate_to_length(x: np.ndarray, y: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate data to a fixed length grid."""
    x_new = np.linspace(x.min(), x.max(), target_length)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new


def create_nr_array(data: dict, target_length: int = 308) -> np.ndarray:
    """
    Create NR array in pyreflect format: shape (1, 2, L)
    
    The pyreflect pipeline expects:
        - nr_train.npy: shape (N, 2, 308) where channels are [Q, R]
        - For inference: single curve is (1, 2, 308)
    
    Args:
        data: parsed reflectivity data
        target_length: number of Q points (pyreflect default is 308)
    
    Returns:
        np.ndarray of shape (1, 2, target_length)
    """
    q, r = interpolate_to_length(data['q'], data['r'], target_length)
    
    # Stack as (2, L) then add batch dimension -> (1, 2, L)
    nr_curve = np.stack([q, r], axis=0)
    nr_batch = np.expand_dims(nr_curve, axis=0)
    
    return nr_batch


def plot_data(data: dict, output_path: Path):
    """Plot NR curve for visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q, r, dr = data['q'], data['r'], data['dr']
    
    # Reflectivity on log scale
    ax.semilogy(q, r, 'b.-', markersize=2, linewidth=0.5, label='R(Q)')
    ax.fill_between(q, np.maximum(r-dr, 1e-10), r+dr, alpha=0.3, color='blue')
    
    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('Reflectivity', fontsize=12)
    ax.set_title(f"Neutron Reflectivity - {data['metadata'].get('Run title', 'Unknown')}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metadata annotation
    meta_text = '\n'.join([
        f"Experiment: {data['metadata'].get('Experiment IPTS-33280 Run 212592', 'N/A')[:50]}",
        f"Run time: {data['metadata'].get('Run start time', 'N/A')[:30]}",
        f"Points: {len(q)}",
        f"Q range: [{q.min():.4f}, {q.max():.4f}] Å⁻¹",
    ])
    ax.text(0.98, 0.98, meta_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Convert ORNL .md reflectivity data to .npy')
    parser.add_argument('input', type=Path, help='Input .md file')
    parser.add_argument('--plot', action='store_true', help='Generate plot')
    parser.add_argument('--target-length', type=int, default=308, 
                        help='Target Q-points (default: 308 for pyreflect)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: same as input)')
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    output_dir = args.output_dir or args.input.parent
    base_name = args.input.stem
    
    print(f"Processing: {args.input}")
    print("=" * 60)
    
    # Parse data
    data = parse_md_reflectivity(args.input)
    
    print(f"  Metadata:")
    for key, value in list(data['metadata'].items())[:5]:
        print(f"    {key}: {value[:50]}...")
    
    print(f"\n  Data statistics:")
    print(f"    Points: {len(data['q'])}")
    print(f"    Q range: [{data['q'].min():.6f}, {data['q'].max():.6f}] Å⁻¹")
    print(f"    R range: [{data['r'].min():.2e}, {data['r'].max():.4f}]")
    print(f"    Mean dR/R: {(data['dr'] / np.maximum(data['r'], 1e-10)).mean():.2%}")
    
    # Create NR array
    nr_array = create_nr_array(data, target_length=args.target_length)
    
    # Save outputs
    nr_path = output_dir / f"{base_name}_nr.npy"
    np.save(nr_path, nr_array)
    print(f"\n  Saved: {nr_path}")
    print(f"    Shape: {nr_array.shape} (N=1 sample, 2 channels [Q,R], {args.target_length} points)")
    
    # Also save raw data with error bars for reference
    raw_path = output_dir / f"{base_name}_raw.npy"
    raw_data = np.stack([data['q'], data['r'], data['dr'], data['dq']], axis=0)
    np.save(raw_path, raw_data)
    print(f"  Saved: {raw_path}")
    print(f"    Shape: {raw_data.shape} (4 channels [Q, R, dR, dQ], {len(data['q'])} original points)")
    
    if args.plot:
        plot_path = output_dir / f"{base_name}_plot.png"
        plot_data(data, plot_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE LEGITIMACY")
    print("=" * 60)
    print("""
✅ This NR curve can be used with pyreflect for:
   1. INFERENCE: Load a trained model and predict SLD profile from this NR curve
   2. VALIDATION: Compare model predictions against fitted/known SLD profiles

⚠️  This data CANNOT be used directly for TRAINING because:
   - Training requires BOTH NR curves AND corresponding SLD profiles (ground truth)
   - Experimental data only has NR (what we measure)
   - SLD profile is the unknown we're trying to determine!

📊 Workflow options:
   A. Use synthetic data (generated by pyreflect) for training
   B. Use this experimental NR for inference with trained model
   C. Compare predicted SLD with manual fitting results (validation)

📁 Next steps:
   1. Upload {}_nr.npy to pyreflect interface
   2. Select a trained model
   3. Run inference to predict SLD profile
   4. Compare with manual fitting results from ORNL analysis
""".format(base_name))


if __name__ == "__main__":
    main()
