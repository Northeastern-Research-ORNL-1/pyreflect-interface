#!/usr/bin/env python3
"""
Convert ORNL reflectometry CSV files to numpy format for pyreflect training.

Extracts BOTH:
1. NR data (Q, R) - model input
2. SLD profile (z, Nb) - model output/ground truth

Usage:
    python convert_csv_to_npy.py [--plot]
"""

import numpy as np
from pathlib import Path
import sys


def parse_reflectivity_csv(csv_path: Path) -> dict:
    """
    Parse an ORNL-style reflectivity CSV file.
    
    Returns dict with:
        - 'nr': (Q, R, dR) arrays - reflectivity data
        - 'sld': (z, Nb) arrays - SLD profile
        - 'layers': list of layer dicts - layer parameters
    """
    # Read with BOM handling
    for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {csv_path}")
    
    # Find column indices from header row (row 2, index 1)
    header_parts = [p.strip() for p in lines[1].split(',')]
    
    # Map column names to indices
    col_map = {}
    for i, name in enumerate(header_parts):
        col_map[name] = i
    
    # Key columns we need:
    # Reflectivity: Q (Å-1), R, dR
    # SLD Profile: z (Å), Nb (Å-2) - the LAST Nb column is the profile
    q_col = next((i for i, n in enumerate(header_parts) if 'Q (' in n), 8)
    r_col = next((i for i, n in enumerate(header_parts) if n == 'R'), 9)
    dr_col = next((i for i, n in enumerate(header_parts) if n == 'dR'), 10)
    
    # Find z and profile Nb columns (they're in the Profile section)
    z_col = next((i for i, n in enumerate(header_parts) if 'z (' in n), 17)
    # The profile Nb is the column right after z
    nb_profile_col = z_col + 1
    
    # Layer parameter columns
    compound_col = next((i for i, n in enumerate(header_parts) if n == 'Compound'), 1)
    nb_col = next((i for i, n in enumerate(header_parts) if 'Nb (' in n), 2)
    d_col = next((i for i, n in enumerate(header_parts) if n == 'd (Å)'), 5)
    sigma_col = next((i for i, n in enumerate(header_parts) if 'σ' in n or 'fwhm' in n), 6)
    
    print(f"  Column mapping: Q={q_col}, R={r_col}, z={z_col}, Nb_profile={nb_profile_col}")
    
    # Parse data rows
    q_values, r_values, dr_values = [], [], []
    z_values, nb_values = [], []
    layers = []
    
    for line in lines[2:]:  # Skip header rows
        parts = line.strip().split(',')
        if len(parts) <= max(q_col, r_col, z_col, nb_profile_col):
            continue
        
        # Extract reflectivity data
        try:
            q_str = parts[q_col].strip()
            r_str = parts[r_col].strip()
            if q_str and r_str:
                q = float(q_str)
                r = float(r_str)
                dr = float(parts[dr_col].strip()) if parts[dr_col].strip() else 0.0
                # Filter valid reflectometry range
                if 0 < q < 0.5 and 0 < r < 2.0:
                    q_values.append(q)
                    r_values.append(r)
                    dr_values.append(dr)
        except (ValueError, IndexError):
            pass
        
        # Extract SLD profile data
        try:
            z_str = parts[z_col].strip()
            nb_str = parts[nb_profile_col].strip()
            if z_str and nb_str:
                z = float(z_str)
                nb = float(nb_str)
                z_values.append(z)
                nb_values.append(nb)
        except (ValueError, IndexError):
            pass
        
        # Extract layer parameters
        try:
            compound = parts[compound_col].strip()
            if compound and compound not in ['', 'Compound']:
                layer = {
                    'name': compound,
                    'Nb': float(parts[nb_col]) if parts[nb_col].strip() else 0,
                    'd': float(parts[d_col]) if parts[d_col].strip() else 0,
                    'sigma': float(parts[sigma_col]) if parts[sigma_col].strip() else 0,
                }
                if layer['name'] not in [l['name'] for l in layers]:
                    layers.append(layer)
        except (ValueError, IndexError):
            pass
    
    return {
        'nr': (np.array(q_values), np.array(r_values), np.array(dr_values)),
        'sld': (np.array(z_values), np.array(nb_values)),
        'layers': layers,
    }


def plot_data(data: dict, title: str):
    """Plot NR and SLD data for visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot NR (Reflectivity)
    q, r, dr = data['nr']
    axes[0].semilogy(q, r, 'b.-', label='R(Q)')
    if dr.any():
        axes[0].fill_between(q, r-dr, r+dr, alpha=0.3)
    axes[0].set_xlabel('Q (Å⁻¹)')
    axes[0].set_ylabel('Reflectivity')
    axes[0].set_title('Neutron Reflectivity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot SLD Profile
    z, nb = data['sld']
    axes[1].plot(z, nb, 'r.-', label='SLD Profile')
    axes[1].set_xlabel('z (Å)')
    axes[1].set_ylabel('Nb (Å⁻²)')
    axes[1].set_title('SLD Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the plot
    plot_path = Path(__file__).parent / f"{title.replace(' ', '_')}_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plot saved: {plot_path.name}")
    plt.close()


def create_training_arrays(all_data: list[dict], target_length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Create training arrays from multiple samples.
    
    Returns:
        nr_train: shape (N, 2, L) - [Q, R] curves
        sld_train: shape (N, 2, L) - [z, Nb] profiles
    """
    nr_curves = []
    sld_curves = []
    
    for data in all_data:
        q, r, _ = data['nr']
        z, nb = data['sld']
        
        if len(q) < 10 or len(z) < 10:
            print(f"  Skipping sample with too few points (NR: {len(q)}, SLD: {len(z)})")
            continue
        
        # Interpolate to common grid
        q_interp = np.linspace(q.min(), q.max(), target_length)
        r_interp = np.interp(q_interp, q, r)
        
        z_interp = np.linspace(z.min(), z.max(), target_length)
        nb_interp = np.interp(z_interp, z, nb)
        
        nr_curves.append(np.stack([q_interp, r_interp], axis=0))
        sld_curves.append(np.stack([z_interp, nb_interp], axis=0))
    
    if not nr_curves:
        raise ValueError("No valid training samples found")
    
    return np.stack(nr_curves, axis=0), np.stack(sld_curves, axis=0)


def main():
    script_dir = Path(__file__).parent
    do_plot = '--plot' in sys.argv
    
    # Find all CSV files
    csv_files = sorted(script_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    all_data = []
    
    for csv_path in csv_files:
        print(f"Processing: {csv_path.name}")
        try:
            data = parse_reflectivity_csv(csv_path)
            q, r, dr = data['nr']
            z, nb = data['sld']
            
            print(f"  NR: {len(q)} points, Q=[{q.min():.4f}, {q.max():.4f}] Å⁻¹")
            print(f"  SLD: {len(z)} points, z=[{z.min():.1f}, {z.max():.1f}] Å")
            print(f"  Layers: {[l['name'] for l in data['layers']]}")
            
            # Save individual files
            base = csv_path.stem.replace("(", "_").replace(")", "_").replace(" ", "_")
            
            # Save NR curve
            nr_arr = np.stack([q, r], axis=0)
            nr_path = script_dir / f"{base}_nr.npy"
            np.save(nr_path, nr_arr)
            print(f"  Saved: {nr_path.name} (shape: {nr_arr.shape})")
            
            # Save SLD profile
            sld_arr = np.stack([z, nb], axis=0)
            sld_path = script_dir / f"{base}_sld.npy"
            np.save(sld_path, sld_arr)
            print(f"  Saved: {sld_path.name} (shape: {sld_arr.shape})")
            
            if do_plot:
                plot_data(data, base)
            
            all_data.append(data)
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")
    
    # Create combined training arrays
    if all_data:
        print("="*60)
        print("Creating training arrays...")
        try:
            nr_train, sld_train = create_training_arrays(all_data)
            
            np.save(script_dir / "nr_train.npy", nr_train)
            np.save(script_dir / "sld_train.npy", sld_train)
            
            print(f"  nr_train.npy: shape {nr_train.shape}")
            print(f"  sld_train.npy: shape {sld_train.shape}")
            print()
            print("These files can be uploaded to pyreflect-interface:")
            print("  - nr_train.npy → role: 'nr_train'")
            print("  - sld_train.npy → role: 'sld_train'")
        except Exception as e:
            print(f"  Error creating training arrays: {e}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. Upload training data via the web interface:
   - Upload nr_train.npy as 'nr_train' 
   - Upload sld_train.npy as 'sld_train'

2. Or use the API:
   POST /api/upload
   - files: [nr_train.npy, sld_train.npy]
   - roles: ['nr_train', 'sld_train']

3. Then start training in the interface!

Note: With only 2 samples, you'll want to either:
   - Add more experimental data files
   - Use synthetic data generation to augment
   - Use these for validation/testing only
""")


if __name__ == "__main__":
    main()
