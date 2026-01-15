import numpy as np
from refnx.reflect import SLD, ReflectModel
from refl1d.sample.reflectivity import reflectivity_amplitude as reflamp, convolve

def compute_nr_from_sld(
    sld_arr,
    Q=None,
    qmin=0.008,
    qmax=0.20,
    n_q=400,
    # Use ONE of these:
    dq_over_q=None,             # fractional resolution (e.g. 0.0125)
    q_resolution=None,          # matches your generator: dq = (q_resolution/2.355)*Q
    # Roughness:
    sigma=0.0,                  # default 0 for smooth continuous SLD(z)
    # Media handling:
    rho_fronting=None,          # if None, infer from profile front end
    rho_backing=None,           # if None, infer from profile back end
    # Numerics:
    order="substrate_to_air",   # or "air_to_substrate"
    dz=None,                    # e.g. 1.0 Å resampling for stable microslicing
    pad_front=200.0,            # Å plateau padding for semi-infinite fronting
    pad_back=200.0,             # Å plateau padding for semi-infinite backing
    plateau_pts=30,             # number of points to average for inferred front/back SLD
    apply_convolution=True,
):
    """
    Compute neutron reflectivity from a continuous SLD profile using refl1d's slab reflectivity.

    Parameters
    ----------
    sld_arr: (z, rho) or (z, rho, irho)
        z in Å, rho in 1e-6 Å^-2
    order:
        "substrate_to_air" means z increases from substrate -> air (your current convention)
        "air_to_substrate" means z increases from air -> substrate
    sigma:
        Interface roughness applied BETWEEN the microslabs you create.
        If your profile already has smooth transitions (e.g., from smooth_profile()),
        sigma should usually be 0.0.
    dq_over_q or q_resolution:
        Use dq_over_q for generic smearing.
        Use q_resolution to match your generator: dq = (q_resolution/2.355) * Q
    """

    # --- Parse input ---
    z = np.asarray(sld_arr[0], dtype=float)
    rho = np.asarray(sld_arr[1], dtype=float)
    irho = None
    if len(sld_arr) >= 3 and sld_arr[2] is not None:
        irho = np.asarray(sld_arr[2], dtype=float)

    if z.ndim != 1 or rho.ndim != 1 or z.size != rho.size:
        raise ValueError("z and rho must be 1D arrays of equal length")

    # Ensure strictly increasing and unique z (your caller already does this, but keep robust)
    z, idx = np.unique(z, return_index=True)
    rho = rho[idx]
    if irho is not None:
        irho = irho[idx]

    if z.size < 3:
        raise ValueError("Need at least 3 z points to compute reflectivity from a profile")

    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing")

    # --- Reorder to fronting -> backing (air/water -> substrate) ---
    # refl1d assumes neutrons come from the *fronting* medium.
    if order == "substrate_to_air":
        # input is substrate -> air, reverse to air -> substrate
        z = z[::-1]
        rho = rho[::-1]
        if irho is not None:
            irho = irho[::-1]
    elif order == "air_to_substrate":
        pass
    else:
        raise ValueError("order must be 'substrate_to_air' or 'air_to_substrate'")

    # Shift so fronting starts at z=0
    z = z - z[0]

    # --- Optional resampling for stable microslicing ---
    if dz is not None:
        dz = float(dz)
        if dz <= 0:
            raise ValueError("dz must be positive")
        z_fine = np.arange(z[0], z[-1] + 0.5*dz, dz)
        rho = np.interp(z_fine, z, rho)
        if irho is not None:
            irho = np.interp(z_fine, z, irho)
        z = z_fine

    # --- Infer fronting/backing from plateaus if not provided ---
    k = min(int(plateau_pts), z.size)
    if rho_fronting is None:
        rho_fronting = float(np.mean(rho[:k]))
    if rho_backing is None:
        rho_backing = float(np.mean(rho[-k:]))

    irho_fronting = 0.0 if irho is None else float(np.mean(irho[:k]))
    irho_backing  = 0.0 if irho is None else float(np.mean(irho[-k:]))

    # --- Pad semi-infinite media as thick plateaus (improves low-Q consistency) ---
    # front pad at z<0 and back pad at z>zmax
    if pad_front and pad_front > 0:
        z = np.r_[0.0, z + pad_front]
        rho = np.r_[rho_fronting, rho]
        if irho is not None:
            irho = np.r_[irho_fronting, irho]
        else:
            irho = None  # keep None

    if pad_back and pad_back > 0:
        z = np.r_[z, z[-1] + pad_back]
        rho = np.r_[rho, rho_backing]
        if irho is not None:
            irho = np.r_[irho, irho_backing]

    # Re-shift to start at 0
    z = z - z[0]

    # --- Build Q ---
    if Q is None:
        Q = np.linspace(qmin, qmax, int(n_q))
    else:
        Q = np.asarray(Q, dtype=float)

    # --- Build dQ for convolution ---
    if q_resolution is not None:
        # matches your generator
        dq_over_q_eff = float(q_resolution) / 2.355
    elif dq_over_q is not None:
        dq_over_q_eff = float(dq_over_q)
    else:
        dq_over_q_eff = 0.0

    dQ = np.clip(Q * dq_over_q_eff, 1e-12, None)

    # --- Convert continuous profile to microslabs correctly ---
    w_film = np.diff(z)                      # length N-1
    rho_mid = 0.5 * (rho[:-1] + rho[1:])     # length N-1
    if irho is None:
        irho_mid = np.zeros_like(rho_mid)
    else:
        irho_mid = 0.5 * (irho[:-1] + irho[1:])

    # Now build slab arrays INCLUDING semi-infinite front/back (0 thickness)
    w = np.r_[0.0, w_film, 0.0]              # length (N-1)+2
    rrho = np.r_[rho_fronting, rho_mid, rho_backing]
    iirho = np.r_[irho_fronting, irho_mid, irho_backing]

    # --- Roughness array per interface ---
    n_interfaces = len(rrho) - 1
    if np.isscalar(sigma):
        sigma_arr = np.full(n_interfaces, float(sigma))
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.size != n_interfaces:
            raise ValueError("sigma must be scalar or length len(rrho)-1")

    # --- Reflectivity ---
    # refl1d's low-level routine expects kz (1/Å), not Q (1/Å).
    # For specular reflectometry, Q = 2*kz.
    kz = Q / 2.0
    r = reflamp(kz, w, rrho, iirho, sigma_arr)
    R = np.abs(r) ** 2

    # --- Instrument resolution convolution (only if you want it) ---
    if apply_convolution and dq_over_q_eff > 0:
        R = convolve(Q, R, Q, dQ)

    return Q, R


def compute_nr_from_sld_refnx(
    sld_arr,
    qmin=0.00843,
    qmax=0.09275,
    n_q=400,
    dq_over_q=0.025,
    sigma=3.0,
):
    """
    Compute neutron reflectivity using refnx from a continuous SLD profile.

    Parameters
    ----------
    sld_arr : tuple
        (z, rho) or (z, rho, irho)
        z, rho ordered from substrate -> air
    qmin, qmax : float
        Q range (Å^-1)
    n_q : int
        Number of Q points
    dq_over_q : float
        Fractional resolution ΔQ / Q
    sigma : float
        Interfacial roughness in Å

    Returns
    -------
    Q : ndarray
        Momentum transfer (Å^-1)
    R : ndarray
        Reflectivity
    """

    # -----------------------------
    # Parse input
    # -----------------------------
    z = np.asarray(sld_arr[0], dtype=float)
    rho = np.asarray(sld_arr[1], dtype=float)

    if z.ndim != 1 or rho.ndim != 1 or z.size != rho.size:
        raise ValueError("z and rho must be 1D arrays of equal length")

    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing (substrate → air)")

    # -----------------------------
    # Reverse: substrate → air  →  air → substrate
    # -----------------------------
    z = z[::-1]
    rho = rho[::-1]

    # shift so fronting interface starts at z = 0
    z = z - z[0]

    # -----------------------------
    # Define fronting & backing
    # -----------------------------
    air = SLD(0.0, name="air")
    substrate = SLD(rho[-1], name="substrate")

    # -----------------------------
    # Build slab layers
    # -----------------------------
    layers = []
    for i in range(len(z) - 1):
        thickness = z[i + 1] - z[i]
        mid_sld = 0.5 * (rho[i] + rho[i + 1])

        layer = SLD(mid_sld, name=f"layer{i}")(thickness, sigma)
        layers.append(layer)

    # -----------------------------
    # Build structure: air | film | substrate
    # -----------------------------
    structure = air(0, sigma)
    for layer in layers:
        structure |= layer
    structure |= substrate(0, sigma)

    # -----------------------------
    # Reflectivity model
    # -----------------------------
    model = ReflectModel(structure, bkg=0.0, scale=1.0)
    model.dq = dq_over_q

    # -----------------------------
    # Compute NR
    # -----------------------------
    Q = np.linspace(qmin, qmax, n_q)
    R = model(Q)

    return Q, R


# for experimental data
def reverse_y_order(sld_array):
    """
    Flip SLD y from left to right, substrate to air direction
    """
    A_flipped = sld_array.copy()
    A_flipped[1] = A_flipped[1, ::-1]
    return A_flipped


def find_substrate_critical_idx(arr, target=2.075):
    """
    the idx of critical point where substrate transit to material film
    """
    y = arr[1]
    for i in range(len(y) - 1):
        if y[i] <= target and y[i + 1] > target:
            return i


def align_points(A, B):
    """
    A is expt sld, B is pred sld

    Align B to A using shift.
    Both A and B are shape (2, N).
    """
    # Step 1: find index in A where y ≈ 2.07 (left edge)
    target_y = 2.07

    # Search from left to right, the closest idx x which y close to target y
    idx_A = find_substrate_critical_idx(A)

    idx_B = find_substrate_critical_idx(B)

    # Step 2: compute the translation vector
    shift = A[:, idx_A] - B[:, idx_B]

    # Step 3: apply the shift
    B_aligned = B + shift[:, np.newaxis]

    return B_aligned
