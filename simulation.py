"""
RNA Nearest-Neighbor Stacking Energy: HUBO vs QUBO Simulation
=============================================================
Implements the full pipeline from the slides:
  1. Binary encoding of RNA base pairs (3 bits each)
  2. Indicator polynomials (degree-3, exact)
  3. HUBO stacking energy (degree-6, exact)
  4. QUBO approximation via least-squares (degree-2)
  5. Comparison, error analysis, and stem energy evaluation

Turner 2004 nearest-neighbor parameters (kcal/mol) used throughout.
"""

import numpy as np
import itertools

# ─────────────────────────────────────────────
# 1. BASE-PAIR ENCODING (Table from slides p.3)
# ─────────────────────────────────────────────

PAIRS = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG']

# Each pair encoded as (p0, p1, p2) binary vector
ENCODING = {
    'AU': np.array([1, 1, 1]),
    'UA': np.array([1, 1, 0]),
    'CG': np.array([1, 0, 1]),
    'GC': np.array([1, 0, 0]),
    'GU': np.array([0, 1, 1]),
    'UG': np.array([0, 1, 0]),
}

# ─────────────────────────────────────────────
# 2. TURNER 2004 NEAREST-NEIGHBOR TABLE (2 d.p.)
#    Rows = pair k (alpha), Cols = pair k+1 (beta)
# ─────────────────────────────────────────────

TURNER = {
    'AU': {'AU': -0.93, 'UA': -1.10, 'CG': -2.11, 'GC': -2.24, 'GU': -0.55, 'UG': -1.36},
    'UA': {'AU': -1.10, 'UA': -0.93, 'CG': -2.11, 'GC': -1.36, 'GU': -1.44, 'UG': -0.55},
    'CG': {'AU': -2.11, 'UA': -1.36, 'CG': -3.26, 'GC': -2.36, 'GU': -1.36, 'UG': -2.11},
    'GC': {'AU': -2.36, 'UA': -2.24, 'CG': -3.42, 'GC': -3.26, 'GU': -1.44, 'UG': -2.51},
    'GU': {'AU': -1.27, 'UA': -1.01, 'CG': -2.51, 'GC': -2.11, 'GU': -0.50, 'UG': -1.27},
    'UG': {'AU': -1.01, 'UA': -1.27, 'CG': -2.11, 'GC': -1.36, 'GU': -1.27, 'UG': -0.50},
}

# ─────────────────────────────────────────────
# 3. INDICATOR POLYNOMIALS (slides p.4)
#    I_alpha(enc) = 1 iff enc matches alpha's encoding
# ─────────────────────────────────────────────

def indicators(enc):
    """
    Given a 3-bit encoding (p0, p1, p2), return dict of indicator values.
    Each I_alpha = product of (p_i or 1-p_i) depending on alpha's encoding.
    Exactly one indicator equals 1; all others equal 0.
    """
    p0, p1, p2 = enc
    return {
        'AU':  p0      *  p1      *  p2,
        'UA':  p0      *  p1      * (1 - p2),
        'CG':  p0      * (1 - p1) *  p2,
        'GC':  p0      * (1 - p1) * (1 - p2),
        'GU': (1 - p0) *  p1      *  p2,
        'UG': (1 - p0) *  p1      * (1 - p2),
    }

# ─────────────────────────────────────────────
# 4. HUBO EXACT STACKING ENERGY (slides p.5-6)
#    H_NN(s,k) = Σ_{α,β} ε(α,β) · I_α(k) · I_β(k+1)
#    Degree 6 (two cubic indicators multiplied)
# ─────────────────────────────────────────────

def hubo_stacking_energy(pair_k, pair_k1):
    """
    Exact 6-local HUBO energy for one stacking position.
    Selects exactly one Turner energy value from the 6x6 table.

    Args:
        pair_k   : string, e.g. 'AU' — base pair at position k
        pair_k1  : string, e.g. 'GC' — base pair at position k+1

    Returns:
        float: stacking energy in kcal/mol
    """
    iA = indicators(ENCODING[pair_k])
    iB = indicators(ENCODING[pair_k1])
    energy = 0.0
    for alpha in PAIRS:
        for beta in PAIRS:
            energy += TURNER[alpha][beta] * iA[alpha] * iB[beta]
    return energy

# ─────────────────────────────────────────────
# 5. QUBO DESIGN MATRIX (slides p.13-14)
#    Features: [1, q1..q6, q1q2, q1q3, ..., q5q6]
#    22 features total: 1 + 6 linear + 15 pairwise
# ─────────────────────────────────────────────

def build_qubo_feature(enc_k, enc_k1):
    """
    Build the 22-dim feature vector for one stacking position.
    q = [p0_k, p1_k, p2_k, p0_{k+1}, p1_{k+1}, p2_{k+1}]

    Features:
        [1, q1, q2, q3, q4, q5, q6,
         q1q2, q1q3, q1q4, q1q5, q1q6,
         q2q3, q2q4, q2q5, q2q6,
         q3q4, q3q5, q3q6,
         q4q5, q4q6,
         q5q6]
    """
    q = np.concatenate([enc_k, enc_k1])          # 6 binary variables
    feat = [1.0] + list(q)                         # constant + linear: 7 terms
    for a in range(6):
        for b in range(a + 1, 6):
            feat.append(q[a] * q[b])               # pairwise: 15 terms
    return np.array(feat)                          # total: 22 features

def build_design_matrix():
    """
    Build 36x22 design matrix Phi and 36-dim energy vector E.
    Each row = one (alpha, beta) stacking combination.
    """
    Phi = []
    E   = []
    for alpha in PAIRS:
        for beta in PAIRS:
            feat = build_qubo_feature(ENCODING[alpha], ENCODING[beta])
            Phi.append(feat)
            E.append(TURNER[alpha][beta])
    return np.array(Phi), np.array(E)

# ─────────────────────────────────────────────
# 6. FIT QUBO COEFFICIENTS via least-squares
#    min ||Phi @ c - E||^2  →  c* = pinv(Phi) @ E
# ─────────────────────────────────────────────

def fit_qubo_coefficients(Phi, E):
    """
    Solve the overdetermined system (36 equations, 22 unknowns)
    using numpy least-squares.

    Returns:
        c      : (22,) array of QUBO coefficients
        residuals : sum of squared residuals
        rank   : rank of Phi
    """
    c, residuals, rank, sv = np.linalg.lstsq(Phi, E, rcond=None)
    return c, residuals, rank

def qubo_stacking_energy(pair_k, pair_k1, coeffs):
    """
    Approximate 2-local QUBO energy for one stacking position.

    Args:
        pair_k, pair_k1 : base pair strings
        coeffs          : (22,) array of fitted QUBO coefficients

    Returns:
        float: approximate stacking energy in kcal/mol
    """
    feat = build_qubo_feature(ENCODING[pair_k], ENCODING[pair_k1])
    return float(feat @ coeffs)

# ─────────────────────────────────────────────
# 7. STEM ENERGY (slides p.8)
#    H_NN_s = Σ_{k=1}^{m-1} H_NN(s,k)
# ─────────────────────────────────────────────

def stem_hubo_energy(stem):
    """
    Total exact HUBO stacking energy for a stem.

    Args:
        stem : list of pair strings, e.g. ['GC', 'AU', 'CG', 'GC']

    Returns:
        float: total stacking energy (sum over m-1 positions)
    """
    return sum(
        hubo_stacking_energy(stem[k], stem[k + 1])
        for k in range(len(stem) - 1)
    )

def stem_qubo_energy(stem, coeffs):
    """
    Total approximate QUBO stacking energy for a stem.
    """
    return sum(
        qubo_stacking_energy(stem[k], stem[k + 1], coeffs)
        for k in range(len(stem) - 1)
    )

# ─────────────────────────────────────────────
# 8. ERROR ANALYSIS
# ─────────────────────────────────────────────

def analyse_fit(Phi, E, coeffs):
    """
    Compute R², RMSE, max error, and per-cell errors.
    """
    predicted = Phi @ coeffs
    errors    = predicted - E
    ss_res    = np.sum(errors ** 2)
    ss_tot    = np.sum((E - E.mean()) ** 2)
    r2        = 1 - ss_res / ss_tot
    rmse      = np.sqrt(ss_res / len(E))
    return {
        'predicted' : predicted,
        'errors'    : errors,
        'rmse'      : rmse,
        'r2'        : r2,
        'max_error' : np.max(np.abs(errors)),
        'mean_error': np.mean(np.abs(errors)),
    }

def noise_experiment(n_trials=50, noise_levels=None):
    """
    Add Gaussian noise to Turner energies and measure RMSE degradation.
    Shows how robust QUBO fitting is to measurement uncertainty.

    Args:
        n_trials    : number of trials per noise level
        noise_levels: list of sigma values (kcal/mol)

    Returns:
        dict: {sigma -> mean RMSE}
    """
    if noise_levels is None:
        noise_levels = np.arange(0.0, 1.55, 0.1)

    Phi, E_true = build_design_matrix()
    results = {}
    for sigma in noise_levels:
        rmses = []
        for _ in range(n_trials):
            E_noisy = E_true + np.random.normal(0, sigma, size=E_true.shape)
            c, _, _ = fit_qubo_coefficients(Phi, E_noisy)
            pred    = Phi @ c
            rmse    = np.sqrt(np.mean((pred - E_noisy) ** 2))
            rmses.append(rmse)
        results[round(sigma, 2)] = np.mean(rmses)
    return results

# ─────────────────────────────────────────────
# 9. COEFFICIENT NAMES (for display)
# ─────────────────────────────────────────────

def coefficient_names():
    names = ['c0']
    for i in range(1, 7):
        names.append(f'c{i}')
    for a in range(1, 7):
        for b in range(a + 1, 7):
            names.append(f'c{a}{b}')
    return names  # 22 names total

# ─────────────────────────────────────────────
# 10. MAIN — run everything and print results
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("RNA NEAREST-NEIGHBOR STACKING: HUBO vs QUBO")
    print("=" * 60)

    # --- Build system and fit ---
    Phi, E = build_design_matrix()
    coeffs, _, rank = fit_qubo_coefficients(Phi, E)
    stats = analyse_fit(Phi, E, coeffs)

    print(f"\n[Design matrix]  Shape: {Phi.shape}  (36 data points, 22 features)")
    print(f"[Matrix rank]    {rank}  (full rank = 22 means unique solution)")

    print(f"\n[Fit quality]")
    print(f"  RMSE      : {stats['rmse']:.4f} kcal/mol")
    print(f"  R²        : {stats['r2']:.6f}")
    print(f"  Max error : {stats['max_error']:.4f} kcal/mol")
    print(f"  Mean |err|: {stats['mean_error']:.4f} kcal/mol")

    # --- Print all 22 coefficients ---
    print(f"\n[QUBO coefficients c* — 22 values]")
    names = coefficient_names()
    for name, val in zip(names, coeffs):
        print(f"  {name:>5s} = {val:+.4f}")

    # --- Per-pair comparison table ---
    print(f"\n[Per-pair HUBO vs QUBO comparison]")
    print(f"  {'Stack':<8} {'HUBO exact':>12} {'QUBO approx':>12} {'Error':>10}  {'Fit'}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}  {'-'*6}")
    for alpha in PAIRS:
        for beta in PAIRS:
            h = hubo_stacking_energy(alpha, beta)
            q = qubo_stacking_energy(alpha, beta, coeffs)
            e = q - h
            fit = 'exact' if abs(e) < 0.01 else ('ok' if abs(e) < 0.10 else 'poor')
            print(f"  {alpha}/{beta:<5} {h:>12.3f} {q:>12.3f} {e:>+10.3f}  {fit}")

    # --- Example stem ---
    print(f"\n[Example stem energy: GC-AU-CG-GC]")
    stem = ['GC', 'AU', 'CG', 'GC']
    h_total = stem_hubo_energy(stem)
    q_total = stem_qubo_energy(stem, coeffs)
    print(f"  Stem       : {' — '.join(stem)}")
    print(f"  Pairs      : {len(stem)}, stacking positions: {len(stem)-1}")
    for k in range(len(stem) - 1):
        h = hubo_stacking_energy(stem[k], stem[k+1])
        q = qubo_stacking_energy(stem[k], stem[k+1], coeffs)
        print(f"  Stack {k+1}→{k+2} ({stem[k]}/{stem[k+1]}): HUBO={h:.3f}, QUBO={q:.3f}, err={q-h:+.3f}")
    print(f"  Total HUBO : {h_total:.3f} kcal/mol")
    print(f"  Total QUBO : {q_total:.3f} kcal/mol")
    print(f"  Total error: {q_total - h_total:+.3f} kcal/mol")

    # --- Noise experiment ---
    print(f"\n[Noise experiment — RMSE vs Gaussian noise sigma]")
    noise_results = noise_experiment(n_trials=30)
    print(f"  {'Sigma':>7}  {'Mean RMSE':>10}")
    for sigma, rmse in noise_results.items():
        bar = '|' * int(rmse * 40)
        print(f"  {sigma:>7.2f}  {rmse:>10.4f}  {bar}")

    # --- Scaling summary (slides p.10-11) ---
    print(f"\n[Scaling analysis]")
    stems_example = [['GC','AU','CG'], ['AU','GC','UG','CG'], ['CG','GC','AU','UA','GC']]
    total_M = sum(len(s) for s in stems_example)
    S = len(stems_example)
    print(f"  S (stems)         : {S}")
    print(f"  M (total pairs)   : {total_M}")
    print(f"  Symbolic summands : 36 × (M - S) = 36 × {total_M - S} = {36*(total_M - S)}")
    print(f"  m_max             : {max(len(s) for s in stems_example)}")
    print(f"  O(S × m_max)      : {S * max(len(s) for s in stems_example)}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)