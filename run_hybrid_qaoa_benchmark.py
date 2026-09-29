import os
import csv
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from collections import defaultdict
import RNA

# Import from existing Phase 1-21 modules
from phase1_rules import extract_stems, ALLOWED_PAIRS
from phase2_turner_energy import get_turner_energy
from phase3_coef_fitter import calculate_qubo_coeffs
from final_pipeline_with_post_processing import (
    get_qubo_pairs_via_annealing_coaxial,
    fill_loops_custom
)

# =====================================================================
# GLOBAL CONFIGURATION VARIABLES
# =====================================================================
P_LAYERS = 5                   # Number of QAOA alternating layers (p)
NUM_RUNS = 1                   # Number of evaluation runs to repeat
TIMEOUT_SECONDS = 300          # 5-minute timeout limit per structure/run
MAX_SUBSPACE_DIM = 10_000_000  # 10M states safeguard limit (~80 MB)
NUM_QUBO_PAIRS = 5             # Number of top candidate pairs to extract
LOOPS_PER_PAIR = 2             # Number of loop variations to generate per pair
TOTAL_SEQ_PER_METHOD = NUM_QUBO_PAIRS * LOOPS_PER_PAIR  # Total sequences evaluated

ALLOWED_PAIRS = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG']
PAIR_TO_IDX = {p: i for i, p in enumerate(ALLOWED_PAIRS)}
IDX_TO_PAIR = {i: p for i, p in enumerate(ALLOWED_PAIRS)}

# 6x6 Ring Mixer Generator (Hadfield One-Hot / Qudit Ring Mixer)
M6 = np.zeros((6, 6), dtype=float)
for i in range(6):
    M6[i, (i + 1) % 6] = 1.0
    M6[(i + 1) % 6, i] = 1.0


# =====================================================================
# 1. FAST VECTORIZED HAMILTONIAN & SUBSPACE MIXER
# =====================================================================
def build_turner_hamiltonian(target_structure):
    """
    Constructs the diagonal Turner Physical Hamiltonian H_C over the 
    feasible subspace of dimension 6^K. Zero penalty terms (W = 0).
    """
    stems = extract_stems(target_structure)
    pair_coords = []
    for stem_idx, stem in enumerate(stems):
        for rung_idx, pair in enumerate(stem):
            pair_coords.append((stem_idx, rung_idx))
    
    n_pairs = len(pair_coords)
    dim = 6 ** n_pairs

    # Memory safeguard check:
    if dim > MAX_SUBSPACE_DIM:
        return None, n_pairs, stems

    H_C = np.zeros(dim, dtype=float)

    for idx in range(dim):
        temp = idx
        p_indices = []
        for _ in range(n_pairs):
            p_indices.append(temp % 6)
            temp //= 6
        p_indices.reverse()
        
        pairs = [IDX_TO_PAIR[pi] for pi in p_indices]
        
        energy = 0.0
        global_pair_idx = 0
        for stem in stems:
            stem_len = len(stem)
            for k in range(stem_len - 1):
                p_top = pairs[global_pair_idx + k]
                p_bottom = pairs[global_pair_idx + k + 1]
                energy += get_turner_energy(p_top, p_bottom)
            global_pair_idx += stem_len
            
        H_C[idx] = energy

    return H_C, n_pairs, stems


def apply_subspace_ring_mixer(state, beta, n_pairs):
    """
    Applies Hadfield's Subspace Ring Mixer:
    U_M(beta) = tensor_product_k exp(-i * beta * M6)
    """
    U_m6 = expm(-1j * beta * M6)
    dim = 6 ** n_pairs
    tensor = state.reshape([6] * n_pairs)
    
    for q in range(n_pairs):
        tensor = np.moveaxis(tensor, q, 0)
        shape_rest = tensor.shape[1:]
        tensor = tensor.reshape(6, -1)
        tensor = U_m6 @ tensor
        tensor = tensor.reshape((6,) + shape_rest)
        tensor = np.moveaxis(tensor, 0, q)
        
    return tensor.reshape(dim)


# =====================================================================
# 2. P-LAYER GENERALIZED QAOA EXECUTOR
# =====================================================================
def run_qaoa_optimization(H_C, n_pairs, mode="cold", seed_pairs=None, 
                          optimizer="COBYLA", p_layers=P_LAYERS, maxiter=150):
    """
    Executes p-layer QAOA with arbitrary p >= 1.
    Cold:  p layers of [U_P(gamma_l), U_M(beta_l)]
    Warm:  initial U_M(beta0) + p layers of [U_P(gamma_l), U_M(beta_l)]
    """
    dim = 6 ** n_pairs
    
    if mode == "cold":
        s = np.ones(dim, dtype=complex) / np.sqrt(dim)
        
        def objective(params):
            psi = s.copy()
            for l in range(p_layers):
                gamma = params[2 * l]
                beta = params[2 * l + 1]
                psi = np.exp(-1j * gamma * H_C) * psi # cost
                psi = apply_subspace_ring_mixer(psi, beta, n_pairs) # mixer 
            probs = np.abs(psi) ** 2
            return np.sum(probs * H_C)
        
        # Initial guess: [0.5, 0.5] repeated for each layer
        x0 = [0.5, 0.5] * p_layers
        opt_res = minimize(objective, x0=x0, method=optimizer, options={'maxiter': maxiter})
        
        # Reconstruct final statevector
        final_psi = s.copy()
        for l in range(p_layers):
            gamma = opt_res.x[2 * l]
            beta = opt_res.x[2 * l + 1]
            final_psi = np.exp(-1j * gamma * H_C) * final_psi
            final_psi = apply_subspace_ring_mixer(final_psi, beta, n_pairs)
        
    elif mode == "warm":
        if seed_pairs is None:
            raise ValueError("Warm start requires seed_pairs from classical backbone!")
        
        seed_idx = 0
        for p in seed_pairs:
            seed_idx = seed_idx * 6 + PAIR_TO_IDX[p]
        s_seed = np.zeros(dim, dtype=complex)
        s_seed[seed_idx] = 1.0
        
        def objective(params):
            beta0 = params[0]
            psi = apply_subspace_ring_mixer(s_seed, beta0, n_pairs)
            for l in range(p_layers):
                gamma = params[1 + 2 * l]
                beta = params[1 + 2 * l + 1]
                psi = np.exp(-1j * gamma * H_C) * psi
                psi = apply_subspace_ring_mixer(psi, beta, n_pairs)
            probs = np.abs(psi) ** 2
            return np.sum(probs * H_C)
            
        # Initial guess: beta0 = 0.2, then [0.5, 0.2] for each layer
        x0 = [0.2] + ([0.5, 0.2] * p_layers)
        opt_res = minimize(objective, x0=x0, method=optimizer, options={'maxiter': maxiter})
        # objective only outputs a scalar energy number and discards the quantum state. 
        # The second loop reconstructs the actual winning quantum statevector (final_psi) 
        # so we can sample biological RNA sequences from it.
        final_psi = apply_subspace_ring_mixer(s_seed, opt_res.x[0], n_pairs)
        for l in range(p_layers):
            gamma = opt_res.x[1 + 2 * l]
            beta = opt_res.x[1 + 2 * l + 1]
            final_psi = np.exp(-1j * gamma * H_C) * final_psi
            final_psi = apply_subspace_ring_mixer(final_psi, beta, n_pairs)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return final_psi, opt_res


# =====================================================================
# 3. SAMPLING & BIOLOGICAL VALIDATION (VIENNARNA)
# =====================================================================
def decode_statevector_to_pairs(psi, n_pairs, top_k=NUM_QUBO_PAIRS):
    """Extracts top_k most probable base-pair assignments from quantum state."""
    dim = 6 ** n_pairs
    probs = np.abs(psi) ** 2
    top_indices = np.argsort(probs)[-top_k:][::-1]
    
    candidate_pairs = []
    for idx in top_indices:
        temp = idx
        p_indices = []
        for _ in range(n_pairs):
            p_indices.append(temp % 6)
            temp //= 6
        p_indices.reverse()
        pairs = tuple(IDX_TO_PAIR[pi] for pi in p_indices)
        candidate_pairs.append((pairs, probs[idx]))
        
    return candidate_pairs


def evaluate_candidate_pairs(target_structure, pair_assignments, loops_per_pair=LOOPS_PER_PAIR):
    """Fills loops and validates biological folding with ViennaRNA."""
    total_seqs = 0
    successes = 0
    unique_seqs = set()
    unique_successes = set()
    mfes = []
    
    for pairs, prob in pair_assignments:
        seqs = fill_loops_custom(target_structure, pairs, num_output=loops_per_pair, penalty_type="mirror_entropy")
        for seq in seqs:
            total_seqs += 1
            unique_seqs.add(seq)
            folded_struct, mfe = RNA.fold(seq)
            mfes.append(mfe)
            if folded_struct == target_structure:
                successes += 1
                unique_successes.add(seq)

    raw_succ_rate = (successes / total_seqs) * 100 if total_seqs > 0 else 0.0
    unique_succ_rate = (len(unique_successes) / len(unique_seqs)) * 100 if len(unique_seqs) > 0 else 0.0
    avg_mfe = np.mean(mfes) if mfes else 0.0
    
    return {
        "total_seqs": total_seqs,
        "raw_succ": successes,
        "raw_rate": raw_succ_rate,
        "unique_gen": len(unique_seqs),
        "unique_succ": len(unique_successes),
        "unique_rate": unique_succ_rate,
        "avg_mfe": avg_mfe
    }


# =====================================================================
# 4. UNIFIED COMPARATIVE BENCHMARK RUNNER
# =====================================================================
def run_fmqa_benchmark(target_name, target_structure, c_coeffs, 
                       optimizers=['COBYLA', 'Nelder-Mead', 'Powell']):
    """Executes comparative benchmark across Classical, Cold QAOA, and Warm QAOA."""
    print("\n" + "=" * 80)
    print(f"BENCHMARKING FMQA STRUCTURE: {target_name}")
    print(f"Target Dot-Bracket : {target_structure}")
    print("=" * 80)
    
    H_C, n_pairs, stems = build_turner_hamiltonian(target_structure)
    dim = 6 ** n_pairs
    print(f"Stems extracted    : {len(stems)} stem(s), {n_pairs} total base pairs")
    print(f"Feasible Subspace  : 6^{n_pairs} = {dim:,} states (Zero invalid states)")
    
    # Check memory limit safeguard:
    if H_C is None:
        print(f"\n[Safeguard Skip] State space dimension ({dim:,}) exceeds MAX_SUBSPACE_DIM limit ({MAX_SUBSPACE_DIM:,}).")
        print("                 Full exact statevector simulation requires > 5 GB RAM. Skipping gracefully.\n")
        return

    print(f"Turner Energy Range: min = {np.min(H_C):.2f} kcal/mol, max = {np.max(H_C):.2f} kcal/mol")
    print(f"QAOA Layers (p)    : {P_LAYERS}")
    print(f"Total Sequences    : {TOTAL_SEQ_PER_METHOD} per method ({NUM_QUBO_PAIRS} pairs x {LOOPS_PER_PAIR} loops)\n")
    
    # -------------------------------------------------------------
    # (A) RUN CLASSICAL BACKBONE BASELINE
    # -------------------------------------------------------------
    print("[1/3] Running Classical Backbone Baseline (D-Wave SA)...")
    t0 = time.time()
    classical_results = get_qubo_pairs_via_annealing_coaxial(
        target_structure, c_coeffs, num_reads=500, sampler_type="SA"
    )
    t_classical = time.time() - t0
    
    top_classical_pairs = [(r["pairs_list"], 1.0) for r in classical_results[:NUM_QUBO_PAIRS]]
    best_classical_seed = top_classical_pairs[0][0]
    
    metrics_classical = evaluate_candidate_pairs(target_structure, top_classical_pairs)
    print(f"  -> Classical SA finished in {t_classical:.2f}s")
    print(f"  -> Raw: {metrics_classical['raw_succ']}/{metrics_classical['total_seqs']} ({metrics_classical['raw_rate']:.1f}%) | "
          f"Uniq: {metrics_classical['unique_succ']}/{metrics_classical['unique_gen']} ({metrics_classical['unique_rate']:.1f}%)")
    print(f"  -> Best Classical Seed: {best_classical_seed}")

    # -------------------------------------------------------------
    # (B) RUN COLD QAOA (WITHOUT CLASSICAL BACKBONE)
    # -------------------------------------------------------------
    cold_results = {}
    print(f"\n[2/3] Running Cold-Start QAOA (p={P_LAYERS}, Without Classical Backbone)...")
    for opt in optimizers:
        t0 = time.time()
        psi_cold, res_cold = run_qaoa_optimization(
            H_C, n_pairs, mode="cold", optimizer=opt, p_layers=P_LAYERS, maxiter=100
        )
        t_opt = time.time() - t0
        pairs_cold = decode_statevector_to_pairs(psi_cold, n_pairs, top_k=NUM_QUBO_PAIRS)
        metrics_cold = evaluate_candidate_pairs(target_structure, pairs_cold)
        cold_results[opt] = {
            "metrics": metrics_cold, "energy": res_cold.fun, 
            "iters": res_cold.nfev, "time": t_opt
        }
        print(f"  -> [{opt:<11}] Energy: {res_cold.fun:.2f} | "
              f"Raw: {metrics_cold['raw_succ']}/{metrics_cold['total_seqs']} ({metrics_cold['raw_rate']:>5.1f}%) | "
              f"Uniq: {metrics_cold['unique_succ']}/{metrics_cold['unique_gen']} ({metrics_cold['unique_rate']:>5.1f}%) | Time: {t_opt:.2f}s")

    # -------------------------------------------------------------
    # (C) RUN WARM QAOA (WITH CLASSICAL BACKBONE SEED)
    # -------------------------------------------------------------
    warm_results = {}
    print(f"\n[3/3] Running Warm-Start QAOA (p={P_LAYERS}, With Classical Backbone Seed)...")
    for opt in optimizers:
        t0 = time.time()
        psi_warm, res_warm = run_qaoa_optimization(
            H_C, n_pairs, mode="warm", seed_pairs=best_classical_seed, 
            optimizer=opt, p_layers=P_LAYERS, maxiter=100
        )
        t_opt = time.time() - t0
        pairs_warm = decode_statevector_to_pairs(psi_warm, n_pairs, top_k=NUM_QUBO_PAIRS)
        metrics_warm = evaluate_candidate_pairs(target_structure, pairs_warm)
        warm_results[opt] = {
            "metrics": metrics_warm, "energy": res_warm.fun, 
            "iters": res_warm.nfev, "time": t_opt
        }
        print(f"  -> [{opt:<11}] Energy: {res_warm.fun:.2f} | "
              f"Raw: {metrics_warm['raw_succ']}/{metrics_warm['total_seqs']} ({metrics_warm['raw_rate']:>5.1f}%) | "
              f"Uniq: {metrics_warm['unique_succ']}/{metrics_warm['unique_gen']} ({metrics_warm['unique_rate']:>5.1f}%) | Time: {t_opt:.2f}s")

    # -------------------------------------------------------------
    # (D) SUMMARY TABLE WITH DETAILED COUNTS
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'Method / Configuration':<35} | {'Optim Energy':<12} | {'Raw Succ (Count/Total)':<22} | {'Uniq Succ / Gen':<20} | {'Time (s)':<8}")
    print("-" * 95)
    raw_str_cl = f"{metrics_classical['raw_succ']}/{metrics_classical['total_seqs']} ({metrics_classical['raw_rate']:.1f}%)"
    unq_str_cl = f"{metrics_classical['unique_succ']}/{metrics_classical['unique_gen']} ({metrics_classical['unique_rate']:.1f}%)"
    print(f"{'Classical Backbone (SA)':<35} | {'N/A':<12} | {raw_str_cl:<22} | {unq_str_cl:<20} | {t_classical:>7.2f}")
    print("-" * 95)
    for opt in optimizers:
        r = cold_results[opt]
        raw_str = f"{r['metrics']['raw_succ']}/{r['metrics']['total_seqs']} ({r['metrics']['raw_rate']:.1f}%)"
        unq_str = f"{r['metrics']['unique_succ']}/{r['metrics']['unique_gen']} ({r['metrics']['unique_rate']:.1f}%)"
        print(f"{('Cold QAOA (' + opt + ') [p=' + str(P_LAYERS) + ']'):<35} | {r['energy']:>10.2f}   | {raw_str:<22} | {unq_str:<20} | {r['time']:>7.2f}")
    print("-" * 95)
    for opt in optimizers:
        r = warm_results[opt]
        raw_str = f"{r['metrics']['raw_succ']}/{r['metrics']['total_seqs']} ({r['metrics']['raw_rate']:.1f}%)"
        unq_str = f"{r['metrics']['unique_succ']}/{r['metrics']['unique_gen']} ({r['metrics']['unique_rate']:.1f}%)"
        print(f"{('Warm QAOA (' + opt + ') [p=' + str(P_LAYERS) + ']'):<35} | {r['energy']:>10.2f}   | {raw_str:<22} | {unq_str:<20} | {r['time']:>7.2f}")
    print("=" * 95 + "\n")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    # to build the classical QUBO and generate the Warm-Start seed:
    print("Pre-computing OLS coefficients...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    csv_path = r"Structures/fmqa_paper_structures.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit(1)
        
    structures = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            structures.append((row["Category"], row["Structure"]))

    print(f"Loaded {len(structures)} structures from {csv_path}.")
    print(f"Config: P_LAYERS = {P_LAYERS}, TIMEOUT = {TIMEOUT_SECONDS}s, NUM_RUNS = {NUM_RUNS}\n")
    
    for name, target in structures:
        run_fmqa_benchmark(
            target_name=name, 
            target_structure=target, 
            c_coeffs=c_coeffs,
            optimizers=['COBYLA', 'Nelder-Mead', 'Powell']
        )
