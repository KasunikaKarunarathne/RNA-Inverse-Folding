"""
Hybrid QAOA with Qiskit Aer Matrix Product State (MPS) Simulator
Enables simulating large RNA structures (K >= 10, e.g. InfoRNA K=14)
in < 50 MB RAM instead of 1.25 TB.
"""

import os
import csv
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import RNA

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

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
P_LAYERS = 5                   # QAOA depth layers (p)
NUM_QUBO_PAIRS = 5             # Top unique candidate pairings extracted
LOOPS_PER_PAIR = 2             # Loop variations generated per candidate
TOTAL_SEQ_PER_METHOD = NUM_QUBO_PAIRS * LOOPS_PER_PAIR
SHOTS = 2048                 # Quantum measurement shots per iteration

PAIR_TO_IDX = {p: i for i, p in enumerate(ALLOWED_PAIRS)}
IDX_TO_PAIR = {i: p for i, p in enumerate(ALLOWED_PAIRS)}

# 6x6 Hadfield Ring Mixer Generator
M6 = np.zeros((6, 6), dtype=float)
for i in range(6):
    M6[i, (i + 1) % 6] = 1.0
    M6[(i + 1) % 6, i] = 1.0

# 8x8 unitary embedding of M6 into 3 qubits
M8 = np.zeros((8, 8), dtype=float)
M8[:6, :6] = M6


def get_mps_simulator():
    """Initializes high-performance Matrix Product State Aer simulator."""
    return AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=32,
        matrix_product_state_truncation_threshold=1e-6
    )


def build_qaoa_mps_circuit(n_pairs, stems, gamma_params, beta_params, mode="cold", seed_pairs=None):
    """
    Constructs a 3*K qubit QAOA circuit tailored for MPS 1D tensor contraction.
    """
    num_qubits = 3 * n_pairs
    qc = QuantumCircuit(num_qubits)

    # -------------------------------------------------------------
    # 1. State Preparation
    # -------------------------------------------------------------
    if mode == "cold":
        # Prepare uniform superposition over the 6 valid basis states on each 3-qubit qudit
        v6 = np.zeros(8, dtype=complex)
        v6[:6] = 1.0 / np.sqrt(6.0)
        Q, R = np.linalg.qr(np.column_stack([v6, np.eye(8)[:, 1:]]))
        if R[0, 0] < 0:
            Q[:, 0] = -Q[:, 0]
        gate_init = UnitaryGate(Q, label="Init_6")
        for q in range(n_pairs):
            qc.append(gate_init, [3 * q, 3 * q + 1, 3 * q + 2])

    elif mode == "warm":
        # Set exact classical seed on each 3-qubit qudit
        for q, p in enumerate(seed_pairs):
            idx = PAIR_TO_IDX[p]
            for b in range(3):
                if (idx >> b) & 1:
                    qc.x(3 * q + b)

    # -------------------------------------------------------------
    # 2. Alternating QAOA Layers
    # -------------------------------------------------------------
    p_layers = len(gamma_params)

    for l in range(p_layers):
        gamma = gamma_params[l]
        beta = beta_params[l]

        # (A) Cost Hamiltonian U_P(gamma)
        global_rung = 0
        for stem in stems:
            stem_len = len(stem)
            for k in range(stem_len - 1):
                q1 = global_rung + k
                q2 = global_rung + k + 1

                diag_phases = np.ones(64, dtype=complex)
                for i1 in range(6):
                    for i2 in range(6):
                        e = get_turner_energy(IDX_TO_PAIR[i1], IDX_TO_PAIR[i2])
                        idx_64 = (i2 << 3) | i1  # Qiskit basis order: q2 is MSB, q1 is LSB
                        diag_phases[idx_64] = np.exp(-1j * gamma * e)

                U_cost = np.diag(diag_phases)
                gate_cost = UnitaryGate(U_cost, label=f"H_stack_{k}")
                qubits_q1 = [3 * q1, 3 * q1 + 1, 3 * q1 + 2]
                qubits_q2 = [3 * q2, 3 * q2 + 1, 3 * q2 + 2]
                qc.append(gate_cost, qubits_q1 + qubits_q2)

            global_rung += stem_len

        # (B) Mixer Unitary U_M(beta)
        U_m = expm(-1j * beta * M8)
        gate_mixer = UnitaryGate(U_m, label="M6_ring")
        for q in range(n_pairs):
            qc.append(gate_mixer, [3 * q, 3 * q + 1, 3 * q + 2])

    qc.measure_all()
    return qc


def run_qaoa_mps(target_structure, mode="warm", seed_pairs=None, p_layers=P_LAYERS, 
                 optimizer="COBYLA", maxiter=30, shots=SHOTS):
    """
    Executes QAOA on arbitrary structure size using Qiskit Aer MPS.
    Supports COBYLA, Nelder-Mead, and Powell.
    """
    stems = extract_stems(target_structure)
    pair_coords = []
    for stem_idx, stem in enumerate(stems):
        for rung_idx, pair in enumerate(stem):
            pair_coords.append((stem_idx, rung_idx))
    n_pairs = len(pair_coords)

    sim = get_mps_simulator()

    def objective(params):
        if mode == "cold":
            gammas = params[0::2]
            betas = params[1::2]
        else:
            betas = [params[0]] + list(params[2::2])
            gammas = list(params[1::2])

        qc = build_qaoa_mps_circuit(n_pairs, stems, gammas, betas, mode=mode, seed_pairs=seed_pairs)
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()

        total_energy = 0.0
        valid_shots = 0

        for bitstring, count in counts.items():
            clean_bits = bitstring.replace(" ", "")[::-1]
            valid = True
            sampled_pairs = []

            for q in range(n_pairs):
                rung_bits = clean_bits[3 * q: 3 * q + 3]
                idx = int(rung_bits[0]) + (int(rung_bits[1]) << 1) + (int(rung_bits[2]) << 2)
                if idx >= 6:
                    valid = False
                    break
                sampled_pairs.append(IDX_TO_PAIR[idx])

            if valid:
                energy = 0.0
                g_idx = 0
                for stem in stems:
                    for k in range(len(stem) - 1):
                        energy += get_turner_energy(sampled_pairs[g_idx + k], sampled_pairs[g_idx + k + 1])
                    g_idx += len(stem)
                total_energy += energy * count
                valid_shots += count

        return (total_energy / valid_shots) if valid_shots > 0 else 0.0

    # Initial guess
    x0 = [0.5, 0.2] * p_layers if mode == "cold" else [0.2] + ([0.5, 0.2] * p_layers)
    
    # Run user-selected optimizer (COBYLA, Nelder-Mead, or Powell)
    opt_res = minimize(objective, x0=x0, method=optimizer, options={"maxiter": maxiter})

    if mode == "cold":
        final_gammas = opt_res.x[0::2]
        final_betas = opt_res.x[1::2]
    else:
        final_betas = [opt_res.x[0]] + list(opt_res.x[2::2])
        final_gammas = list(opt_res.x[1::2])

    final_qc = build_qaoa_mps_circuit(n_pairs, stems, final_gammas, final_betas, mode=mode, seed_pairs=seed_pairs)
    final_counts = sim.run(final_qc, shots=2048).result().get_counts()

    candidate_dict = {}
    for bitstring, count in final_counts.items():
        clean_bits = bitstring.replace(" ", "")[::-1]
        valid = True
        sampled_pairs = []
        for q in range(n_pairs):
            idx = int(clean_bits[3 * q]) + (int(clean_bits[3 * q + 1]) << 1) + (int(clean_bits[3 * q + 2]) << 2)
            if idx >= 6:
                valid = False
                break
            sampled_pairs.append(IDX_TO_PAIR[idx])
        if valid:
            t_pairs = tuple(sampled_pairs)
            candidate_dict[t_pairs] = candidate_dict.get(t_pairs, 0) + count

    sorted_candidates = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)
    return [(pairs, count / 2048.0) for pairs, count in sorted_candidates[:NUM_QUBO_PAIRS]], opt_res

def evaluate_candidate_pairs(target_structure, pair_assignments, total_target_seqs=TOTAL_SEQ_PER_METHOD):
    """
    Fills loops and validates biological folding with ViennaRNA.
    Guarantees that strictly total_target_seqs (10) sequences are evaluated,
    even if the quantum optimizer concentrated on only 1 or 2 unique pairings.
    """
    if not pair_assignments:
        return {
            "total_seqs": total_target_seqs, "raw_succ": 0, "raw_rate": 0.0,
            "unique_gen": 0, "unique_succ": 0, "unique_rate": 0.0, "avg_mfe": 0.0
        }

    n_candidates = len(pair_assignments)
    # Dynamically distribute the 10 sequence slots across available pair candidates
    base_loops = total_target_seqs // n_candidates
    remainder = total_target_seqs % n_candidates

    generated_seqs = []
    for idx, (pairs, prob) in enumerate(pair_assignments):
        # Give remaining slots to top-ranked candidates
        num_loops = base_loops + (1 if idx < remainder else 0)
        if num_loops > 0:
            seqs = fill_loops_custom(target_structure, pairs, num_output=num_loops, penalty_type="shifted_mirror")
            generated_seqs.extend(seqs)

    # Fallback padding just in case loop filler returned fewer than requested
    while len(generated_seqs) < total_target_seqs:
        generated_seqs.append("A" * len(target_structure))

    # Evaluate biological folding with ViennaRNA
    total_seqs = len(generated_seqs)
    successes = 0
    unique_seqs = set(generated_seqs)
    unique_successes = set()
    mfes = []

    for seq in generated_seqs:
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


def run_fmqa_benchmark_mps(target_name, target_structure, c_coeffs, 
                           optimizers=['COBYLA', 'Nelder-Mead', 'Powell']):
    """Executes comparative benchmark across Classical, Cold QAOA, and Warm QAOA."""
    stems = extract_stems(target_structure)
    pair_coords = []
    for stem_idx, stem in enumerate(stems):
        for rung_idx, pair in enumerate(stem):
            pair_coords.append((stem_idx, rung_idx))
    n_pairs = len(pair_coords)
    num_qubits = 3 * n_pairs

    print("\n" + "=" * 85)
    print(f"BENCHMARKING FMQA STRUCTURE: {target_name}")
    print(f"Target Dot-Bracket : {target_structure}")
    print(f"Base Pairs (K)     : {n_pairs} pairs ({num_qubits} physical qubits in MPS)")
    print(f"Classical Subspace : 6^{n_pairs} = {6 ** n_pairs:,} states (Simulated via MPS)")
    print("=" * 85)

    # 1. Classical Backbone (D-Wave SA)
    print("[1/3] Running Classical Backbone Baseline (D-Wave SA)...")
    t0 = time.time()
    classical_results = get_qubo_pairs_via_annealing_coaxial(
        target_structure, c_coeffs, num_reads=500, sampler_type="SA"
    )
    t_classical = time.time() - t0
    top_classical_pairs = [(r["pairs_list"], 1.0) for r in classical_results[:NUM_QUBO_PAIRS]]
    best_classical_seed = top_classical_pairs[0][0]
    metrics_cl = evaluate_candidate_pairs(target_structure, top_classical_pairs)
    print(f"  -> Classical SA finished in {t_classical:.2f}s")
    print(f"  -> Success: {metrics_cl['raw_succ']}/{metrics_cl['total_seqs']} ({metrics_cl['raw_rate']:.1f}%) | "
          f"Uniq: {metrics_cl['unique_succ']}/{metrics_cl['unique_gen']} ({metrics_cl['unique_rate']:.1f}%)")
    print(f"  -> Best Classical Seed: {best_classical_seed}")

    # 2. Cold-Start QAOA (MPS) across Optimizers
    cold_results = {}
    print(f"\n[2/3] Running Cold-Start QAOA (MPS, p={P_LAYERS})...")
    for opt in optimizers:
        t0 = time.time()
        pairs_cold, res_cold = run_qaoa_mps(target_structure, mode="cold", p_layers=P_LAYERS, optimizer=opt, maxiter=25)
        t_opt = time.time() - t0
        metrics_cold = evaluate_candidate_pairs(target_structure, pairs_cold)
        cold_results[opt] = {"metrics": metrics_cold, "energy": res_cold.fun, "time": t_opt}
        print(f"  -> [{opt:<11}] Energy: {res_cold.fun:>6.2f} | Succ: {metrics_cold['raw_succ']}/{metrics_cold['total_seqs']} ({metrics_cold['raw_rate']:>5.1f}%) | Time: {t_opt:.2f}s")

    # 3. Warm-Start QAOA (MPS) across Optimizers
    warm_results = {}
    print(f"\n[3/3] Running Warm-Start QAOA (MPS, p={P_LAYERS}, Seeded from SA)...")
    for opt in optimizers:
        t0 = time.time()
        pairs_warm, res_warm = run_qaoa_mps(target_structure, mode="warm", seed_pairs=best_classical_seed, 
                                            p_layers=P_LAYERS, optimizer=opt, maxiter=25)
        t_opt = time.time() - t0
        metrics_warm = evaluate_candidate_pairs(target_structure, pairs_warm)
        warm_results[opt] = {"metrics": metrics_warm, "energy": res_warm.fun, "time": t_opt}
        print(f"  -> [{opt:<11}] Energy: {res_warm.fun:>6.2f} | Succ: {metrics_warm['raw_succ']}/{metrics_warm['total_seqs']} ({metrics_warm['raw_rate']:>5.1f}%) | Time: {t_opt:.2f}s")

    # Summary Table
    print("\n" + "-" * 95)
    print(f"{'Methodology':<35} | {'Energy (kcal)':<15} | {'Raw Success':<18} | {'Unique Success':<15}")
    print("-" * 95)
    print(f"{'Classical Backbone (SA)':<35} | {'N/A':<15} | {metrics_cl['raw_succ']}/{metrics_cl['total_seqs']} ({metrics_cl['raw_rate']:.1f}%)  | {metrics_cl['unique_succ']}/{metrics_cl['unique_gen']} ({metrics_cl['unique_rate']:.1f}%)")
    for opt in optimizers:
        rc = cold_results[opt]
        print(f"{('Cold QAOA (' + opt + ', p=' + str(P_LAYERS) + ')'):<35} | {rc['energy']:<15.2f} | {rc['metrics']['raw_succ']}/{rc['metrics']['total_seqs']} ({rc['metrics']['raw_rate']:.1f}%)  | {rc['metrics']['unique_succ']}/{rc['metrics']['unique_gen']} ({rc['metrics']['unique_rate']:.1f}%)")
    for opt in optimizers:
        rw = warm_results[opt]
        print(f"{('Warm QAOA (' + opt + ', p=' + str(P_LAYERS) + ')'):<35} | {rw['energy']:<15.2f} | {rw['metrics']['raw_succ']}/{rw['metrics']['total_seqs']} ({rw['metrics']['raw_rate']:.1f}%)  | {rw['metrics']['unique_succ']}/{rw['metrics']['unique_gen']} ({rw['metrics']['unique_rate']:.1f}%)")
    print("-" * 95 + "\n")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
if __name__ == "__main__":
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
    print(f"Config: P_LAYERS = {P_LAYERS}, SHOTS = {SHOTS}\n")

    for name, target in structures:
        run_fmqa_benchmark_mps(
            target_name=name,
            target_structure=target,
            c_coeffs=c_coeffs
        )