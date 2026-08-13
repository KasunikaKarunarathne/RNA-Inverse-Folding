"""
Phase 15: Full End-to-End RNA Inverse Folding Evaluation Pipeline
=================================================================
This script addresses all weaknesses identified in the pipeline audit:
  1. Uses OLS coefficients (proven most accurate, RMSE 0.236 kcal/mol)
  2. 2x2 Factorial Design isolating QUBO vs Penalty contributions
  3. Logs unique pair assignments and SA diversity metrics
  4. Multi-stem compatible penalty functions

Factors:
  A) Pair Selection: Random Uniform vs QUBO Simulated Annealing Top 10%
  B) Loop Filling:   No Penalty (random) vs Mirror+Entropy Penalty (filtered)

This gives 4 experimental conditions per target structure.
"""

import os
import csv
import random
import RNA
import neal
import numpy as np
from scipy.stats import spearmanr
from collections import Counter

# ── Reuse existing phase functions ──────────────────────────────────────────
from phase1_rules import extract_stems, ALLOWED_PAIRS
from phase2_turner_energy import get_turner_energy
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase8_paired_sampling import generate_random_pairs, evaluate_qubo_from_pairs
from phase10_annealin import decode_bits_to_pairs


# ============================================================================
# HELPER: Convert flat pair tuple to nested stem-based list
# ============================================================================
def flat_to_nested(flat_tuple, stems):
    """Converts ('GC','AU','CG','UA') into [['GC','AU'],['CG','UA']] for 2 stems."""
    nested, idx = [], 0
    for stem in stems:
        stem_pairs = []
        for _ in stem:
            stem_pairs.append(flat_tuple[idx])
            idx += 1
        nested.append(stem_pairs)
    return nested


# ============================================================================
# FIXED PENALTY FUNCTIONS (Multi-stem compatible)
# ============================================================================
def calculate_mirror_penalty_multistem(sequence, target_structure):
    """
    Multi-stem mirror penalty. Unlike the original (phases 9-11) which only 
    checked stems[-1], this iterates over EVERY stem's terminal pair.
    
    For each stem, the terminal (innermost) pair borders a loop region.
    We penalize symmetric inward extensions that could form spurious stacking.
    """
    stems = extract_stems(target_structure)
    if not stems:
        return 0.0

    total_penalty = 0.0

    for stem in stems:
        left, right = stem[-1]  # Terminal (innermost) pair of this stem

        L = right - left - 1
        m = L // 2
        if m <= 0:
            continue

        prev_pair_str = sequence[left] + sequence[right]
        curr_l, curr_r = left + 1, right - 1

        for d in range(1, m + 1):
            if curr_l >= curr_r:
                break

            curr_pair_str = sequence[curr_l] + sequence[curr_r]
            w_d = 1.0 - 0.5 * ((d - 1) / (m - 1)) if m > 1 else 1.0

            if curr_pair_str in ALLOWED_PAIRS and prev_pair_str in ALLOWED_PAIRS:
                stack_energy = get_turner_energy(prev_pair_str, curr_pair_str)
                phi = max(0.0, -stack_energy)
                total_penalty += w_d * phi

            prev_pair_str = curr_pair_str
            curr_l += 1
            curr_r -= 1

    return total_penalty


def calculate_loop_entropy_penalty_multistem(sequence, target_structure):
    """
    Multi-stem loop entropy penalty. Groups consecutive dots into separate 
    loop regions and only checks for potential base pairs WITHIN each region.
    
    The original (phase 10) checked all dots globally, which incorrectly 
    penalized bases from different stems' loops pairing with each other.
    """
    penalty = 0.0
    loop_indices = [i for i, char in enumerate(target_structure) if char == "."]
    if not loop_indices:
        return 0.0

    # Group consecutive dot indices into separate loop regions
    loop_regions = []
    current_region = [loop_indices[0]]
    for i in range(1, len(loop_indices)):
        if loop_indices[i] == loop_indices[i - 1] + 1:
            current_region.append(loop_indices[i])
        else:
            loop_regions.append(current_region)
            current_region = [loop_indices[i]]
    loop_regions.append(current_region)

    # Apply penalty WITHIN each loop region independently
    for region in loop_regions:
        for i in range(len(region)):
            # RNA loops need minimum 3 bases to bend, so skip pairs < 4 apart
            for j in range(i + 4, len(region)):
                idx1 = region[i]
                idx2 = region[j]
                pair_str = sequence[idx1] + sequence[idx2]
                if pair_str in ALLOWED_PAIRS:
                    penalty += 2
    return penalty

def get_qubo_pairs_via_noisy_annealing(target_structure, c_coeffs, num_reads=1000):
    """
    Solves the QUBO via Simulated Annealing with dynamically shifting Gaussian noise.
    Simulates Quantum Tunneling by continuously shifting the energy landscape.
    """
    stems = extract_stems(target_structure)
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)

    total_pairs = sum(len(stem) for stem in stems)
    dynamic_noise_scale = min(0.15, max(0.03, total_pairs * 0.015))

    sampler = neal.SimulatedAnnealingSampler()
    all_results = []
    
    # Run 10 separate batches. Generate a BRAND NEW noisy landscape for each batch!
    num_batches = 10
    reads_per_batch = num_reads // num_batches
    
    for _ in range(num_batches):
        noisy_Q = {}
        for k, v in Q_dict.items():
            noisy_Q[k] = v + np.random.normal(0, dynamic_noise_scale * max(0.1, abs(v)))
            
        sampleset = sampler.sample_qubo(noisy_Q, num_reads=reads_per_batch)
        
        for sample, _ in sampleset.data(["sample", "energy"]):
            pairs_list = decode_bits_to_pairs(sample, stems)
            if pairs_list is not None:
                nested_pairs = flat_to_nested(pairs_list, stems)
                true_energy = evaluate_qubo_from_pairs(nested_pairs, stems, Q_dict, offset)
                all_results.append({"pairs_list": pairs_list, "qubo": true_energy})
    
    all_results.sort(key=lambda x: x["qubo"])
    all_pair_tuples = [r["pairs_list"] for r in all_results]
    unique_total = len(set(all_pair_tuples))
    freq_counter = Counter(all_pair_tuples)

    ten_pct_idx = max(1, int(len(all_results) * 0.10))
    top_slice = all_results[:ten_pct_idx]
    top_unique = []
    seen = set()
    for r in top_slice:
        if r["pairs_list"] not in seen:
            seen.add(r["pairs_list"])
            top_unique.append(r)

    diversity = {
        "total_valid_reads": len(all_results),
        "total_unique": unique_total,
        "top10_count": ten_pct_idx,
        "top10_unique": len(top_unique),
        "ground_state_freq": freq_counter.most_common(1)[0][1] if freq_counter else 0,
        "top3_assignments": freq_counter.most_common(3),
    }
    return top_unique, diversity


# ============================================================================
# PAIR SELECTION METHODS
# ============================================================================
def get_qubo_pairs_via_annealing(target_structure, c_coeffs, num_reads=1000):
    """
    Solves the QUBO via Simulated Annealing and returns deduplicated top-10%
    pair assignments, plus diversity statistics for logging.
    """
    stems = extract_stems(target_structure)
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)

    all_results = []
    for sample, energy in sampleset.data(["sample", "energy"]):
        pairs_list = decode_bits_to_pairs(sample, stems)
        if pairs_list is not None:
            all_results.append({"pairs_list": pairs_list, "qubo": energy + offset})

    all_results.sort(key=lambda x: x["qubo"])

    # ── Diversity metrics ──
    all_pair_tuples = [r["pairs_list"] for r in all_results]
    unique_total = len(set(all_pair_tuples))
    freq_counter = Counter(all_pair_tuples)

    # Take top 10% then deduplicate
    ten_pct_idx = max(1, int(len(all_results) * 0.10))
    top_slice = all_results[:ten_pct_idx]
    top_unique = []
    seen = set()
    for r in top_slice:
        if r["pairs_list"] not in seen:
            seen.add(r["pairs_list"])
            top_unique.append(r)

    diversity = {
        "total_valid_reads": len(all_results),
        "total_unique": unique_total,
        "top10_count": ten_pct_idx,
        "top10_unique": len(top_unique),
        "ground_state_freq": freq_counter.most_common(1)[0][1] if freq_counter else 0,
        "top3_assignments": freq_counter.most_common(3),
    }
    return top_unique, diversity


def get_random_pairs_for_experiment(target_structure, num_assignments):
    """
    Generates purely random pair assignments (no QUBO guidance).
    Returns the same dict format as the QUBO method for uniform handling.
    """
    stems = extract_stems(target_structure)
    samples = generate_random_pairs(stems, num_assignments)
    results = [{"pairs_list": flat_tuple, "qubo": None} for _, flat_tuple in samples]

    diversity = {
        "total_valid_reads": num_assignments,
        "total_unique": num_assignments,
        "top10_count": num_assignments,
        "top10_unique": num_assignments,
        "ground_state_freq": 1,
        "top3_assignments": [],
    }
    return results, diversity


# ============================================================================
# LOOP FILLING
# ============================================================================
def fill_loops(target_structure, pair_assignment, pool_size, num_output, use_penalty):
    """
    Fills unpaired positions with random nucleotides.
    
    If use_penalty=True:  generates `pool_size` candidates, ranks by 
                          mirror+entropy penalty, returns the best `num_output`.
    If use_penalty=False: generates `num_output` directly (no filtering at all).
    """
    stems = extract_stems(target_structure)
    full_length = len(target_structure)

    # Map fixed paired bases
    fixed_bases = {}
    pair_idx = 0
    for stem in stems:
        for (left, right) in stem:
            pair_string = pair_assignment[pair_idx]
            fixed_bases[left] = pair_string[0]
            fixed_bases[right] = pair_string[1]
            pair_idx += 1
            
    # Inject stabilizing Tetraloops (Positive Design)
    import re
    # Find all (....) hairpin loops
    for match in re.finditer(r'\(\.\.\.\.\)', target_structure):
        close_left = match.start()   # The '(' position
        close_right = match.end() - 1  # The ')' position
        # Force closing pair to C-G (optimal for GNRA tetraloops)
        fixed_bases[close_left] = 'C'
        fixed_bases[close_right] = 'G'
        # Force loop to GCAA
        idx = match.start() + 1
        fixed_bases[idx] = 'G'
        fixed_bases[idx+1] = 'C'
        fixed_bases[idx+2] = 'A'
        fixed_bases[idx+3] = 'A'


    bases = ["A", "U", "C", "G"]
    gen_count = pool_size if use_penalty else num_output
    candidates = []

    for _ in range(gen_count):
        seq = []
        for i in range(full_length):
            if i in fixed_bases:
                seq.append(fixed_bases[i])
            else:
                #seq.append(random.choice(bases))
                # Negative Design: 90% chance to pick A or C to prevent unwanted loop pairings
                seq.append(random.choices(["A", "C", "G", "U"], weights=[45, 45, 5, 5])[0])
        full_seq = "".join(seq)

        if use_penalty:
            penalty = calculate_mirror_penalty_multistem(full_seq, target_structure)
            penalty += calculate_loop_entropy_penalty_multistem(full_seq, target_structure)
            candidates.append((penalty, full_seq))
        else:
            candidates.append((0.0, full_seq))

    if use_penalty:
        candidates.sort(key=lambda x: x[0])

    return [seq for _, seq in candidates[:num_output]]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_forward_folding(sequences, target_structure):
    """Folds every sequence with ViennaRNA, returning stats and successful sequences."""
    success_seqs = []
    for seq in sequences:
        struct, mfe = RNA.fold(seq)
        if struct == target_structure:
            success_seqs.append({"seq": seq, "mfe": mfe})
            
    total = len(sequences)
    success = len(success_seqs)
    rate = (success / total * 100) if total > 0 else 0.0
    return total, success, rate, success_seqs


def compute_correlation(target_structure, stems, Q_dict, offset, corr_samples, use_penalty, pool_size):
    """
    Computes Spearman correlation (QUBO vs Vienna) over a set of random pair 
    assignments. The `use_penalty` flag determines how loops are filled, which 
    affects the Vienna energy and therefore the correlation.
    
    Returns (overall_spearman, bottom_10%_spearman).
    """
    combined = []
    for nested_pairs, flat_tuple in corr_samples:
        q_energy = evaluate_qubo_from_pairs(nested_pairs, stems, Q_dict, offset)

        seqs = fill_loops(target_structure, flat_tuple, pool_size=pool_size, num_output=1, use_penalty=use_penalty)
        if not seqs:
            continue
        full_seq = seqs[0]

        fc = RNA.fold_compound(full_seq)
        v_energy = fc.eval_structure(target_structure)
        combined.append({"qubo": q_energy, "vienna": v_energy})

    if len(combined) < 4:
        return float("nan"), float("nan")

    sorted_c = sorted(combined, key=lambda x: x["qubo"])
    all_q = [x["qubo"] for x in sorted_c]
    all_v = [x["vienna"] for x in sorted_c]

    ov_corr, _ = spearmanr(all_v, all_q)

    b10_idx = max(2, int(len(sorted_c) * 0.10))
    b10_corr_val, _ = spearmanr(
        [x["vienna"] for x in sorted_c[:b10_idx]],
        [x["qubo"] for x in sorted_c[:b10_idx]],
    )
    return ov_corr, b10_corr_val


# ============================================================================
def run_full_evaluation(
    target_structures,
    num_sa_reads=1000,
    min_random_assignments=10,
    pool_size=1000,
    variations_per_assignment=10,
    num_corr_samples=500,
):
    """
    Runs the full 2x2 factorial experiment across all target structures.
    
    Outputs:
      - factorial_results.csv       Main success rates and correlations
      - diversity_log.csv           SA sampling diversity analysis
      - pair_assignments_log.csv    Every pair assignment used (full audit trail)
    """
    conditions = [
        {"name": "Random_NoPenalty",  "pair": "random", "penalty": False},
        {"name": "Random_Penalty",    "pair": "random", "penalty": True},
        {"name": "QUBO_SA_NoPenalty", "pair": "qubo",   "penalty": False},
        {"name": "QUBO_SA_Penalty",   "pair": "qubo",   "penalty": True},
        {"name": "Noisy_SA_NoPenalty","pair": "noisy",  "penalty": False},
        {"name": "Noisy_SA_Penalty",  "pair": "noisy",  "penalty": True},
    ]

    out_dir = "results/phase15_full_evaluation"
    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-compute OLS coefficients ONCE ──
    print("=" * 70)
    print("PHASE 15: Full Pipeline Evaluation")
    print("=" * 70)
    print("Pre-computing QUBO coefficients using OLS (RMSE=0.236 kcal/mol)...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    print("Coefficients ready.\n")

    # ── Open all output files ──
    results_path = os.path.join(out_dir, "factorial_results.csv")
    diversity_path = os.path.join(out_dir, "diversity_log.csv")
    pairs_path = os.path.join(out_dir, "pair_assignments_log.csv")
    best_seqs_path = os.path.join(out_dir, "best_final_sequences.csv")

    rf = open(results_path, "w", newline="")
    divf = open(diversity_path, "w", newline="")
    pf = open(pairs_path, "w", newline="")
    bsf = open(best_seqs_path, "w", newline="")

    rw = csv.writer(rf)
    dw = csv.writer(divf)
    pw = csv.writer(pf)
    bsw = csv.writer(bsf)

    rw.writerow([
        "Target Structure", "Length", "Num Stems", "Total Pairs",
        "Condition", "Pair Method", "Penalty Used",
        "Total Tested", "Successful Folds", "Success Rate (%)",
        "Spearman Corr Overall", "Spearman Corr B10%",
        "Unique Pair Assignments Used",
    ])

    dw.writerow([
        "Target Structure", "Pair Method",
        "Total Valid Reads", "Total Unique Assignments",
        "Top 10% Count", "Top 10% Unique",
        "Ground State Frequency", "Diversity Ratio (%)",
        "Top 3 Most Frequent Assignments",
    ])

    pw.writerow([
        "Target Structure", "Pair Method", "Assignment Index",
        "Pair Assignment (dash-separated)", "QUBO Energy",
    ])

    try:
        for struct in target_structures:
            stems = extract_stems(struct)
            total_pairs = sum(len(s) for s in stems)

            print(f"\n{'━' * 70}")
            print(f"TARGET: {struct}")
            print(f"Length={len(struct)} | Stems={len(stems)} | Pairs={total_pairs}")
            print(f"{'━' * 70}")

            Q_dict, offset = build_approx_qubo(stems, c_coeffs)

            # ── 1. Get QUBO/SA pair assignments ──
            qubo_pairs, q_div = get_qubo_pairs_via_annealing(struct, c_coeffs, num_reads=num_sa_reads)
            div_ratio = q_div["total_unique"] / max(1, q_div["total_valid_reads"]) * 100
            top3_str = "; ".join(
                [f"{'-'.join(t[0])} (x{t[1]})" for t in q_div["top3_assignments"]]
            )
            print(
                f"  [SA] Reads={q_div['total_valid_reads']}, "
                f"Unique={q_div['total_unique']} ({div_ratio:.1f}%), "
                f"Top10%={q_div['top10_unique']} unique, "
                f"Ground-state freq={q_div['ground_state_freq']}"
            )
            dw.writerow([
                struct, "QUBO_SA",
                q_div["total_valid_reads"], q_div["total_unique"],
                q_div["top10_count"], q_div["top10_unique"],
                q_div["ground_state_freq"], f"{div_ratio:.2f}",
                top3_str,
            ])

            for idx, item in enumerate(qubo_pairs):
                pw.writerow([
                    struct, "QUBO_SA", idx,
                    "-".join(item["pairs_list"]),
                    f"{item['qubo']:.4f}" if item["qubo"] is not None else "N/A",
                ])


            # ── 1.5 Get Noisy SA pair assignments (Simulated Quantum Tunneling) ──
            noisy_pairs, n_div = get_qubo_pairs_via_noisy_annealing(struct, c_coeffs, num_reads=num_sa_reads)
            n_div_ratio = n_div["total_unique"] / max(1, n_div["total_valid_reads"]) * 100
            n_top3_str = "; ".join([f"{'-'.join(t[0])} (x{t[1]})" for t in n_div["top3_assignments"]])
            print(
                f"  [Noisy SA] Reads={n_div['total_valid_reads']}, "
                f"Unique={n_div['total_unique']} ({n_div_ratio:.1f}%), "
                f"Top10%={n_div['top10_unique']} unique"
            )
            dw.writerow([
                struct, "Noisy_SA",
                n_div["total_valid_reads"], n_div["total_unique"],
                n_div["top10_count"], n_div["top10_unique"],
                n_div["ground_state_freq"], f"{n_div_ratio:.2f}",
                n_top3_str,
            ])
            for idx, item in enumerate(noisy_pairs):
                pw.writerow([
                    struct, "Noisy_SA", idx,
                    "-".join(item["pairs_list"]),
                    f"{item['qubo']:.4f}" if item["qubo"] is not None else "N/A",
                ])

            # ── 2. Get Random pair assignments (at least as many as QUBO unique) ──
            num_rand = max(min_random_assignments, len(qubo_pairs), len(noisy_pairs))
            random_pairs, r_div = get_random_pairs_for_experiment(struct, num_rand)
            print(f"  [Random] Generated {len(random_pairs)} unique assignments")

            dw.writerow([
                struct, "Random",
                r_div["total_valid_reads"], r_div["total_unique"],
                r_div["top10_count"], r_div["top10_unique"],
                r_div["ground_state_freq"], "100.00",
                "",
            ])

            for idx, item in enumerate(random_pairs):
                pw.writerow([
                    struct, "Random", idx,
                    "-".join(item["pairs_list"]),
                    "N/A",
                ])

            divf.flush()
            pf.flush()

            # ── 3. Generate shared random pair sample for correlation ──
            # Same sample used for both penalty settings to ensure fair comparison
            print(f"  Generating {num_corr_samples} random pairs for correlation...")
            corr_sample_raw = generate_random_pairs(stems, num_corr_samples)

            # ── 4. Compute correlation ONCE per penalty setting ──
            print("  Computing Spearman correlations...")
            corr_no_penalty = compute_correlation(
                struct, stems, Q_dict, offset, corr_sample_raw,
                use_penalty=False, pool_size=pool_size
            )
            corr_with_penalty = compute_correlation(
                struct, stems, Q_dict, offset, corr_sample_raw,
                use_penalty=True, pool_size=pool_size
            )
            print(
                f"  Corr(NoPenalty): Overall={corr_no_penalty[0]:.4f}, B10%={corr_no_penalty[1]:.4f}"
            )
            print(
                f"  Corr(Penalty):   Overall={corr_with_penalty[0]:.4f}, B10%={corr_with_penalty[1]:.4f}"
            )

            # ── 5. Run 4 factorial conditions ──
            print(
                f"\n  {'Condition':<25} | {'Folds':>10} | {'Rate':>8} | "
                f"{'Corr(All)':>10} | {'Corr(B10)':>10} | {'Unique':>7}"
            )
            print("  " + "-" * 82)

            for cond in conditions:
                if cond["pair"] == "qubo":
                    pair_list = qubo_pairs
                elif cond["pair"] == "noisy":
                    pair_list = noisy_pairs
                else:
                    pair_list = random_pairs
                    
                corr = corr_with_penalty if cond["penalty"] else corr_no_penalty

                # Generate all sequences for this condition
                all_seqs = []
                for item in pair_list:
                    seqs = fill_loops(
                        struct, item["pairs_list"],
                        pool_size=pool_size,
                        num_output=variations_per_assignment,
                        use_penalty=cond["penalty"],
                    )
                    all_seqs.extend(seqs)

                total, succ, rate, success_seqs = evaluate_forward_folding(all_seqs, struct)
                unique_used = len(set(item["pairs_list"] for item in pair_list))

                # Log top 10 unique successful sequences
                success_seqs.sort(key=lambda x: x["mfe"])
                unique_seqs = []
                seen_seq = set()
                for item in success_seqs:
                    if item["seq"] not in seen_seq:
                        seen_seq.add(item["seq"])
                        unique_seqs.append(item)
                
                for r_idx, item in enumerate(unique_seqs[:10]):
                    bsw.writerow([struct, cond["name"], r_idx+1, item["seq"], f"{item['mfe']:.2f}"])
                bsf.flush()

                print(
                    f"  {cond['name']:<25} | {succ:>4}/{total:<5} | {rate:>6.1f}% | "
                    f"{corr[0]:>10.4f} | {corr[1]:>10.4f} | {unique_used:>7}"
                )

                rw.writerow([
                    struct, len(struct), len(stems), total_pairs,
                    cond["name"], cond["pair"], cond["penalty"],
                    total, succ, f"{rate:.2f}",
                    f"{corr[0]:.4f}", f"{corr[1]:.4f}",
                    unique_used,
                ])
                rf.flush()

    finally:
        rf.close()
        divf.close()
        pf.close()
        bsf.close()


    print(f"\n{'=' * 70}")
    print("ALL OUTPUTS:")
    print(f"  {results_path}")
    print(f"  {diversity_path}")
    print(f"  {pairs_path}")
    print(f"  {best_seqs_path}")
    print(f"{'=' * 70}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # ── Test structures: hairpins + multi-stem ──
    test_structures = [
        # Simple hairpins (increasing complexity)
        "(((...)))",
        "((((((((....))))))))",
        "((((((((..........))))))))",
        "(((((((((((((((....)))))))))))))))",
        # Multi-stem structures
        "(((...)))(((...)))",
        # With flanking unpaired regions
        "..............(((((.....)))))",
        "...((((((.........)))))).",
    ]

    run_full_evaluation(
        target_structures=test_structures,
        num_sa_reads=1000,
        min_random_assignments=10,
        pool_size=1000,
        variations_per_assignment=10,
        num_corr_samples=500,
    )
