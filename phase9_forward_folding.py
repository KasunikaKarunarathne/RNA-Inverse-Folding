from phase2_turner_energy import get_turner_energy
from phase1_rules import ALLOWED_PAIRS
import RNA
import random
import numpy as np
from phase1_rules import extract_stems
from phase8_paired_sampling import generate_random_pairs, calculate_turner_from_pairs, evaluate_qubo_from_pairs
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import csv
import os

def calculate_mirror_penalty(sequence, target_structure):
    """
    Implements the soft-mirror heuristic from the paper.
    Penalizes symmetric inward extensions into the loop with weight decay.
    """
    stems = extract_stems(target_structure)
    if not stems: return 0.0
    
    stem = stems[-1]
    left, right = stem[-1] # q_0 (terminal pair)
    
    # Calculate loop length L
    L = right - left - 1
    m = L // 2
    if m == 0: return 0.0
        
    penalty = 0.0
    prev_pair_str = sequence[left] + sequence[right]
    curr_l, curr_r = left + 1, right - 1
    
    for d in range(1, m + 1):
        curr_pair_str = sequence[curr_l] + sequence[curr_r]
        
        # Calculate weight w_d (lambda = 0.5)
        w_d = 1.0 - 0.5 * ((d - 1) / (m - 1)) if m > 1 else 1.0
            
        if curr_pair_str in ALLOWED_PAIRS and prev_pair_str in ALLOWED_PAIRS:
            # stack_energy is negative for stable stacks. phi = max(0, -E)
            stack_energy = get_turner_energy(prev_pair_str, curr_pair_str)
            phi = max(0.0, -stack_energy) 
            penalty += w_d * phi
            
        prev_pair_str = curr_pair_str
        curr_l += 1
        curr_r -= 1

    return penalty



def get_qubo_top_10_percent(target_structure , num_samples=5000, math_method="ols"):
    """
    Runs the phase 8 pipeline to get the top 10% sequences (lowest QUBO energy).
    Returns a list of pair combinations.
    """
    stems = extract_stems(target_structure)
    c_coeffs = calculate_qubo_coeffs(method=math_method)
    Q_dict , offset = build_approx_qubo(stems , c_coeffs)
    samples = generate_random_pairs(stems, num_samples)
    combined_results =[]

    for stem_pairs_list , flat_tuple in samples:
        t_energy = calculate_turner_from_pairs(stem_pairs_list)
        q_energy = evaluate_qubo_from_pairs(stem_pairs_list,stems,Q_dict,offset)
        combined_results.append({
            'pairs_list': flat_tuple,
            'true': t_energy,
            'qubo': q_energy
        })
    qubo_sorted = sorted(combined_results,key=lambda x:x['qubo'])
    return qubo_sorted[:10] 

def generate_full_sequence(target_structure, pair_assignment, num_samples=10):
    """
    Takes a target structure and a pair assignment (e.g., ['GC', 'AU', ...]),
    and generates `num_samples` full-length sequences by filling the 
    unpaired dots with random uniform nucleotides.
    """
    stems = extract_stems(target_structure)
    full_length = len(target_structure)

    # figure our which index in the string gets which base from the pairs 
    fixed_bases = {}
    pair_idx = 0
    for stem in stems:
        for (left,right) in stem:
            pair_string = pair_assignment[pair_idx]
            fixed_bases[left] = pair_string[0]
            fixed_bases[right] = pair_string[1]
            pair_idx +=1
    # Generate multiple full length variations by randomly filling the dots 

    bases = ['A','U','C','G']
    # Phase 1 Scale up: 1000 loop fills for aggressive filtering
    pool_size = 1000
    candidates = []

    for _ in range(pool_size):
        seq = []
        for i in range(full_length):
            if i in fixed_bases:
                seq.append(fixed_bases[i])
            else:
                seq.append(random.choice(bases))
        full_seq = "".join(seq)

        penalty = calculate_mirror_penalty(full_seq, target_structure)
        candidates.append((penalty, full_seq))
    
    candidates.sort(key=lambda x: x[0])
    return [seq for penalty, seq in candidates[:num_samples]]
        


def run_forward_folding_pipeline(target_structure ,initial_samples=5000 ,variations =10, math_method="ols" ):
    print(f"\n==== Phase 9: Forward Folding Validation ====")
    print(f"Target Structure: {target_structure}")
    
    print("\n1. Running QUBO to find Top 10% base-pair assignments...")
    top_10_qubo = get_qubo_top_10_percent(target_structure,initial_samples,math_method)
    print(f"Extracted {len(top_10_qubo)} top QUBO-predicted pair assignments.")
    
    succcess_count =0
    total_variation_tested = 0
    print(f"\n2. Generating {variations} uniform random loop variations for each assignment and forward folding...")

    for item in top_10_qubo:
        pair_assignment = item['pairs_list']
        full_seqs = generate_full_sequence(target_structure,pair_assignment,num_samples=variations)

        for seq in full_seqs:
            total_variation_tested +=1
            # forward fold using vienna fold 
            folded_structure , energy = RNA.fold(seq)
            if folded_structure == target_structure:
                succcess_count +=1
        
    succcess_rate = (succcess_count/total_variation_tested)*100 if total_variation_tested>0 else 0
    print("\n==== RESULTS ====")
    print(f"Total full-length sequences tested: {total_variation_tested}")
    print(f"Successful foldings back to target : {succcess_count}")
    print(f"Success Rate                       : {succcess_rate:.2f}%")
    return total_variation_tested, succcess_count, succcess_rate

def run_correlation(target_structure, num_samples=5000, math_method="ols"):
    print(f"\n==== Phase 9 Correlation (with Realistic Loops) ====")
    print(f"Target Structure: {target_structure}")
    
    stems = extract_stems(target_structure)
    
    print("1. Building QUBO model...")
    c_coeffs = calculate_qubo_coeffs(method=math_method)
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)
    
    print("2. Generating random pair combinations...")
    # Generate random stem pairings
    samples = generate_random_pairs(stems, num_samples)
    
    combined_results = []
    
    print("3. Generating full sequences and evaluating Vienna energies...")
    for stem_pairs_list, flat_tuple in samples:
        # 1. Score QUBO on the bare pair assignments
        q_energy = evaluate_qubo_from_pairs(stem_pairs_list, stems, Q_dict, offset)
        
        # 2. Generate the safest realistic full sequence (best of pool to avoid mirror pairings)
        # We only need 1 sample back since it picks the one with the lowest mirror penalty
        best_seqs = generate_full_sequence(target_structure, flat_tuple, num_samples=1)
        if not best_seqs:
            continue
        full_seq = best_seqs[0]
        
        # 3. Calculate Vienna energy on the target structure
        fc = RNA.fold_compound(full_seq)
        v_energy = fc.eval_structure(target_structure)
        
        combined_results.append({
            'seq': full_seq,
            'pairs': flat_tuple,
            'qubo': q_energy,
            'vienna': v_energy
        })

    # Sort by QUBO to mimic reading from a quantum computer
    qubo_sorted = sorted(combined_results, key=lambda x: x['qubo'])
    all_qubo = [x['qubo'] for x in qubo_sorted]
    all_vienna = [x['vienna'] for x in qubo_sorted]
    
    overall_corr, _ = spearmanr(all_vienna, all_qubo)
    
    ten_percent_idx = max(1, int(len(qubo_sorted) * 0.10))
    b10_qubo = all_qubo[:ten_percent_idx]
    b10_vienna = all_vienna[:ten_percent_idx]
    b10_corr, _ = spearmanr(b10_vienna, b10_qubo)
    
    print("\n--- RESULTS ---")
    print(f"Overall Spearman Correlation (Vienna vs QUBO): {overall_corr:.4f}")
    print(f"Bottom 10% Spearman Correlation (Vienna vs QUBO): {b10_corr:.4f}")
    
    # Plotting
    save_dir = "correlation_plots_N1000_K100"
    save_dir_terminal_outputs = "results/terminal_outputs_N1000_K100/"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_terminal_outputs, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(b10_vienna, b10_qubo, color='#2ca02c', alpha=0.7, s=50, edgecolor='black')
    
    # Trendline
    z_v = np.polyfit(b10_vienna, b10_qubo, 1)
    p_v = np.poly1d(z_v)
    plt.plot(b10_vienna, p_v(b10_vienna), "r--", alpha=0.8, label=f"Trendline")

    plt.title(f"Phase 9 (Realistic Loops): Lowest 10% QUBO vs Vienna ({math_method.upper()})\nSpearman Correlation: {b10_corr:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Vienna Energy [kcal/mol]", fontsize=12)
    plt.ylabel("QUBO Approximated Energy [kcal/mol]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Phase9_Vienna_Corr_{target_structure}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {save_dir}/Phase9_Vienna_Corr_{target_structure}.png")

    # Export full results to CSV
    csv_path = os.path.join(save_dir_terminal_outputs, f"Phase9_Results_{target_structure}.csv")
    if qubo_sorted:
        keys = qubo_sorted[0].keys()
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(qubo_sorted)
    print(f"Saved full results to {csv_path}")
    return overall_corr, b10_corr

if __name__ == "__main__":
    # Create the directory for the hairpin test results
    out_dir = "hairpin_rna_test_N1000_K100"
    os.makedirs(out_dir, exist_ok=True)
    # Use a different filename so we don't overwrite the Annealer results!
    summary_csv = os.path.join(out_dir, "summary_results_baseline.csv")    
    # Using 'w' to overwrite/create the summary file
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Target Structure", "Length", "Stems", 
            "Total Variations Tested", "Successful Folds", "Success Rate (%)",
            "Spearman Corr Overall", "Spearman Corr Bottom 10%"
        ])
        
        # 1. First, process the original structures
        target_list = ["..............(((((.....)))))", "(((...)))", "...((((((.........))))))."]
        for i in target_list:
            print(f"\n\n{'='*50}\nTesting ORIGINAL structure: {i}\n{'='*50}")
            # original settings (1000 initial samples, 10 variations)
            tot_vars, succ_count, succ_rate = run_forward_folding_pipeline(i, initial_samples=1000, variations=10)
            overall_corr, b10_corr = run_correlation(i, num_samples=1000)
            
            writer.writerow([
                i, len(i), i.count('('),
                tot_vars, succ_count, f"{succ_rate:.2f}",
                f"{overall_corr:.4f}", f"{b10_corr:.4f}"
            ])
            f.flush()

        # 2. Phase 1 Scale Up: Test lengthy Hairpin targets (Stems 8-15, Loops 4-10)
        target_list_hairpin_rna = []
        for s in range(8, 16):
            for L in [4, 5, 6, 7, 8, 9, 10]:
                target_list_hairpin_rna.append("(" * s + "." * L + ")" * s)
        
        for i in target_list_hairpin_rna:
            print(f"\n\n{'='*50}\nTesting ADDITIONAL structure: {i} (Stems: {i.count('(')})\n{'='*50}")
            # Using 20 variations as requested for the additional tests
            tot_vars, succ_count, succ_rate = run_forward_folding_pipeline(i, initial_samples=1000, variations=10)
            overall_corr, b10_corr = run_correlation(i, num_samples=1000)
            
            writer.writerow([
                i, len(i), i.count('('),
                tot_vars, succ_count, f"{succ_rate:.2f}",
                f"{overall_corr:.4f}", f"{b10_corr:.4f}"
            ])
            f.flush()
            
    print(f"\nAll summary results successfully saved to {summary_csv}")
