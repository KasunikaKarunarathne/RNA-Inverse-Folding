import os
import csv
import random
import RNA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Reusing functions from earlier phases! 
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase8_paired_sampling import generate_random_pairs, evaluate_qubo_from_pairs
from phase9_forward_folding import calculate_mirror_penalty
from phase10_annealin import calculate_loop_entropy_penalty
from phase11_penalty import calculate_shifted_mirror_penalty, get_qubo_top_10_percent

def generate_full_sequence_with_strategy(target_structure, pair_assignment, penalty_strategy, num_samples=10):
    stems = extract_stems(target_structure)
    full_length = len(target_structure)

    fixed_bases = {}
    pair_idx = 0
    for stem in stems:
        for (left, right) in stem:
            pair_string = pair_assignment[pair_idx]
            fixed_bases[left] = pair_string[0]
            fixed_bases[right] = pair_string[1]
            pair_idx += 1

    bases = ['A', 'U', 'C', 'G']
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

        penalty = 0.0
        if penalty_strategy == "Baseline":
            penalty = 0.0
        elif penalty_strategy == "Mirror":
            penalty = calculate_mirror_penalty(full_seq, target_structure)
        elif penalty_strategy == "Entropy":
            penalty = calculate_loop_entropy_penalty(full_seq, target_structure)
        elif penalty_strategy == "Mirror + Entropy":
            penalty = calculate_mirror_penalty(full_seq, target_structure) + \
                      calculate_loop_entropy_penalty(full_seq, target_structure)
        elif penalty_strategy == "Shifted Mirror":
            penalty = calculate_shifted_mirror_penalty(full_seq, target_structure, max_shift=2)
        elif penalty_strategy == "Shifted Mirror + Entropy":
            penalty = calculate_shifted_mirror_penalty(full_seq, target_structure, max_shift=2) + \
                      calculate_loop_entropy_penalty(full_seq, target_structure)

        candidates.append((penalty, full_seq))
    
    candidates.sort(key=lambda x: x[0])
    return [seq for penalty, seq in candidates[:num_samples]]

def run_forward_folding_for_strategy(target_structure, penalty_strategy, top_10_qubo, variations=10):
    success_count = 0
    total_variation_tested = 0

    for item in top_10_qubo:
        pair_assignment = item['pairs_list']
        full_seqs = generate_full_sequence_with_strategy(
            target_structure, pair_assignment, penalty_strategy, num_samples=variations
        )

        for seq in full_seqs:
            total_variation_tested += 1
            folded_structure, _ = RNA.fold(seq)
            if folded_structure == target_structure:
                success_count += 1
        
    success_rate = (success_count / total_variation_tested) * 100 if total_variation_tested > 0 else 0
    return total_variation_tested, success_count, success_rate

def run_correlation_for_strategy(target_structure, penalty_strategy, stems, Q_dict, offset, random_samples):
    combined_results = []
    
    for stem_pairs_list, flat_tuple in random_samples:
        q_energy = evaluate_qubo_from_pairs(stem_pairs_list, stems, Q_dict, offset)
        
        # We only need 1 sample back for correlation since it picks the lowest penalty
        best_seqs = generate_full_sequence_with_strategy(
            target_structure, flat_tuple, penalty_strategy, num_samples=1
        )
        if not best_seqs: continue
        full_seq = best_seqs[0]
        
        fc = RNA.fold_compound(full_seq)
        v_energy = fc.eval_structure(target_structure)
        
        combined_results.append({
            'qubo': q_energy,
            'vienna': v_energy
        })

    qubo_sorted = sorted(combined_results, key=lambda x: x['qubo'])
    all_qubo = [x['qubo'] for x in qubo_sorted]
    all_vienna = [x['vienna'] for x in qubo_sorted]
    
    if len(all_qubo) < 2: return 0.0, 0.0
    
    overall_corr, _ = spearmanr(all_vienna, all_qubo)
    
    ten_percent_idx = max(2, int(len(qubo_sorted) * 0.10))
    b10_qubo = all_qubo[:ten_percent_idx]
    b10_vienna = all_vienna[:ten_percent_idx]
    b10_corr, _ = spearmanr(b10_vienna, b10_qubo)
    
    return overall_corr, b10_corr

def evaluate_all_strategies(target_structures, initial_samples=1000, variations=10, num_corr_samples=1000):
    strategies = [
        "Baseline",
        "Mirror",
        "Entropy",
        "Mirror + Entropy",
        "Shifted Mirror",
        "Shifted Mirror + Entropy"
    ]
    
    out_dir = "results/phase14_penalty_evaluation"
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "penalty_comparison.csv")
    
    results_succ = {strat: [] for strat in strategies}
    
    print("Pre-computing QUBO coefficients...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Target Structure", "Length", "Stems", "Penalty Strategy", 
            "Total Tested", "Successful Folds", "Success Rate (%)",
            "Spearman Corr Overall", "Spearman Corr B10%"
        ])
        
        for struct in target_structures:
            print(f"\n{'='*70}\nEvaluating Structure: {struct}\n{'='*70}")
            stems = extract_stems(struct)
            Q_dict, offset = build_approx_qubo(stems, c_coeffs)
            
            # Step 1: Run Generation once per structure to keep playing field perfectly level
            print("1. Running Simulated Annealing for Forward Folding...")
            top_10_qubo = get_qubo_top_10_percent(struct, num_samples=initial_samples, math_method="ols")
            
            print(f"2. Generating {num_corr_samples} random pair assignments for Correlation...")
            random_corr_samples = generate_random_pairs(stems, num_corr_samples)
            
            print("\n3. Testing Strategies (Success Rate & Spearman Correlation):")
            for strat in strategies:
                # Forward Folding
                tot_vars, succ_count, succ_rate = run_forward_folding_for_strategy(
                    struct, strat, top_10_qubo, variations=variations
                )
                # Correlation
                ov_corr, b10_corr = run_correlation_for_strategy(
                    struct, strat, stems, Q_dict, offset, random_corr_samples
                )
                
                print(f"   -> {strat:<28} | Success: {succ_rate:5.1f}% | Corr(All): {ov_corr:6.3f} | Corr(B10%): {b10_corr:6.3f}")
                
                results_succ[strat].append(succ_rate)
                writer.writerow([
                    struct, len(struct), struct.count('('), strat, 
                    tot_vars, succ_count, f"{succ_rate:.2f}", 
                    f"{ov_corr:.4f}", f"{b10_corr:.4f}"
                ])
                f.flush()
                
    # Generate grouped bar chart for Success Rate
    print("\nGenerating Success Rate visualization...")
    x = np.arange(len(target_structures))
    width = 0.12 
    fig, ax = plt.subplots(figsize=(14, 8))
    
    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, strat in enumerate(strategies):
        ax.bar(x + offsets[i] * width, results_succ[strat], width, label=strat, color=colors[i])
        
    ax.set_ylabel('Forward Folding Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Loop-Filling Penalty Strategies (Phases 9-13)', fontsize=14, fontweight='bold')
    
    labels = [f"L={len(s)}\n{s[:8]}..." if len(s)>15 else s for s in target_structures]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(title="Penalty Strategy")
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "penalty_strategies_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Data saved to {summary_csv}")
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    test_structures = [
        # 1. Simple hairpin (Baseline control)
        "((((((((..........))))))))",
        
        # 2. HIV-1 TAR-like element (Stem with a bulge)
        "(((((...(((((......)))))...)))))",
        
        # 3. Pre-miRNA-like element (Long stem with large internal loop)
        "(((((((...((((((.........))))))....)))))))",
        
        # 4. Three-way junction (Y-shape)
        "((((...((((....))))...((((....))))...))))",
        
        # 5. Fragmented stem (Multiple small internal loops / bulges)
        "(((..(((...(((...(((....)))...)))...)))..)))",
        
        # 6. tRNA-like Cloverleaf (4-way junction, highly complex)
        "(((((((..((((......)))).(((((.......))))).....(((((.......))))))))))))"
    ]
    
    evaluate_all_strategies(test_structures, initial_samples=1000, variations=10, num_corr_samples=1000)
