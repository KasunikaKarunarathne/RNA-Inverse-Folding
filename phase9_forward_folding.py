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

def calculate_mirror_penalty(sequence , target_structure):
    """
    Calculates a positive penalty for any mirror-symmetry dots that accidentally form valid pairs.
    It extends stems outwards and inwards. If a valid pair is found, it adds the positive
    Turner stacking energy to the penalty.
    """
    stems = extract_stems(target_structure)
    unpaired = set(i for i,c in enumerate(target_structure) if c ==".")
    penalty = 0.0

    for stem in stems:
        # check outward extensions
        left,right = stem[0]
        prev_pair_str = sequence[left]+ sequence[right]
        curr_l,curr_r = left-1,right+1 
        while curr_l in unpaired and curr_r in unpaired:
            curr_pair_str = sequence[curr_l]+sequence[curr_r]
            if curr_pair_str in ALLOWED_PAIRS:
                # It formed a valid pair! calculate the Truner stack energy 
                stack_energy = get_turner_energy(curr_pair_str,prev_pair_str)
                # add the positive abs value as a penalty
                penalty += abs(stack_energy)
                # update prev_pair for the next outward step
                prev_pair_str = curr_pair_str
            else:
                # if the chain breaks we stop penalizing this extension
                break
            curr_l -=1 
            curr_r+=1

            # check the inwards extensions
            left,right = stem[-1]
            prev_pair_str = sequence[left] + sequence[right]
            curr_l,curr_r = left+1,right-1

            while curr_l in unpaired and curr_r in unpaired:
                curr_pair_str = sequence[curr_l] + sequence[curr_r]
                if curr_pair_str in ALLOWED_PAIRS:
                    stack_energy = get_turner_energy(prev_pair_str,curr_pair_str)
                    penalty += abs(stack_energy)
                    prev_pair_str = curr_pair_str
                else:
                    break
                curr_l +=1
                curr_r -=1

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
    ten_percent_idx = int(len(qubo_sorted)*0.10)
    return qubo_sorted[:ten_percent_idx] 

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

    pool_size = num_samples *20
    candidates =[]

    for _ in range(pool_size):
        seq =[]
        for i in range(full_length):
            if i in fixed_bases:
                seq.append(fixed_bases[i])
            else:
                seq.append(random.choice(bases))
        full_seq = "".join(seq)

        # calculate how badly this random sequeces violate the mirror symmetry rules 
        penalty = calculate_mirror_penalty(full_seq , target_structure)
        candidates.append((penalty, full_seq))
    
    # sort by the lowest penalty
    candidates.sort(key=lambda x : x[0])
    # take only the safest lowest penalty sequences
    best_sequences = [seq for penalty,seq in candidates[:num_samples]]
    return best_sequences
        


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

def run_correlation(target_structure, num_samples=1000, math_method="ols"):
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
        # Calculate Deviations (Vienna - QUBO)
    errors = []
    for res in qubo_sorted:
        error = res['vienna'] - res['qubo']
        res['error'] = error
        errors.append(error)
        
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print("\n--- DEVIATION ANALYSIS (Vienna - QUBO) ---")
    print(f"Mean Deviation (Average Loop Penalty) : {mean_error:.4f} kcal/mol")
    print(f"Standard Deviation of Error (Spread)  : {std_error:.4f} kcal/mol")
    print(f"Min Deviation: {min_error:.4f} | Max Deviation: {max_error:.4f}")
    
    print("\n[Interpretation]")
    if std_error < 1.5:
        print("-> EXCELLENT: The standard deviation is low! The loop penalties are consistent, meaning QUBO preserves the correct ranking of sequences.")
    elif std_error < 3.0:
        print("-> OKAY: There is moderate variance in loop penalties. QUBO rankings might be slightly scrambled compared to Vienna, but still a useful approximation.")
    else:
        print("-> POOR: The standard deviation is high! The loop energies vary so wildly that the QUBO model's predictions are getting heavily scrambled.")
 
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
    save_dir = "correlation_plots"
    os.makedirs(save_dir, exist_ok=True)
    
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
    # plot of deviations
        # PLOT 2: The Deviation Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}')
    
    plt.title(f"Deviation Spread (Vienna - QUBO)\nStd Dev: {std_error:.2f} kcal/mol", fontsize=14, fontweight='bold')
    plt.xlabel("Energy Difference (kcal/mol)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Phase9_Deviation_{target_structure}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.close()
    
    print(f"Saved plot to {save_dir}/Phase9_Vienna_Corr_{target_structure}.png")

if __name__ == "__main__":
    target_list = ["..............(((((.....)))))", "(((...)))", "...((((((.........))))))."]
    for i in target_list:
        run_forward_folding_pipeline(i, initial_samples=1000, variations=10)
        run_correlation(i, num_samples=1000)
