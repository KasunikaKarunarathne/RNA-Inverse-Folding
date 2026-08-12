import os
import random
import numpy as np
import matplotlib.pyplot as plt
import RNA
import neal

from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase10_annealin import (
    decode_bits_to_pairs,
    calculate_mirror_penalty, 
    calculate_loop_entropy_penalty
)

def get_qubo_top_10_percent(target_structure,num_samples=5000,c_coeffs=None):
    """
    Runs Simulated Annealing to sample the QUBO landscape and returns the Top 10%.
    """
    stems = extract_stems(target_structure)
    Q_dict,offset = build_approx_qubo(stems,c_coeffs)

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q_dict , num_reads=num_samples)

    combined_results =[]
    for sample, energy in sampleset.data(['sample','energy']):
        pairs_list = decode_bits_to_pairs(sample,stems)
        if pairs_list is not None:
            combined_results.append({
                'pairs_list':pairs_list,
                'qubo':energy+offset
            })

    qubo_sorted = sorted(combined_results,key=lambda x:x['qubo'])
    ten_percent_idx = max(1,int(len(qubo_sorted)*0.10))
    return qubo_sorted[:ten_percent_idx]

def generate_full_sequence_parameterized(target_structure, pair_assignment, K=10, N=10000):
    """
    Generates N random sequence fills for the unpaired dots, scores them by the penalties, 
    and returns the top K variations (the candidate set).
    """
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
    candidates = []
     # Generate a pool of N random variations
    for _ in range(N):
        seq = []
        for i in range(full_length):
            if i in fixed_bases:
                seq.append(fixed_bases[i])
            else:
                seq.append(random.choice(bases))
        full_seq = "".join(seq)

        # Apply phase 10 penalties
        penalty = calculate_mirror_penalty(full_seq, target_structure)
        penalty += calculate_loop_entropy_penalty(full_seq, target_structure)
        candidates.append((penalty, full_seq))
    
    # Sort by penalty and return the top K
    candidates.sort(key=lambda x: x[0])
    return [seq for penalty, seq in candidates[:K]]

def run_n_k_analysis(target_structure, N_list, K_list, c_coeffs, initial_samples=1000):
    print(f"\n==== Running N and K Analysis for: {target_structure} ====")
    
    # 1. Get the Top 10% QUBO pair assignments ONCE for this target
    top_10_qubo = get_qubo_top_10_percent(target_structure, num_samples=initial_samples, c_coeffs=c_coeffs)
    total_assignments = len(top_10_qubo)
    
    print(f"Extracted {total_assignments} top 10% QUBO-predicted pair assignments.")
    
    # Matrix to store results for our heatmap (rows=K, cols=N)
    results_matrix = np.zeros((len(K_list), len(N_list)))
    
    for i, K in enumerate(K_list):
        for j, N in enumerate(N_list):
            
            # If K is accidentally larger than N, we just test N
            actual_K = min(K, N)
            successful_assignments = 0
            total_successful_sequences = 0
            total_sequences_tested = 0
            
            for item in top_10_qubo:
                pair_assignment = item['pairs_list']
                
                # Get the K best sequences from a pool of size N
                full_seqs = generate_full_sequence_parameterized(
                    target_structure, pair_assignment, K=actual_K, N=N
                )
                
                found_success = False
                for seq in full_seqs:
                    total_sequences_tested += 1
                    folded_structure, _ = RNA.fold(seq)
                    if folded_structure == target_structure:
                        found_success = True
                        total_successful_sequences += 1
                        
                if found_success:
                    successful_assignments += 1
            
            success_rate_at_least_1 = (successful_assignments / total_assignments) * 100 if total_assignments > 0 else 0
            overall_success_rate = (total_successful_sequences / total_sequences_tested) * 100 if total_sequences_tested > 0 else 0
            
            results_matrix[i, j] = success_rate_at_least_1
            
            print(f"Pool(N)={N:5d}, Candidates(K)={K:2d} | At-Least-1: {success_rate_at_least_1:5.1f}% | Overall: {overall_success_rate:5.1f}%")
            
    return results_matrix



def plot_n_k_heatmap(target_structure, N_list, K_list, results_matrix, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the matrix as a heatmap
    cax = ax.imshow(results_matrix, cmap="YlGnBu", aspect="auto")
    
    # Setup the axes ticks
    ax.set_xticks(np.arange(len(N_list)))
    ax.set_yticks(np.arange(len(K_list)))
    ax.set_xticklabels(N_list)
    ax.set_yticklabels(K_list)
    
    # Print the success rate text on each block of the heatmap
    for i in range(len(K_list)):
        for j in range(len(N_list)):
            text_color = "black" if results_matrix[i, j] < 50 else "white"
            ax.text(j, i, f"{results_matrix[i, j]:.1f}",
                    ha="center", va="center", color=text_color, fontweight='bold')
                           
    ax.set_title(f"Success Rate (%) vs Pool Size (N) & Candidates (K)\nTarget: {target_structure}")
    fig.colorbar(cax, label="Success Rate (%)")
    ax.set_xlabel("Pool Size (N)")
    ax.set_ylabel("Candidate Set Size (K)")
    
    # Create a safe filename (replace dots and brackets)
    safe_target = target_structure.replace(".", "d").replace("(", "L").replace(")", "R")
    
    save_path = os.path.join(save_dir, f"NK_Analysis_{safe_target}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved heatmap visualization to {save_path}")


if __name__ == "__main__":
    # Define our test grid! 
    # N = Pool Size (how many random fills we generate to pick from)
    # K = Candidate Set (how many of the best ones we actually test with Vienna Fold)
    N_list = [50, 100, 500, 1000, 5000, 10000]
    K_list = [1, 5, 10, 20, 50]
    
    print("=========================================================================")
    print("Pre-computing QUBO regression coefficients ONCE for all tests...")
    global_coeffs = calculate_qubo_coeffs(method="quantum_anneal")
    print("=========================================================================")
    
    # A simple hairpin target containing a loop of length 10
    test_targets = [
        "((((((((((..........))))))))))",
        "..............(((((.....)))))"
    ]
    
    for target in test_targets:
        results = run_n_k_analysis(
            target_structure=target, 
            N_list=N_list, 
            K_list=K_list, 
            c_coeffs=global_coeffs, 
            initial_samples=1000  # How many QUBO predictions to sample before taking Top 10%
        )
        
        plot_n_k_heatmap(target, N_list, K_list, results)
