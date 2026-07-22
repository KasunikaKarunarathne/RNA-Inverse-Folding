import os 
import random
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import spearmanr

# Import your existing modules!
from phase1_rules import extract_stems, PAIR_ENCODING, ALLOWED_PAIRS
from phase2_turner_energy import get_turner_energy
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo

def generate_random_pairs(stems, num_samples):
    """
    Generates random pair assignments ONLY for the paired positions.
    Returns a list of samples. Each sample is a list of stems, 
    where each stem is a list of pair strings (e.g., 'GC').
    """
    samples = []
    seen = set()
    
    # Calculate total possible combinations to prevent infinite loops
    total_pairs = sum(len(stem) for stem in stems)
    max_combinations = 6 ** total_pairs
    num_samples = min(num_samples, max_combinations)

    while len(samples) < num_samples:
        current_sample = []
        tuple_repr = []
        
        # Only assign pairs to the stems, completely ignoring loops
        for stem in stems:
            stem_pairs = []
            for _ in range(len(stem)):
                pair = random.choice(ALLOWED_PAIRS)
                stem_pairs.append(pair)
                tuple_repr.append(pair)
            current_sample.append(stem_pairs)
        
        flat_tuple = tuple(tuple_repr)
        
        # Check for duplicates using a flat tuple
        if flat_tuple not in seen:
            seen.add(flat_tuple)
            # Store both the nested list (for easy evaluation) and flat tuple (for string representation)
            samples.append((current_sample, flat_tuple))
            
    return samples

def calculate_turner_from_pairs(stem_pairs_list):
    """
    Calculates the exact 6-local biological energy directly from pair assignments.
    (Bypasses full string slicing entirely)
    """
    total_energy = 0.0
    for stem_pairs in stem_pairs_list:
        for k in range(len(stem_pairs)-1):
            p1_str = stem_pairs[k]
            p2_str = stem_pairs[k+1]
            total_energy += get_turner_energy(p1_str, p2_str)
    return total_energy

def evaluate_qubo_from_pairs(stem_pairs_list, stems, Q_dict, offset):
    """
    Calculates QUBO energy directly from pair assignments.
    """
    active_vars = set()
    for stem, pairs in zip(stems, stem_pairs_list):
        for (left, right), pair_string in zip(stem, pairs):
            bits = PAIR_ENCODING[pair_string]
            # Map directly to the QUBO variables
            if bits[0] == 1: active_vars.add(f"p_{left}_{right}_0")
            if bits[1] == 1: active_vars.add(f"p_{left}_{right}_1")
            if bits[2] == 1: active_vars.add(f"p_{left}_{right}_2")

    qubo_energy = offset
    for (var1, var2), weight in Q_dict.items():
        if var1 in active_vars and var2 in active_vars:
            qubo_energy += weight 

    return qubo_energy

def run_paired_sampling(target_structure, num_samples, math_method="ols"):
    print(f"\n---- Phase 8 Paired Sampling: {target_structure} ----")
    print(f"Mathematical method used : {math_method}")

    stems = extract_stems(target_structure)
    total_pairs = sum(len(stem) for stem in stems)
    print(f"Number of paired positions (base pairs): {total_pairs}")

    # 1. Generate the coeffs and build the matrix
    c_coeffs = calculate_qubo_coeffs(method=math_method)
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)

    # 2. Generate random sequences focusing ONLY on paired bases
    samples = generate_random_pairs(stems, num_samples)
    print(f"Generated {len(samples)} unique valid pair combinations out of {6**total_pairs} possible.")

    combined_results = []
    
    # 3. Score every sequence combination
    print("Scoring pair combinations with HUBO and QUBO...")
    for stem_pairs_list, flat_tuple in samples:
        t_energy = calculate_turner_from_pairs(stem_pairs_list)
        q_energy = evaluate_qubo_from_pairs(stem_pairs_list, stems, Q_dict, offset)
        
        # Our sequence is now represented as a string of concatenated pairs! (e.g. GC-AU-CG-UA-GU)
        seq_str = "-".join(flat_tuple) 
        combined_results.append({'seq': seq_str, 'true': t_energy, 'qubo': q_energy})

    # 4. Statistical Analysis
    all_true = [x['true'] for x in combined_results]
    all_qubo = [x['qubo'] for x in combined_results]
    correlation, _ = spearmanr(all_true, all_qubo)
    
    print("\n--- RESULTS ---")
    print(f"Overall Full Landscape Correlation : {correlation: .4f}")

    # Zoom in to bottom 10%
    qubo_sorted = sorted(combined_results, key=lambda x: x['qubo'])
    true_sorted = sorted(combined_results, key=lambda x: x['true'])
    
    ten_percent_idx = int(len(qubo_sorted) * 0.10)
    bottom_10_qubo = qubo_sorted[:ten_percent_idx]
    
    b10_true = [x['true'] for x in bottom_10_qubo]
    b10_qubo = [x['qubo'] for x in bottom_10_qubo]
    b10_corr, _ = spearmanr(b10_true, b10_qubo)
    
    print(f"Spearman Correlation (Lowest 10% of QUBO): {b10_corr:.4f}")

    bottom_10_qubo_seq = set([x['seq'] for x in qubo_sorted[:ten_percent_idx]])
    bottom_10_true_seq = set([x['seq'] for x in true_sorted[:ten_percent_idx]])
    overlap = bottom_10_qubo_seq.intersection(bottom_10_true_seq)
    overlap_pct = (len(overlap)/ten_percent_idx)*100
    
    print(f"Tail Overlap Percentage (Top 10% Match): {overlap_pct:.1f}%")

if __name__ == "__main__":
    # Test on a few structures
    target_list = ["..............(((((.....)))))", "(((...)))", "...((((((.........))))))."]
    for i in target_list:
        run_paired_sampling(i, num_samples=100, math_method="ols")
