import os
import re
import pandas as pd
import RNA
from collections import defaultdict
import dimod
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib.pyplot as plt
import random

# Import the brilliant, efficient components from Phase 17
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase17_extended import calculate_structural_difficulty, decode_extended_sample, generate_sequence
from phase18_unified_factorial import evaluate_vienna_ensemble

def build_qamoo_extended_qubo(target_structure, stems, c_coeffs, alpha=0.5):
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)
    Q_ext = defaultdict(float, Q_dict)
    
    # Scale positive design by (1 - alpha)
    for key in Q_ext.keys():
        Q_ext[key] *= (1.0 - alpha)
        
    difficulty = calculate_structural_difficulty(target_structure, stems)
    
    tetraloop_stems = []
    fixed_loops = {}
    for match in re.finditer(r'\(\.\.\.\.\)', target_structure):
        close_left = match.start()
        close_right = match.end() - 1
        tetraloop_stems.append((close_left, close_right))
        
        idx = match.start() + 1
        tetraloop_choice = "UUCG" 
        for k in range(4):
            fixed_loops[idx+k] = tetraloop_choice[k]
            
    for (left, right) in tetraloop_stems:
        P_force = 100.0 * (1.0 - alpha)
        Q_ext[(f"p_{left}_{right}_0", f"p_{left}_{right}_0")] -= P_force
        Q_ext[(f"p_{left}_{right}_1", f"p_{left}_{right}_1")] += P_force
        Q_ext[(f"p_{left}_{right}_2", f"p_{left}_{right}_2")] -= P_force

    paired_indices = set()
    for stem in stems:
        for (left, right) in stem:
            paired_indices.add(left)
            paired_indices.add(right)
            
    for stem in stems:
        if len(stem) == 1:
            l, r = stem[0]
            Q_ext[(f"p_{l}_{r}_0", f"p_{l}_{r}_0")] -= 1.0
            Q_ext[(f"p_{l}_{r}_1", f"p_{l}_{r}_1")] += 1.0
            Q_ext[(f"p_{l}_{r}_2", f"p_{l}_{r}_2")] -= 1.0
            
    loop_indices = [i for i in range(len(target_structure)) if i not in paired_indices and i not in fixed_loops]
    bases = ["A", "C", "G", "U"]
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]

    P_anti = 50.0 * difficulty * alpha
    P_mirror = 100.0 * difficulty * alpha
    
    bias_tax = {"A": 0.0, "C": 5.0, "G": 15.0, "U": 5.0} # change according to desirna maybe 
    for idx in loop_indices:
        for b in bases:
            Q_ext[(f"x_{idx}_{b}", f"x_{idx}_{b}")] += bias_tax[b] * alpha

    for i in range(len(loop_indices)):
        for j in range(i+1, len(loop_indices)):
            idx1 = loop_indices[i]
            idx2 = loop_indices[j]
            distance = abs(idx1 - idx2)
            
            if distance >= 4:
                probabilistic_weight = P_anti * (4.0 / distance)
                has_inner = (idx1 + 1 in loop_indices) and (idx2 - 1 in loop_indices)
                has_outer = (idx1 - 1 in loop_indices) and (idx2 + 1 in loop_indices)
                penalty = probabilistic_weight * (2.0 if not (has_inner or has_outer) else 1.0)
                
                for b1, b2 in valid_pairs:
                    var1, var2 = f"x_{idx1}_{b1}", f"x_{idx2}_{b2}"
                    key = tuple(sorted([var1, var2]))
                    Q_ext[key] += penalty
                    
    for stem in stems:
        left, right = stem[-1]
        L = right - left - 1
        m = L // 2
        curr_l, curr_r = left + 1, right - 1
        
        for d in range(1, m + 1):
            if curr_l >= curr_r: break
            if curr_l in loop_indices and curr_r in loop_indices:
                w_d = 1.0 - 0.5 * ((d - 1) / (m - 1)) if m > 1 else 1.0
                for b1, b2 in valid_pairs:
                    var1, var2 = f"x_{curr_l}_{b1}", f"x_{curr_r}_{b2}"
                    key = tuple(sorted([var1, var2]))
                    Q_ext[key] += P_mirror * w_d
            curr_l += 1
            curr_r -= 1

    P_onehot = 100.0 * max(1.0, difficulty * 0.5)
    for idx in loop_indices:
        for b in bases:
            Q_ext[(f"x_{idx}_{b}", f"x_{idx}_{b}")] -= P_onehot
        for i in range(4):
            for j in range(i+1, 4):
                var1, var2 = f"x_{idx}_{bases[i]}", f"x_{idx}_{bases[j]}"
                key = tuple(sorted([var1, var2]))
                Q_ext[key] += 2 * P_onehot

    return dict(Q_ext), loop_indices, fixed_loops

def classical_polish(seq, target_structure, stems, loop_indices, max_steps=100):
    """
    Takes a Quantum-Optimized seed sequence and runs a fast greedy local search
    to polish out any 3-body edge cases (like Dangling Ends on 2-bp stems).
    """
    current_seq = list(seq)
    mfe_struct, _ = RNA.fold("".join(current_seq))
    current_dist = RNA.bp_distance(mfe_struct, target_structure)
    
    if current_dist == 0:
        return "".join(current_seq), 0, True # Already perfect!
        
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]
    bases = ["A", "C", "G", "U"]
    
    stem_pairs = []
    for stem in stems:
        for (l, r) in stem:
            stem_pairs.append((l, r))
            
    for step in range(max_steps):
        mutated_seq = current_seq.copy()
        
        # Randomly mutate a stem or loop
        if random.random() < 0.5 and stem_pairs:
            l, r = random.choice(stem_pairs)
            new_pair = random.choice(valid_pairs)
            mutated_seq[l] = new_pair[0]
            mutated_seq[r] = new_pair[1]
        elif loop_indices:
            idx = random.choice(loop_indices)
            mutated_seq[idx] = random.choice(bases)
            
        new_seq_str = "".join(mutated_seq)
        new_struct, _ = RNA.fold(new_seq_str)
        new_dist = RNA.bp_distance(new_struct, target_structure)
        
        # Accept if it's strictly better OR same distance (to escape flat plateaus)
        if new_dist <= current_dist:
            current_seq = mutated_seq
            current_dist = new_dist
            
        if current_dist == 0:
            return "".join(current_seq), step + 1, True
            
    return "".join(current_seq), max_steps, False

def run_hybrid_sweep(target_structure, c_coeffs, target_name="Structure"):
    sampler = SimulatedAnnealingSampler()
    stems = extract_stems(target_structure)
    
    successes, total_defect, total_diversity = 0, 0.0, 0.0
    plot_alphas, plot_mfes, plot_defects = [], [], []
    
    for i in range(11):
        alpha = i / 10.0
        print(f"\n--- Running Alpha = {alpha:.1f} ---")
        
        Q_ext, loop_indices, fixed_loops = build_qamoo_extended_qubo(target_structure, stems, c_coeffs, alpha=alpha)
        sampleset = sampler.sample_qubo(Q_ext, num_reads=3000)
        
        # Evaluate the top 5 unique quantum samples
        best_polished_seq = None
        best_metrics = None
        best_defect = float('inf')
        
        unique_samples = []
        for sample in sampleset.samples():
            decoded = decode_extended_sample(dict(sample), stems, loop_indices, fixed_loops)
            if decoded and decoded not in unique_samples:
                unique_samples.append(decoded)
            if len(unique_samples) >= 5:
                break
                
        for decoded in unique_samples:
            pairs_list, loop_assignment = decoded
            quantum_seed_seq = generate_sequence(target_structure, stems, pairs_list, loop_assignment)
            
            # THE CLASSICAL POLISH STEP!
            polished_seq, steps_taken, polished_success = classical_polish(
                quantum_seed_seq, target_structure, stems, loop_indices, max_steps=200
            )
            
            if polished_success:
                metrics = evaluate_vienna_ensemble(polished_seq, target_structure)
                if metrics["ensemble_defect"] < best_defect:
                    best_defect = metrics["ensemble_defect"]
                    best_metrics = metrics
                    best_polished_seq = polished_seq
                    
        if best_metrics:
            successes += 1
            total_defect += best_metrics["ensemble_defect"]
            total_diversity += best_metrics["diversity"]
            plot_alphas.append(alpha)
            plot_mfes.append(best_metrics["mfe"])
            plot_defects.append(best_metrics["ensemble_defect"])
            print(f"SUCCESS (Hybrid) | MFE: {best_metrics['mfe']:.2f} | Defect: {best_metrics['ensemble_defect']:.2f} | Seq: {best_polished_seq}")
        else:
            print(f"FAILED TO FOLD (Even after Hybrid Polish)")

    # Plotting
    if plot_alphas:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 5))
        sc = plt.scatter(plot_mfes, plot_defects, c=plot_alphas, cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(sc, label='Alpha')
        plt.title(f"Hybrid QAMOO Pareto: {target_name}")
        plt.xlabel("MFE (Lower is Better)")
        plt.ylabel("Ensemble Defect (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', target_name).strip()
        plt.savefig(f"plots/hybrid_qamoo_pareto_{safe_name}.png")
        plt.close()

    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return 11, successes, avg_defect, avg_div

if __name__ == "__main__":
    print("Fitting OLS coefficients for Turner energy...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    df = pd.read_csv("Structures/fmqa_paper_structures.csv")
    print(f"Loaded {len(df)} structures from benchmark dataset.")
    
    # Test ALL structures in the benchmark
    for _, row in df.iterrows():
        name = row['Category']
        struct = row['Structure']
        print(f"\n==============================================")
        print(f"BENCHMARKING: {name}")
        print(f"STRUCTURE: {struct}")
        print(f"==============================================")
        
        tot, succ, avg_def, avg_div = run_hybrid_sweep(struct, c_coeffs, target_name=name)
        print(f"\n[FINAL] {name} -> Success Rate: {(succ/tot)*100:.1f}% | Avg Defect: {avg_def:.2f}")
