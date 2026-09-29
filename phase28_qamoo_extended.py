import os
import re
import pandas as pd
import RNA
from collections import defaultdict
import dimod
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib.pyplot as plt

# Import the brilliant, efficient components from Phase 17
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase17_extended import calculate_structural_difficulty, decode_extended_sample, generate_sequence
from phase18_unified_factorial import evaluate_vienna_ensemble

def build_qamoo_extended_qubo(target_structure, stems, c_coeffs, alpha=0.5):
    # 1. STEM QUBO (3-Qubit Encoding, Positive Design)
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)
    Q_ext = defaultdict(float, Q_dict)
    
    # Scale positive design by (1 - alpha)
    for key in Q_ext.keys():
        Q_ext[key] *= (1.0 - alpha)
        
    difficulty = calculate_structural_difficulty(target_structure, stems)
    
    # 2. TETRALOOPS (Positive Design)
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
        # Force C-G closing pair (p0=1, p1=0, p2=1)
        P_force = 100.0 * (1.0 - alpha)
        Q_ext[(f"p_{left}_{right}_0", f"p_{left}_{right}_0")] -= P_force
        Q_ext[(f"p_{left}_{right}_1", f"p_{left}_{right}_1")] += P_force
        Q_ext[(f"p_{left}_{right}_2", f"p_{left}_{right}_2")] -= P_force

    # 3. IDENTIFY LOOPS
    paired_indices = set()
    for stem in stems:
        for (left, right) in stem:
            paired_indices.add(left)
            paired_indices.add(right)
            
    loop_indices = [i for i in range(len(target_structure)) if i not in paired_indices and i not in fixed_loops]
    bases = ["A", "C", "G", "U"]
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]

    # 4. NEGATIVE DESIGN (Anti-Pairing & Mirror Penalties, Scaled by Alpha)
    P_anti = 50.0 * difficulty * alpha
    P_mirror = 100.0 * difficulty * alpha
    
    # Bias tax for loops
    bias_tax = {"A": 0.0, "C": 5.0, "G": 15.0, "U": 5.0}
    for idx in loop_indices:
        for b in bases:
            Q_ext[(f"x_{idx}_{b}", f"x_{idx}_{b}")] += bias_tax[b] * alpha

    # Anti-Pairing in Loops
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
                    
    # Mirror Penalty (Boundary)
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

    # 5. ONE-HOT ENCODING (Hard Constraint, Unscaled)
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

def run_qamoo_sweep(target_structure, c_coeffs, target_name="Structure"):
    sampler = SimulatedAnnealingSampler()
    stems = extract_stems(target_structure)
    
    successes, total_defect, total_diversity = 0, 0.0, 0.0
    plot_alphas, plot_mfes, plot_defects = [], [], []
    
    for i in range(11):
        alpha = i / 10.0
        print(f"\n--- Running Alpha = {alpha:.1f} ---")
        
        Q_ext, loop_indices, fixed_loops = build_qamoo_extended_qubo(target_structure, stems, c_coeffs, alpha=alpha)
        if i == 0:
            num_qubits = len(set([var for pair in Q_ext.keys() for var in pair]))
            print(f"QUBO Built: {num_qubits} Qubits (Smooth 2-Body Landscape!)")
            
        sampleset = sampler.sample_qubo(Q_ext, num_reads=3000)
        best_sample = sampleset.first.sample
        
        decoded = decode_extended_sample(best_sample, stems, loop_indices, fixed_loops)
        if decoded:
            pairs_list, loop_assignment = decoded
            seq = generate_sequence(target_structure, stems, pairs_list, loop_assignment)
            mfe_struct, mfe = RNA.fold(seq)
            
            if mfe_struct == target_structure:
                metrics = evaluate_vienna_ensemble(seq, target_structure)
                successes += 1
                total_defect += metrics["ensemble_defect"]
                total_diversity += metrics["diversity"]
                
                plot_alphas.append(alpha)
                plot_mfes.append(metrics["mfe"])
                plot_defects.append(metrics["ensemble_defect"])
                print(f"SUCCESS | MFE: {metrics['mfe']:.2f} | Defect: {metrics['ensemble_defect']:.2f} | Seq: {seq}")
            else:
                print(f"FAILED TO FOLD")
        else:
            print(f"FAILED TO FOLD (Invalid Annealer State)")

    # Plotting
    if plot_alphas:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 5))
        sc = plt.scatter(plot_mfes, plot_defects, c=plot_alphas, cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(sc, label='Alpha')
        plt.title(f"QAMOO Pareto Front: {target_name}")
        plt.xlabel("MFE (Lower is Better)")
        plt.ylabel("Ensemble Defect (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', target_name).strip()
        plt.savefig(f"plots/qamoo_pareto_{safe_name}.png")
        plt.close()

    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return 11, successes, avg_defect, avg_div

if __name__ == "__main__":
    print("Fitting OLS coefficients for Turner energy...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    df = pd.read_csv("Structures/fmqa_paper_structures.csv")
    print(f"Loaded {len(df)} structures from benchmark dataset.")
    
    for _, row in df.head(5).iterrows():
        name = row['Category']
        struct = row['Structure']
        print(f"\n==============================================")
        print(f"BENCHMARKING: {name}")
        print(f"STRUCTURE: {struct}")
        print(f"==============================================")
        
        tot, succ, avg_def, avg_div = run_qamoo_sweep(struct, c_coeffs, target_name=name)
        print(f"\n[FINAL] {name} -> Success Rate: {(succ/tot)*100:.1f}% | Avg Defect: {avg_def:.2f}")
