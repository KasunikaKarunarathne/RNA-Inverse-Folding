import os
import re
import pandas as pd
import RNA
from collections import defaultdict
import dimod
from dwave.samplers import SimulatedAnnealingSampler
import matplotlib.pyplot as plt

# Import existing logic
from phase1_rules import extract_stems
from phase18_unified_factorial import evaluate_vienna_ensemble
from phase17_extended import calculate_structural_difficulty
from phase2_turner_energy import TURNER_2004

# Raw Dangling End Physics (kcal/mol)
DANGLE_REWARDS = {
    ("C", "G", "A"): -1.5, ("C", "G", "G"): -1.3, ("C", "G", "U"): -0.8, ("C", "G", "C"): -0.4,
    ("G", "C", "A"): -1.5, ("G", "C", "G"): -1.3, ("G", "C", "U"): -0.8, ("G", "C", "C"): -0.4,
    ("A", "U", "A"): -1.0, ("A", "U", "G"): -0.8, ("A", "U", "U"): -0.6, ("A", "U", "C"): -0.2,
    ("U", "A", "A"): -1.0, ("U", "A", "G"): -0.8, ("U", "A", "U"): -0.6, ("U", "A", "C"): -0.2,
    ("G", "U", "A"): -0.6, ("G", "U", "G"): -0.5, ("G", "U", "U"): -0.3, ("G", "U", "C"): -0.1,
    ("U", "G", "A"): -0.6, ("U", "G", "G"): -0.5, ("U", "G", "U"): -0.3, ("U", "G", "C"): -0.1,
}

VALID_PAIRS = [("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")]
BASES = ["A", "C", "G", "U"]

def find_dangling_ends(target_structure, stems):
    dangling_map = defaultdict(set)
    for stem in stems:
        top_pair = stem[0]
        bottom_pair = stem[-1]
        
        left, right = bottom_pair
        if target_structure[left+1] == ".": dangling_map[(left, right)].add(left+1)
        if target_structure[right-1] == ".": dangling_map[(left, right)].add(right-1)

        left_outer, right_outer = top_pair
        if left_outer-1 >= 0 and target_structure[left_outer-1] == ".": dangling_map[(left_outer, right_outer)].add(left_outer-1)
        if right_outer+1 < len(target_structure) and target_structure[right_outer+1] == ".": dangling_map[(left_outer, right_outer)].add(right_outer+1)
    return dangling_map

def build_advanced_mo_hubo(target_structure, stems, alpha=0.5):
    poly = defaultdict(float)
    difficulty = calculate_structural_difficulty(target_structure, stems)
    
    paired_indices = set()
    for stem in stems:
        for (l, r) in stem:
            paired_indices.add(l)
            paired_indices.add(r)
            
    # ---------------------------------------------------------
    # 1. EXACT TURNER STEMS (4-Body Terms)
    # ---------------------------------------------------------
    max_turner = max([abs(v) for v in TURNER_2004.values()])
    
    for stem in stems:
        # Penalize invalid pairs at the base pair level (2-body)
        for (l, r) in stem:
            for b1 in BASES:
                for b2 in BASES:
                    if (b1, b2) not in VALID_PAIRS:
                        poly[(f"x_{l}_{b1}", f"x_{r}_{b2}")] += 5.0 * (1.0 - alpha)
                        
        # 4-Body Stack Energies
        for i in range(len(stem) - 1):
            L1, R1 = stem[i]
            L2, R2 = stem[i+1]
            
            for (p1_l, p1_r) in VALID_PAIRS:
                for (p2_l, p2_r) in VALID_PAIRS:
                    pair1_str = p1_l + p1_r
                    pair2_str = p2_l + p2_r
                    if (pair1_str, pair2_str) in TURNER_2004:
                        energy = TURNER_2004[(pair1_str, pair2_str)]
                        scaled_energy = (energy / max_turner) * (1.0 - alpha)
                        
                        vL1, vR1 = f"x_{L1}_{p1_l}", f"x_{R1}_{p1_r}"
                        vL2, vR2 = f"x_{L2}_{p2_l}", f"x_{R2}_{p2_r}"
                        poly[(vL1, vR1, vL2, vR2)] += scaled_energy

    # ---------------------------------------------------------
    # 2. EXACT DANGLING ENDS (3-Body Terms)
    # ---------------------------------------------------------
    dangling_map = find_dangling_ends(target_structure, stems)
    for (left, right), d_ends in dangling_map.items():
        for d_idx in d_ends:
            for (b_left, b_right, d_base), bonus in DANGLE_REWARDS.items():
                vL, vR, vD = f"x_{left}_{b_left}", f"x_{right}_{b_right}", f"x_{d_idx}_{d_base}"
                poly[(vL, vR, vD)] += (bonus / max_turner) * (1.0 - alpha)

    # ---------------------------------------------------------
    # 3. TETRALOOP POSITIVE DESIGN
    # ---------------------------------------------------------
    fixed_loops = {}
    for match in re.finditer(r'\(\.\.\.\.\)', target_structure):
        idx = match.start() + 1
        tetraloop = "UUCG"
        for k in range(4):
            fixed_loops[idx+k] = tetraloop[k]
            poly[(f"x_{idx+k}_{tetraloop[k]}",)] -= 5.0 * (1.0 - alpha) # Reward correct base
            
        l, r = match.start(), match.end() - 1
        poly[(f"x_{l}_C", f"x_{r}_G")] -= 5.0 * (1.0 - alpha) # Force C-G closing

    # ---------------------------------------------------------
    # 4. NEGATIVE DESIGN (Anti-Pairing & Mirror Penalty)
    # ---------------------------------------------------------
    loop_indices = [i for i in range(len(target_structure)) if i not in paired_indices and i not in fixed_loops]
    
    # Bias tax in loops to encourage A placements
    bias_tax = {"A": 0.0, "C": 0.5, "G": 1.5, "U": 0.5}
    for idx in loop_indices:
        for b in BASES:
            poly[(f"x_{idx}_{b}",)] += bias_tax[b] * alpha

    # Loop Anti-Pairing
    for i in range(len(loop_indices)):
        for j in range(i+1, len(loop_indices)):
            dist = abs(loop_indices[i] - loop_indices[j])
            if dist >= 4:
                penalty = (4.0 / dist) * alpha
                for b1, b2 in VALID_PAIRS:
                    poly[(f"x_{loop_indices[i]}_{b1}", f"x_{loop_indices[j]}_{b2}")] += penalty

    # Mirror Penalty (boundary kissing)
    for stem in stems:
        l, r = stem[-1]
        for d in [1, 2]:
            curr_l, curr_r = l + d, r - d
            if curr_l >= curr_r: break
            if curr_l in loop_indices and curr_r in loop_indices:
                for b1, b2 in VALID_PAIRS:
                    poly[(f"x_{curr_l}_{b1}", f"x_{curr_r}_{b2}")] += 2.0 * alpha

    # ---------------------------------------------------------
    # 5. ONE-HOT ENCODING (Hard Constraint)
    # ---------------------------------------------------------
    P_onehot = 10.0 * max(1.0, difficulty)
    for idx in range(len(target_structure)):
        for b in BASES:
            poly[(f"x_{idx}_{b}",)] -= P_onehot
        for i in range(4):
            for j in range(i+1, 4):
                poly[(f"x_{idx}_{BASES[i]}", f"x_{idx}_{BASES[j]}")] += 2 * P_onehot

    # ---------------------------------------------------------
    # 6. ROSENBERG REDUCTION
    # ---------------------------------------------------------
    P_aux = 20.0 * max(1.0, difficulty)
    bqm = dimod.make_quadratic(poly, strength=P_aux, vartype=dimod.BINARY)
    return bqm

def run_advanced_mo_sweep(target_structure, target_name="Structure"):
    sampler = SimulatedAnnealingSampler()
    stems = extract_stems(target_structure)
    
    successes, total_defect, total_diversity = 0, 0.0, 0.0
    plot_alphas, plot_mfes, plot_defects = [], [], []
    
    for i in range(11):
        alpha = i / 10.0
        print(f"\n--- Running Alpha = {alpha:.1f} ---")
        
        bqm = build_advanced_mo_hubo(target_structure, stems, alpha=alpha)
        if i == 0:
            print(f"BQM Built: {len(bqm.variables)} Qubits")
            
        sampleset = sampler.sample(bqm, num_reads=3000)
        
        # Decode the unified One-Hot Sequence directly!
        seq_chars = [""] * len(target_structure)
        valid = True
        best_sample = sampleset.first.sample
        
        for idx in range(len(target_structure)):
            active = [b for b in BASES if best_sample.get(f"x_{idx}_{b}", 0) == 1]
            if len(active) == 1:
                seq_chars[idx] = active[0]
            else:
                valid = False # One-Hot constraint failed
                break
                
        if valid:
            seq = "".join(seq_chars)
            mfe_struct, mfe = RNA.fold(seq)
            if mfe_struct == target_structure:
                metrics = evaluate_vienna_ensemble(seq, target_structure)
                successes += 1
                total_defect += metrics["ensemble_defect"]
                total_diversity += metrics["diversity"]
                
                plot_alphas.append(alpha)
                plot_mfes.append(metrics["mfe"])
                plot_defects.append(metrics["ensemble_defect"])
                print(f"SUCCESS | MFE: {metrics['mfe']:.2f} | Defect: {metrics['ensemble_defect']:.2f}")
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
        plt.title(f"HUBO Pareto Front: {target_name}")
        plt.xlabel("MFE (Lower is Better)")
        plt.ylabel("Ensemble Defect (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', target_name).strip()
        plt.savefig(f"plots/hubo_pareto_{safe_name}.png")
        plt.close()

    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return 11, successes, avg_defect, avg_div

if __name__ == "__main__":
    df = pd.read_csv("Structures/fmqa_paper_structures.csv")
    print(f"Loaded {len(df)} structures from benchmark dataset.")
    
    # We will test the first 5 to see it destroy the benchmark!
    for _, row in df.head(5).iterrows():
        name = row['Category']
        struct = row['Structure']
        print(f"\n==============================================")
        print(f"BENCHMARKING: {name}")
        print(f"STRUCTURE: {struct}")
        print(f"==============================================")
        
        tot, succ, avg_def, avg_div = run_advanced_mo_sweep(struct, target_name=name)
        print(f"\n[FINAL] {name} -> Success Rate: {(succ/tot)*100:.1f}% | Avg Defect: {avg_def:.2f}")
