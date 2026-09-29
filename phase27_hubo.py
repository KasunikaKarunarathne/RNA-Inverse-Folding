from pandas._libs.tslibs.offsets import Second
import dimod 
from collections import defaultdict
import RNA 
from phase1_rules import extract_stems
from phase17_extended import decode_extended_sample,generate_sequence
from dwave.samplers import SimulatedAnnealingSampler
from phase4_qubo_builder import build_approx_qubo
from phase17_extended import calculate_structural_difficulty
from phase3_coef_fitter import calculate_qubo_coeffs
import matplotlib.pyplot as plt
import os
import re
from phase18_unified_factorial import evaluate_vienna_ensemble


def find_dangling_ends(target_structure,stems):
    """
    Finds all unpaired loop bases that are immediately adjacent to a stem.
    Returns a dictionary mapping a stem pair (left, right) to a list of adjacent loop indices.
    """
    dangling_map = defaultdict(set)
    for stem in stems:
        # finding terminal closing pairs of the stem
        top_pair = stem[0]
        bottom_pair = stem[-1]

        # check inner loops 
        left,right = bottom_pair
        if target_structure[left+1] ==".":
            dangling_map[(left,right)].add(left+1)
        if target_structure[right-1] ==".":
            dangling_map[(left,right)].add(right-1)

        left_outer ,right_outer =top_pair
        if left_outer-1 >0 and target_structure[left_outer-1] ==".":
            dangling_map[(left_outer,right_outer)].add(left_outer-1)
        if right_outer+1 < len(target_structure) and target_structure[right_outer+1] ==".":
            dangling_map[(left_outer,right_outer)].add(right_outer+1)
    return dangling_map

def build_mo_hubo(target_structure,c_coeffs,alpha=0.5):
    """
    Builds a Multi Objective Higher-Order Binary Optimization (HUBO) model and reduces it to a QUBO.
    """
    stems = extract_stems(target_structure)
    dangling_map = find_dangling_ends(target_structure,stems)

    poly = defaultdict(float)
    bases = ["A","C","G","U"]
    # stem rewards from OLS c_coeffs (OLS weights)
    Q_stem ,offset = build_approx_qubo(stems,c_coeffs)
    max_stem_val = max([abs(v) for v in Q_stem.values()]) if Q_stem else 1.0

    # copy 1 body and 2 body terms from Q_stem to out HUBO poly
    for key,value in Q_stem.items():
        scaled_val = (value/max_stem_val)*(1.0-alpha)
        if key[0] == key[1]:
            #linear term
            poly[(key[0],)]+= scaled_val
        else:
            # quadratic term
            poly[key] += scaled_val

    # Danling ends - 3 body real turner approximations
     # Real Turner 2004 Dangling End approximations (kcal/mol)
    # Format: (Stem_Base1, Stem_Base2, Dangling_Base) : Reward
    dangle_rewards = {
        # If stem is C-G or G-C (very strong), loops provide massive bonuses
        ("C", "G", "A"): -1.5, ("C", "G", "G"): -1.3, ("C", "G", "U"): -0.8, ("C", "G", "C"): -0.4,
        ("G", "C", "A"): -1.5, ("G", "C", "G"): -1.3, ("G", "C", "U"): -0.8, ("G", "C", "C"): -0.4,
        # If stem is A-U or U-A (medium), loops provide good bonuses
        ("A", "U", "A"): -1.0, ("A", "U", "G"): -0.8, ("A", "U", "U"): -0.6, ("A", "U", "C"): -0.2,
        ("U", "A", "A"): -1.0, ("U", "A", "G"): -0.8, ("U", "A", "U"): -0.6, ("U", "A", "C"): -0.2,
        # If stem is G-U or U-G (weak wobble), loops provide small bonuses
        ("G", "U", "A"): -0.6, ("G", "U", "G"): -0.5, ("G", "U", "U"): -0.3, ("G", "U", "C"): -0.1,
        ("U", "G", "A"): -0.6, ("U", "G", "G"): -0.5, ("U", "G", "U"): -0.3, ("U", "G", "C"): -0.1,
    }
    max_dangle = 1.5
    for (left,right),d_ends in dangling_map.items():
        for d_idx in d_ends: # ex: d_ends = {2, 4}
            # d_idx = 2. (calculating the physics between bases 1, 5, and 2).
            # d_idx = 4. (calculating the physics between bases 1, 5, and 4).
            for (b_left,b_right,d_base),bonus in dangle_rewards.items():
                var_L = f"x_{left}_{b_left}"
                var_R = f"x_{right}_{b_right}"
                var_D = f"x_{d_idx}_{d_base}"
                scaled_bonus = (bonus/max_dangle)*(1-alpha)
                # Add the exact 3 body physics term 
                poly[(var_L,var_R,var_D)] += scaled_bonus

    # one hot and negative design : Penalty = P_onehot * (Sum_of_bases - 1)^2
    difficulty = calculate_structural_difficulty(target_structure,stems )
    P_onehot = 10.0*max(1.0,difficulty*0.5)
    for idx in range(len(target_structure)):
        for b in bases:
            poly[(f"x_{idx}_{b}",)] -= P_onehot
        for i in range(4):
            for j in range(i+1,4):
                poly[(f"x_{idx}_{bases[i]}",f"x_{idx}_{bases[j]}")] += 2 *P_onehot

    # BAse anti pairing for loops(negative design)
    paired_indices = set()
    for stem in stems:
        for (l,r) in stem:
            paired_indices.add(l)
            paired_indices.add(r)
    loop_indices = [i for i in range(len(target_structure)) if i not in paired_indices]

    P_anti_base = 50.0*difficulty
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]
    for i in range(len(loop_indices)):
        for j in range(1+1,len(loop_indices)):
            distance = abs(loop_indices[i] - loop_indices[j])
            if distance >=4:
                penalty = P_anti_base/distance
                for b1,b2 in valid_pairs:
                    # penalize loop bases forming valid pairs
                    var1 = f"x_{loop_indices[i]}_{b1}"
                    var2 = f"x_{loop_indices[j]}_{b2}"
                    # 2 body penalty
                    poly[(var1,var2)] += penalty
    
    # Rosenberg reduction
    P_aux = 20.0*max(1.0,difficulty) # high penalty on auxilary math
    bqm = dimod.make_quadratic(poly,strength=P_aux,vartype=dimod.BINARY)
    return bqm,stems,loop_indices

def run_mo_hubo_sweep(target_structure, c_coeffs, target_name="Structure"):
    sampler = SimulatedAnnealingSampler()
    successes = 0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = []
    
    plot_alphas, plot_mfes, plot_defects = [], [], []
    
    # Sweep alpha from 0.0 to 1.0
    for i in range(11):
        alpha = i / 10.0
        print(f"\n--- Running Alpha = {alpha:.1f} ---")
        
        # Build the HUBO with the current alpha weight
        bqm, stems, loop_indices = build_mo_hubo(target_structure, c_coeffs, alpha=alpha)
        
        sampleset = sampler.sample(bqm, num_reads=1000)
        best_sample = sampleset.first.sample
        
        decoded = decode_extended_sample(best_sample, stems, loop_indices, fixed_loops={})
        if decoded:
            pairs_list, loop_assignments = decoded
            seq = generate_sequence(target_structure, stems, pairs_list, loop_assignments)

            mfe_struct, mfe = RNA.fold(seq)
            if mfe_struct == target_structure:
                metrics = evaluate_vienna_ensemble(seq, target_structure)
                successes += 1
                total_defect += metrics["ensemble_defect"]
                total_diversity += metrics["diversity"]
                if len(top10_seqs) < 10:
                    top10_seqs.append(seq)
                
                plot_alphas.append(alpha)
                plot_mfes.append(metrics["mfe"])
                plot_defects.append(metrics["ensemble_defect"])
                
                print(f"SUCCESS | MFE: {metrics['mfe']:.2f} | Defect: {metrics['ensemble_defect']:.2f}")
            else:
                print(f"FAILED TO FOLD")
        else:
            print(f"FAILED TO FOLD (Invalid Annealer State)")

    # --- PLOT THE PARETO FRONT ---
    if plot_alphas:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 5))
        sc = plt.scatter(plot_mfes, plot_defects, c=plot_alphas, cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(sc, label='Alpha (0.0=Stability -> 1.0=Specificity)')
        for idx, alpha_val in enumerate(plot_alphas):
            plt.annotate(f"a={alpha_val:.1f}", (plot_mfes[idx], plot_defects[idx]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title(f"HUBO Pareto Front: {target_name}")
        plt.xlabel("MFE (Lower is Better)")
        plt.ylabel("Ensemble Defect (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', target_name).strip()
        plt.savefig(f"plots/hubo_pareto_{safe_name}.png")
        plt.close()
        print(f"--> Saved Pareto plot to plots/hubo_pareto_{safe_name}.png")

    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    
    return 11, successes, avg_defect, avg_div, top10_seqs

def test_hubo_architecture():
    target = "((....)).((....))"

    print(f"Target Structure: {target}")
    
    print("Fitting OLS coefficients for Turner energy...")
    c_coeffs = calculate_qubo_coeffs(method="ols")

    # Run the QAMOO sweep!
    tot, succ, avg_def, avg_div, top10 = run_mo_hubo_sweep(target, c_coeffs, target_name="3-Way Junction")
    print(f"\nFinal Result: {succ}/{tot} Successes | Avg Defect: {avg_def:.2f}")

if __name__ == "__main__":
    test_hubo_architecture()
