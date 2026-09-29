from phase17_extended import calculate_structural_difficulty
from collections import defaultdict
from phase20_coaxial_stacking import build_approx_qubo
import time
import RNA
import re
import os
import subprocess
import shutil
import glob
import random
import matplotlib.pyplot as plt
from phase3_coef_fitter import calculate_qubo_coeffs
from phase1_rules import extract_stems
from phase15_full_evaluation import (
    get_qubo_pairs_via_annealing,
    get_qubo_pairs_via_noisy_annealing,
    get_random_pairs_for_experiment,
    fill_loops
)
from phase17_extended import calculate_structural_difficulty, decode_extended_sample, generate_sequence

from phase17_extended import build_extended_qubo, decode_extended_sample, generate_sequence
# pyrefly: ignore [missing-import]
from dwave.samplers import SimulatedAnnealingSampler
# pyrefly: ignore [missing-import]
from dwave.samplers import PathIntegralAnnealingSampler
import csv
import os

# Define the path to the dataset
csv_path = r"d:\Academic UOP\Internship\simulation\Implementation\NN - Copy\Structures\fmqa_paper_structures.csv"

def build_mo_qubo(target_structure, stems, c_coeffs, alpha=0.5):
    """
    Multi-Objective QUBO Builder for RNA Inverse Folding
    alpha controls the trade-off between Stability and Specificity.
    """
    # Get the raw base stem rewards
    Q_stem, offset = build_approx_qubo(stems, c_coeffs)
    Q_ext = defaultdict(float)

    # --- NORMALIZE OBJECTIVE 1 (STEM STABILITY) ---
    max_stem_val = max([abs(v) for v in Q_stem.values()]) if Q_stem else 1.0
    for key, value in Q_stem.items():
        # Normalize to 1.0, then multiply by (1.0 - alpha)
        Q_ext[key] = (value / max_stem_val) * (1.0 - alpha)
    
    # 2. Identify Tetraloops and Force C-G Closing Pairs (Positive Design)
    tetraloop_stems = []
    fixed_loops = {}
    import re
    import random
    for match in re.finditer(r'\(\.\.\.\.\)', target_structure):
        close_left = match.start()
        close_right = match.end() - 1
        tetraloop_stems.append((close_left, close_right))
        
        idx = match.start() + 1
        tetraloop_choice = random.choice(["GCAA", "UUCG"])
        fixed_loops[idx] = tetraloop_choice[0]
        fixed_loops[idx+1] = tetraloop_choice[1]
        fixed_loops[idx+2] = tetraloop_choice[2]
        fixed_loops[idx+3] = tetraloop_choice[3]
        
    for (left, right) in tetraloop_stems:
        # HARD CONSTRAINT: Do NOT multiply by alpha!
        P_force = 1000.0
        Q_ext[(f"p_{left}_{right}_0", f"p_{left}_{right}_0")] -= P_force
        Q_ext[(f"p_{left}_{right}_1", f"p_{left}_{right}_1")] += P_force
        Q_ext[(f"p_{left}_{right}_2", f"p_{left}_{right}_2")] -= P_force

    # 3. Identify remaining loop indices
    paired_indices = set()
    for stem in stems:
        for (left, right) in stem:
            paired_indices.add(left)
            paired_indices.add(right)
            
    loop_indices = [i for i in range(len(target_structure)) 
                    if i not in paired_indices and i not in fixed_loops]
    
    bases = ["A", "C", "G", "U"]
    difficulty_multiplier = calculate_structural_difficulty(target_structure, stems)
    
    # --- HARD CONSTRAINT: One-Hot Penalty ---
    # Do NOT multiply by alpha. We MUST have exactly 1 base per position.
    P_onehot = 1000.0 
    for idx in loop_indices:
        for b in bases:
            var = f"x_{idx}_{b}"
            Q_ext[(var, var)] += (-P_onehot)  
            
        for i in range(4):
            for j in range(i+1, 4):
                key = tuple(sorted([f"x_{idx}_{bases[i]}", f"x_{idx}_{bases[j]}"]))
                Q_ext[key] += 2 * P_onehot

    # --- NORMALIZE OBJECTIVE 2 (LOOP SPECIFICITY / NEGATIVE DESIGN) ---
    # Base anti-pairing weight
    P_anti_base = 50.0 * difficulty_multiplier
    
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]
    for i in range(len(loop_indices)):
        for j in range(i+1, len(loop_indices)):
            distance = abs(loop_indices[i] - loop_indices[j])
            if distance >= 4:
                # Normalize the penalty down to a magnitude of 1.0, then multiply by alpha
                penalty = (4.0 / distance) * alpha
                
                for b1, b2 in valid_pairs:
                    key = tuple(sorted([f"x_{loop_indices[i]}_{b1}", f"x_{loop_indices[j]}_{b2}"]))
                    Q_ext[key] += penalty

    return dict(Q_ext), loop_indices, fixed_loops

def run_mo_qubo_sweep(target_structure, c_coeffs, target_name="Structure"):
    stems = extract_stems(target_structure)
    sampler = SimulatedAnnealingSampler()
    
    successes = 0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = []
    
    # Store points for plotting
    plot_alphas = []
    plot_mfes = []
    plot_defects = []
    
    # Sweep alpha from 0.0 to 1.0 in steps of 0.1
    for i in range(11):
        alpha = i / 10.0
        print(f"\n--- Running Alpha = {alpha:.1f} ---")
        
        Q_ext, loop_indices, fixed_loops = build_mo_qubo(target_structure, stems, c_coeffs, alpha=alpha)
        sampleset = sampler.sample_qubo(Q_ext, num_reads=500)
        best_sample = sampleset.first.sample
        decoded = decode_extended_sample(best_sample, stems, loop_indices, fixed_loops)
        
        if decoded:
            pairs_list, loop_assignments = decoded
            seq = generate_sequence(target_structure, stems, pairs_list, loop_assignments)

            if check_structure(seq, target_structure):
                metrics = evaluate_vienna_ensemble(seq, target_structure)
                successes += 1
                total_defect += metrics["ensemble_defect"]
                total_diversity += metrics["diversity"]
                if len(top10_seqs) < 10:
                    top10_seqs.append(seq)
                    
                # Save the successful points for plotting
                plot_alphas.append(alpha)
                plot_mfes.append(metrics["mfe"])
                plot_defects.append(metrics["ensemble_defect"])
                
                print(f"SUCCESS | MFE: {metrics['mfe']:.2f} | Defect: {metrics['ensemble_defect']:.2f}")
            else:
                print(f"FAILED TO FOLD (Incorrect Structure)")
        else:
            print(f"FAILED TO FOLD (Invalid Annealer State)")

    # --- PLOTTING THE PARETO FRONT ---
    if plot_alphas:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 5))
        
        # Create a scatter plot colored by the Alpha value
        sc = plt.scatter(plot_mfes, plot_defects, c=plot_alphas, cmap='viridis', s=100, edgecolor='black')
        plt.colorbar(sc, label='Alpha (0.0=Stability -> 1.0=Specificity)')
        
        # Add labels to the dots
        for idx, alpha_val in enumerate(plot_alphas):
            plt.annotate(f"a={alpha_val:.1f}", (plot_mfes[idx], plot_defects[idx]), 
                         textcoords="offset points", xytext=(0,10), ha='center')
            
        plt.title(f"Pareto Front: {target_name}")
        plt.xlabel("MFE (Lower is Better)")
        plt.ylabel("Ensemble Defect (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Remove bad characters from the target name so we can save it as a file
        safe_name = re.sub(r'[^a-zA-Z0-9_\- ]', '', target_name).strip()
        file_path = f"plots/pareto_{safe_name}.png"
        plt.savefig(file_path)
        plt.close()
        print(f"--> Saved Pareto plot to {file_path}")

    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    total_generated = 11
    
    return total_generated, successes, avg_defect, avg_div, top10_seqs


def check_structure(seq, target):
    mfe_struct, _ = RNA.fold(seq)
    return mfe_struct == target

def evaluate_vienna_ensemble(seq, target_structure):
    """
    Evaluates the ensemble stability and diversity of a sequence.
    Run this ONLY on sequences that successfully fold into the target MFE structure.
    """
    fc = RNA.fold_compound(seq)
    
    # 1. MFE Stability
    struct_mfe, mfe = fc.mfe()
    
    # Rescale params based on MFE (often required before PF for stable computation)
    fc.exp_params_rescale(mfe)
    
    # 2. Partition Function & Ensemble Diversity
    struct_pf, free_energy_ensemble = fc.pf()
    
    # 3. Ensemble Defect 
    ensemble_defect = fc.ensemble_defect(target_structure)
    
    # 4. Vienna Diversity
    diversity = fc.mean_bp_distance()
    
    return {
        "mfe": mfe,
        "ensemble_free_energy": free_energy_ensemble,
        "ensemble_defect": ensemble_defect,
        "diversity": diversity
    }

def run_vienna_inverse(target_structure, num_output=10):
    valid_sequences = set()
    total_generated = 0
    attempts = 0
    max_attempts = num_output * 10
    
    while len(valid_sequences) < num_output and attempts < max_attempts:
        attempts += 1
        total_generated += 1
        # Random start sequence
        start_seq = "".join(random.choices("ACGU", k=len(target_structure)))
        seq, dist = RNA.inverse_fold(start_seq, target_structure)
        if dist == 0.0:
            valid_sequences.add(seq)
            
    successes = 0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = list(valid_sequences)[:10]
    
    for seq in valid_sequences:
        successes += 1
        metrics = evaluate_vienna_ensemble(seq, target_structure)
        total_defect += metrics["ensemble_defect"]
        total_diversity += metrics["diversity"]
        
    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return total_generated, successes, avg_defect, avg_div, top10_seqs

def run_desirna_inverse(target_structure, num_output=10):
    # Setup temp input for DesiRNA
    input_file = "d.txt"
    seq_restr = "N" * len(target_structure)
    input_content = f">name\nDesign\n>seq_restr\n{seq_restr}\n>sec_struct\n{target_structure}\n"
    
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(input_content)
        
    python_exe = r".\protein_sim\Scripts\python.exe"
    desirna_script = r"DesiRNA/DesiRNA.py"
    
    # Run DesiRNA for 15 seconds per target
    cmd = [python_exe, desirna_script, "-f", input_file, "-t", "60", "-r", str(num_output)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total_generated = 0
    # Parse output
    valid_sequences = set()
    output_csvs = glob.glob("d_*/*_results.csv")
    if output_csvs:
        with open(output_csvs[0], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_generated +=1
                seq = row.get("sequence", "")
                if seq and check_structure(seq, target_structure):
                    valid_sequences.add(seq)
                    
    # Cleanup DesiRNA files
    if os.path.exists(input_file):
        os.remove(input_file)
    for folder in glob.glob("d_*"):
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
            
    successes =0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = list(valid_sequences)[:10]
    
    for seq in valid_sequences:
        successes += 1
        metrics = evaluate_vienna_ensemble(seq, target_structure)
        total_defect += metrics["ensemble_defect"]
        total_diversity += metrics["diversity"]
        
    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    
    if total_generated == 0: total_generated = 1
    return total_generated, successes, avg_defect, avg_div, top10_seqs

def run_extended_qubo(target_structure, c_coeffs, num_reads=1000, num_take=10, sampler_type="SA"):
    stems = extract_stems(target_structure)
    Q_ext, offset, loop_indices, fixed_loops = build_extended_qubo(target_structure, stems, c_coeffs)
    
    if sampler_type == "SA":
        sampler = SimulatedAnnealingSampler()
    else:
        sampler = PathIntegralAnnealingSampler() 
    
    num_ensembles = 5
    reads_per_ensemble = num_reads // num_ensembles
    
    seen = set()
    valid_sequences_with_energy = []
    
    for e in range(num_ensembles):
        # Vary sweeps per ensemble to increase diversity (e.g. 200, 250, 300, 350, 400)
        sweeps = 200 + (e * 50)
        if sampler_type == "SQA":
            sampleset = sampler.sample_qubo(Q_ext, num_reads=reads_per_ensemble, num_sweeps=sweeps, num_trotter_slices=10)
        else:
            sampleset = sampler.sample_qubo(Q_ext, num_reads=reads_per_ensemble, num_sweeps=sweeps)
        
        for sample, energy in sampleset.data(["sample", "energy"]):
            decoded = decode_extended_sample(sample, stems, loop_indices, fixed_loops)
            if decoded:
                pairs_list, loop_assignment = decoded
                seq = generate_sequence(target_structure, stems, pairs_list, loop_assignment)
                if seq not in seen:
                    seen.add(seq)
                    valid_sequences_with_energy.append({
                        'seq': seq,
                        'energy': energy
                    })
                    
    # Sort pooled ensemble results by quantum energy
    valid_sequences_with_energy.sort(key=lambda x: x['energy'])
    
    # Take the top N
    top_samples = valid_sequences_with_energy[:num_take]
    valid_sequences = [x['seq'] for x in top_samples]
                
    successes = 0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = []
    
    for seq in valid_sequences:
        if check_structure(seq, target_structure):
            successes += 1
            if len(top10_seqs) < 10:
                top10_seqs.append(seq)
            metrics = evaluate_vienna_ensemble(seq, target_structure)
            total_defect += metrics["ensemble_defect"]
            total_diversity += metrics["diversity"]
            
    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return len(valid_sequences), successes, avg_defect, avg_div, top10_seqs

def run_classical_loop_pipeline(target_structure, pair_results, use_penalty, num_per_assignment=10):
    successes = 0
    total = 0
    total_defect = 0.0
    total_diversity = 0.0
    top10_seqs = []
    for res in pair_results:
        candidates = fill_loops(
            target_structure=target_structure,
            pair_assignment=res["pairs_list"],
            pool_size=100,
            num_output=num_per_assignment,
            use_penalty=use_penalty
        )
        total += len(candidates)
        for seq in candidates:
            if check_structure(seq, target_structure):
                successes += 1
                if len(top10_seqs) < 10:
                    top10_seqs.append(seq)
                metrics = evaluate_vienna_ensemble(seq, target_structure)
                total_defect += metrics["ensemble_defect"]
                total_diversity += metrics["diversity"]
                
    avg_defect = total_defect / successes if successes > 0 else 0.0
    avg_div = total_diversity / successes if successes > 0 else 0.0
    return total, successes, avg_defect, avg_div, top10_seqs

def run_unified_benchmark():
    print("Pre-computing OLS coefficients...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    targets = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Create a name like "Hairpin 1", "Bulge 57", etc.
                name = f"{row['Category']} {i+1}"
                targets.append((name, row["Structure"]))
    else:
        print(f"Error: Could not find dataset at {csv_path}")
    
    results_table = []
    
    # Log top 10 sequences to a file
    top10_file = open("top10_sequences.txt", "w")
    
    for name, target in targets:
        print(f"\n==============================================")
        print(f"Testing {name}: {target}")
        print(f"==============================================")
        top10_file.write(f"\nTarget: {name} | {target}\n")
        top10_file.write(f"{'='*60}\n")
        
        # 1. Baseline (Random Stems + Random Loops, No Penalty)
        print("Running Baseline...")
        rand_pairs, _ = get_random_pairs_for_experiment(target, 10)
        tot, succ, avg_def, avg_div, top10 = run_classical_loop_pipeline(target, rand_pairs, use_penalty=False, num_per_assignment=10)
        results_table.append((name, "Baseline (Random)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Baseline (Random) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 2. Normal QUBO (SA Stems + Penalty Loops)
        print("Running Normal QUBO (SA)...")
        sa_pairs, _ = get_qubo_pairs_via_annealing(target, c_coeffs, num_reads=500, sampler_type="SA")
        sa_pairs = sa_pairs[:10]
        tot, succ, avg_def, avg_div, top10 = run_classical_loop_pipeline(target, sa_pairs, use_penalty=True, num_per_assignment=10)
        results_table.append((name, "Normal QUBO (SA+Penalty)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Normal QUBO (SA+Penalty) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 3. Normal QUBO (SQA Stems + Penalty Loops)
        print("Running Normal QUBO (SQA)...")
        sqa_pairs, _ = get_qubo_pairs_via_annealing(target, c_coeffs, num_reads=500, sampler_type="SQA")
        sqa_pairs = sqa_pairs[:10]
        tot, succ, avg_def, avg_div, top10 = run_classical_loop_pipeline(target, sqa_pairs, use_penalty=True, num_per_assignment=10)
        results_table.append((name, "Normal QUBO (SQA+Penalty)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Normal QUBO (SQA+Penalty) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 4. Noisy QUBO (Noisy SA Stems + Penalty Loops)
        print("Running Noisy QUBO (SA)...")
        nsa_pairs, _ = get_qubo_pairs_via_noisy_annealing(target, c_coeffs, num_reads=500, sampler_type="SA")
        nsa_pairs = nsa_pairs[:10]
        tot, succ, avg_def, avg_div, top10 = run_classical_loop_pipeline(target, nsa_pairs, use_penalty=True, num_per_assignment=10)
        results_table.append((name, "Noisy QUBO (Noisy SA+Penalty)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Noisy QUBO (Noisy SA+Penalty) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 5. Noisy QUBO (Noisy SQA Stems + Penalty Loops)
        print("Running Noisy QUBO (SQA)...")
        nsqa_pairs, _ = get_qubo_pairs_via_noisy_annealing(target, c_coeffs, num_reads=500, sampler_type="SQA")
        nsqa_pairs = nsqa_pairs[:10]
        tot, succ, avg_def, avg_div, top10 = run_classical_loop_pipeline(target, nsqa_pairs, use_penalty=True, num_per_assignment=10)
        results_table.append((name, "Noisy QUBO (Noisy SQA+Penalty)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Noisy QUBO (Noisy SQA+Penalty) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 6. Extended QUBO (SA All-in-One Quantum)
        print("Running Extended QUBO (SA)...")
        tot, succ, avg_def, avg_div, top10 = run_extended_qubo(target, c_coeffs, num_reads=10000, num_take=100, sampler_type="SA")
        if tot == 0: tot = 1
        results_table.append((name, "Extended QUBO (SA All-in-One)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Extended QUBO (SA All-in-One) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        # 7. Extended QUBO (SQA All-in-One Quantum)
        print("Running Extended QUBO (SQA)...")
        tot, succ, avg_def, avg_div, top10 = run_extended_qubo(target, c_coeffs, num_reads=10000, num_take=100, sampler_type="SQA")
        if tot == 0: tot = 1
        results_table.append((name, "Extended QUBO (SQA All-in-One)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Extended QUBO (SQA All-in-One) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")

        print("Running Extended QUBO (SA)...")
        tot, succ, avg_def, avg_div, top10 = run_mo_qubo_sweep(target, c_coeffs, target_name=name)
        if tot == 0: tot = 1
        results_table.append((name, "Extended QUBO MO (SA All-in-One)", tot, succ, avg_def, avg_div))
        top10_file.write(f"Extended QUBO MO (SA All-in-One) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")
        
        
        # 8. Vienna RNAinverse Baseline
        # print("Running Vienna RNAinverse...")
        # tot, succ, avg_def, avg_div, top10 = run_vienna_inverse(target, num_output=10)
        # if tot == 0: tot = 1
        # results_table.append((name, "Vienna RNAinverse", tot, succ, avg_def, avg_div))
        # top10_file.write(f"Vienna RNAinverse Top Sequences:\n" + "\n".join(top10) + "\n\n")
        # print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")

        # 9. DesiRNA Baseline
        print("Running DesiRNA...")
        tot, succ, avg_def, avg_div, top10 = run_desirna_inverse(target, num_output=10)
        if tot == 0: tot = 1
        results_table.append((name, "DesiRNA (State-of-the-Art)", tot, succ, avg_def, avg_div))
        top10_file.write(f"DesiRNA (State-of-the-Art) Top Sequences:\n" + "\n".join(top10) + "\n\n")
        print(f"  -> {succ}/{tot} ({(succ/tot*100):.1f}%) | Defect: {avg_def:.2f} | Div: {avg_div:.2f}")

    top10_file.close()

        
    print("\n\nFINAL UNIFIED RESULTS TABLE")
    print("-" * 80)
    print(f"{'Target':<12} | {'Methodology':<32} | {'Total':<6} | {'Succ':<6} | {'Rate %':<8} | {'Avg Defect':<10} | {'Avg Div'}")
    print("-" * 110)
    for name, method, tot, succ, avg_def, avg_div in results_table:
        print(f"{name:<12} | {method:<32} | {tot:<6} | {succ:<6} | {(succ/tot*100):.1f}%     | {avg_def:<10.2f} | {avg_div:.2f}")
        
if __name__ == "__main__":
    run_unified_benchmark()
