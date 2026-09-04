import time
import RNA
from phase3_coef_fitter import calculate_qubo_coeffs
from phase1_rules import extract_stems
from phase15_full_evaluation import (
    get_qubo_pairs_via_annealing,
    get_qubo_pairs_via_noisy_annealing,
    get_random_pairs_for_experiment,
    fill_loops
)
from phase20_coaxial_stacking import build_extended_qubo, decode_extended_sample, generate_sequence

# pyrefly: ignore [missing-import]
from dwave.samplers import SimulatedAnnealingSampler
# pyrefly: ignore [missing-import]
from dwave.samplers import PathIntegralAnnealingSampler
import csv
import os

# Define the path to the dataset
csv_path = r"d:\Academic UOP\Internship\simulation\Implementation\NN - Copy\Structures\fmqa_paper_structures.csv"

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

    top10_file.close()

        
    print("\n\nFINAL UNIFIED RESULTS TABLE")
    print("-" * 80)
    print(f"{'Target':<12} | {'Methodology':<32} | {'Total':<6} | {'Succ':<6} | {'Rate %':<8} | {'Avg Defect':<10} | {'Avg Div'}")
    print("-" * 110)
    for name, method, tot, succ, avg_def, avg_div in results_table:
        print(f"{name:<12} | {method:<32} | {tot:<6} | {succ:<6} | {(succ/tot*100):.1f}%     | {avg_def:<10.2f} | {avg_div:.2f}")
        
if __name__ == "__main__":
    run_unified_benchmark()
