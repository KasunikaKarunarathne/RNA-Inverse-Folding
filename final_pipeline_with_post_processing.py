import os
import csv
import random
import RNA
import time
import numpy as np
from collections import Counter, defaultdict
import re
import glob
import subprocess
import shutil
import sys

# pyrefly: ignore [missing-import]
from dwave.samplers import PathIntegralAnnealingSampler
# pyrefly: ignore [missing-import]
from dwave.samplers import SimulatedAnnealingSampler

from phase1_rules import extract_stems, ALLOWED_PAIRS
from phase2_turner_energy import get_turner_energy
from phase3_coef_fitter import calculate_qubo_coeffs
from phase10_annealin import decode_bits_to_pairs
from phase8_paired_sampling import generate_random_pairs

# Coaxial stacking QUBO builder
from phase20_coaxial_stacking import build_approx_qubo as coaxial_build_approx_qubo
from phase20_coaxial_stacking import build_extended_qubo as coaxial_build_extended_qubo
from phase20_coaxial_stacking import decode_extended_sample, generate_sequence

# Existing penalties
from phase15_full_evaluation import (
    calculate_mirror_penalty_multistem,
    calculate_loop_entropy_penalty_multistem,
    calculate_terminal_penalty,
    flat_to_nested
)

# CPLEX exact solver
try:
    from docplex.mp.model import Model
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False

# Supervisor's 2-bit Extended Binary QUBO
from phase17_extended_binary import (
    calculate_binary_loop_coeffs,
    build_supervisor_qubo,
    decode_supervisor_sample,
    generate_sequence as generate_binary_sequence
)

# ==========================================
# GLOBAL CONFIGURATION VARIABLES
# ==========================================
NUM_RUNS = 20              # Number of times to repeat the whole evaluation
NUM_QUBO_PAIRS = 10       # Number of unique top pairs to extract from Annealer
LOOPS_PER_PAIR = 2        # Number of variations to generate per pair
TOTAL_SEQ_PER_METHOD = NUM_QUBO_PAIRS * LOOPS_PER_PAIR  # Total sequences evaluated per algorithm
# ==========================================

def calculate_shifted_mirror_penalty_multistem(sequence, target_structure, max_shift=2):
    """Adapted from Phase 11 for multi-stem support."""
    stems = extract_stems(target_structure)
    if not stems:
        return 0.0

    total_penalty = 0.0
    for stem in stems:
        left, right = stem[-1] 
        for s_left in range(max_shift + 1):
            for s_right in range(max_shift + 1):
                curr_l = left + 1 + s_left
                curr_r = right - 1 - s_right
                L_remaining = curr_r - curr_l - 1
                m = L_remaining // 2
                if m <= 0: continue
                
                shift_penalty = 0.0
                prev_pair_str = sequence[left] + sequence[right]
                for d in range(1, m + 1):
                    if curr_l >= curr_r: break
                    curr_pair_str = sequence[curr_l] + sequence[curr_r]
                    w_d = 1.0 - 0.5 * ((d - 1) / (m - 1)) if m > 1 else 1.0
                    if curr_pair_str in ALLOWED_PAIRS and prev_pair_str in ALLOWED_PAIRS:
                        stack_energy = get_turner_energy(prev_pair_str, curr_pair_str)
                        phi = max(0.0, -stack_energy)
                        shift_penalty += w_d * phi
                    prev_pair_str = curr_pair_str
                    curr_l += 1
                    curr_r -= 1
                total_penalty += shift_penalty
    return total_penalty

def fill_loops_custom(target_structure, pair_assignment, num_output, penalty_type):
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

    for match in re.finditer(r'\(\.\.\.\.\)', target_structure):
        close_left = match.start()
        close_right = match.end() - 1
        fixed_bases[close_left] = 'C'
        fixed_bases[close_right] = 'G'
        idx = match.start() + 1
        tetraloop_choice = random.choice(["GCAA","UUCG"])
        fixed_bases[idx] = tetraloop_choice[0]
        fixed_bases[idx+1] = tetraloop_choice[1]
        fixed_bases[idx+2] = tetraloop_choice[2]
        fixed_bases[idx+3] = tetraloop_choice[3]

    bases = ["A", "U", "C", "G"]
    pool_size = 1000 if penalty_type != "none" else num_output
    candidates = []

    for _ in range(pool_size):
        seq = []
        for i in range(full_length):
            if i in fixed_bases:
                seq.append(fixed_bases[i])
            else:
                seq.append(random.choices(["A", "C", "G", "U"], weights=[45, 45, 5, 5])[0])
        full_seq = "".join(seq)
        if penalty_type == "none":
            candidates.append((0.0, full_seq))
        else:
            penalty = calculate_terminal_penalty(full_seq, target_structure)
            if penalty_type == "entropy":
                penalty += calculate_loop_entropy_penalty_multistem(full_seq, target_structure)
            elif penalty_type == "mirror":
                penalty += calculate_mirror_penalty_multistem(full_seq, target_structure)
            elif penalty_type == "mirror_entropy":
                penalty += calculate_loop_entropy_penalty_multistem(full_seq, target_structure)
                penalty += calculate_mirror_penalty_multistem(full_seq, target_structure)
            elif penalty_type == "shifted_mirror":
                penalty += calculate_shifted_mirror_penalty_multistem(full_seq, target_structure)
            elif penalty_type == "shifted_mirror_entropy":
                penalty += calculate_loop_entropy_penalty_multistem(full_seq, target_structure)
                penalty += calculate_shifted_mirror_penalty_multistem(full_seq, target_structure)
            candidates.append((penalty, full_seq))

    if penalty_type != "none":
        candidates.sort(key=lambda x: x[0])
    return [seq for _, seq in candidates[:num_output]]

def get_qubo_pairs_via_annealing_coaxial(target_structure, c_coeffs, num_reads=1000, sampler_type="SA"):
    stems = extract_stems(target_structure)
    Q_dict, offset = coaxial_build_approx_qubo(stems, c_coeffs)
    
    if sampler_type == "SA":
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads)
    else:
        sampler = PathIntegralAnnealingSampler()
        sampleset = sampler.sample_qubo(Q_dict, num_reads=num_reads, num_trotter_slices=10)
        
    all_results = []
    for sample, energy in sampleset.data(["sample", "energy"]):
        pairs_list = decode_bits_to_pairs(sample, stems)
        if pairs_list is not None:
            all_results.append({"pairs_list": pairs_list, "qubo": energy + offset})

    all_results.sort(key=lambda x: x["qubo"])
    
    ten_pct_idx = max(1, int(len(all_results) * 0.10))
    top_slice = all_results[:ten_pct_idx]
    top_unique = []
    seen = set()
    for r in top_slice:
        if r["pairs_list"] not in seen:
            seen.add(r["pairs_list"])
            top_unique.append(r)
    return top_unique

def get_qubo_pairs_via_noisy_annealing_coaxial(target_structure, c_coeffs, num_reads=1000, sampler_type="SA"):
    stems = extract_stems(target_structure)
    Q_dict, offset = coaxial_build_approx_qubo(stems, c_coeffs)
    total_pairs = sum(len(stem) for stem in stems)
    dynamic_noise_scale = min(0.15, max(0.03, total_pairs * 0.015))
    
    if sampler_type == "SA":
        sampler = SimulatedAnnealingSampler()
    else:
        sampler = PathIntegralAnnealingSampler()
        
    all_results = []
    num_batches = 10
    reads_per_batch = max(1, num_reads // num_batches)
    
    for _ in range(num_batches):
        noisy_Q = {}
        for k, v in Q_dict.items():
            noisy_Q[k] = v + np.random.normal(0, dynamic_noise_scale * max(0.1, abs(v)))
            
        if sampler_type == "SQA":
            sampleset = sampler.sample_qubo(noisy_Q, num_reads=reads_per_batch, num_trotter_slices=10)
        else:
            sampleset = sampler.sample_qubo(noisy_Q, num_reads=reads_per_batch)
            
        for sample, _ in sampleset.data(["sample", "energy"]):
            pairs_list = decode_bits_to_pairs(sample, stems)
            if pairs_list is not None:
                all_results.append({"pairs_list": pairs_list, "qubo": 0}) 

    def eval_qubo(pairs_list):
        from phase8_paired_sampling import evaluate_qubo_from_pairs
        nested = flat_to_nested(pairs_list, stems)
        return evaluate_qubo_from_pairs(nested, stems, Q_dict, offset)
        
    for r in all_results:
        r["qubo"] = eval_qubo(r["pairs_list"])

    all_results.sort(key=lambda x: x["qubo"])
    
    ten_pct_idx = max(1, int(len(all_results) * 0.10))
    top_slice = all_results[:ten_pct_idx]
    top_unique = []
    seen = set()
    for r in top_slice:
        if r["pairs_list"] not in seen:
            seen.add(r["pairs_list"])
            top_unique.append(r)
    return top_unique

def get_random_pairs_for_experiment(target_structure, num_assignments=10):
    stems = extract_stems(target_structure)
    samples = generate_random_pairs(stems, num_assignments)
    return [{"pairs_list": flat_tuple, "qubo": None} for _, flat_tuple in samples]

def generate_classical_sequences(target_structure, pair_results, penalty_type, num_pairs, loops_per_pair):
    generated = []
    pair_results = pair_results[:num_pairs]
    
    for res in pair_results:
        candidates = fill_loops_custom(
            target_structure=target_structure,
            pair_assignment=res["pairs_list"],
            num_output=loops_per_pair,
            penalty_type=penalty_type
        )
        generated.extend(candidates)
        
    total_needed = num_pairs * loops_per_pair
    # Pad with dummy failures if annealer didn't find enough unique pairs
    while len(generated) < total_needed:
        generated.append("A" * len(target_structure)) 
        
    return generated[:total_needed]

def generate_extended_qubo_sequences(target_structure, c_coeffs, num_reads=1000, num_take=20, sampler_type="SA", penalty_type="mirror"):
    stems = extract_stems(target_structure)
    Q_ext, offset, loop_indices, fixed_loops = coaxial_build_extended_qubo(target_structure, stems, c_coeffs)
    
    if sampler_type == "SA":
        sampler = SimulatedAnnealingSampler()
    else:
        sampler = PathIntegralAnnealingSampler()
        
    num_ensembles = 5
    reads_per_ensemble = max(1, num_reads // num_ensembles)
    valid_sequences_with_energy = []
    seen = set()
    
    for e in range(num_ensembles):
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
                    pen = calculate_terminal_penalty(seq, target_structure)
                    if penalty_type == "entropy":
                        pen += calculate_loop_entropy_penalty_multistem(seq, target_structure)
                    elif penalty_type == "mirror":
                        pen += calculate_mirror_penalty_multistem(seq, target_structure)
                    elif penalty_type == "mirror_entropy":
                        pen += calculate_loop_entropy_penalty_multistem(seq, target_structure)
                        pen += calculate_mirror_penalty_multistem(seq, target_structure)
                    elif penalty_type == "shifted_mirror":
                        pen += calculate_shifted_mirror_penalty_multistem(seq, target_structure)
                    elif penalty_type == "shifted_mirror_entropy":
                        pen += calculate_loop_entropy_penalty_multistem(seq, target_structure)
                        pen += calculate_shifted_mirror_penalty_multistem(seq, target_structure)
                        
                    total_score = energy + offset + pen
                    valid_sequences_with_energy.append({'seq': seq, 'score': total_score})
                    
    valid_sequences_with_energy.sort(key=lambda x: x['score'])
    generated = [x['seq'] for x in valid_sequences_with_energy[:num_take]]
    
    while len(generated) < num_take:
        generated.append("A" * len(target_structure))
        
    return generated

def get_qubo_pairs_via_cplex(target_structure, c_coeffs, num_pairs=10):
    """
    Extracts top distinct valid stem pair assignments using IBM CPLEX Branch & Bound
    with solution exclusion cuts.
    """
    if not CPLEX_AVAILABLE:
        stems = extract_stems(target_structure)
        return [{"pairs_list": tuple(['AU'] * sum(len(s) for s in stems)), "qubo": 0.0}]

    stems = extract_stems(target_structure)
    Q_dict, offset = coaxial_build_approx_qubo(stems, c_coeffs)
    var_names = sorted(list(set([u for pair in Q_dict.keys() for u in pair])))

    mdl = Model(name="Stem_QUBO_CPLEX")
    mdl.context.solver.log_output = False
    x = mdl.binary_var_dict(var_names, name="x")

    objective_terms = []
    for (u, v), weight in Q_dict.items():
        if u == v:
            objective_terms.append(weight * x[u])
        else:
            objective_terms.append(weight * x[u] * x[v])
    mdl.minimize(mdl.sum(objective_terms))

    unique_pairs = []
    seen = set()

    for _ in range(num_pairs * 2):
        solution = mdl.solve()
        if not solution:
            break
        sample = {var: int(solution.get_value(x[var])) for var in var_names}
        pairs_list = decode_bits_to_pairs(sample, stems)

        if pairs_list is not None and pairs_list not in seen:
            seen.add(pairs_list)
            unique_pairs.append({"pairs_list": pairs_list, "qubo": solution.objective_value + offset})
            if len(unique_pairs) >= num_pairs:
                break

        # Integer exclusion cut to find next best solution
        active_vars = [x[v] for v in var_names if solution.get_value(x[v]) > 0.5]
        inactive_vars = [x[v] for v in var_names if solution.get_value(x[v]) <= 0.5]
        mdl.add_constraint(mdl.sum(active_vars) - mdl.sum(inactive_vars) <= len(active_vars) - 1)

    while len(unique_pairs) < num_pairs:
        unique_pairs.append({"pairs_list": tuple(['AU'] * sum(len(s) for s in stems)), "qubo": 0.0})

    return unique_pairs[:num_pairs]

def generate_extended_binary_sequences(target_structure, stem_coeffs, loop_coeffs, num_take=20, sampler_type="SA"):
    """
    Generates sequences using the Supervisor 2-bit Extended Binary QUBO.
    Supports SA, SQA, and CPLEX.
    """
    stems = extract_stems(target_structure)
    Q_ext, offset, loop_indices, fixed_loops = build_supervisor_qubo(
        target_structure, stems, stem_coeffs, loop_coeffs
    )

    valid_sequences_with_energy = []
    seen = set()

    if sampler_type == "CPLEX" and CPLEX_AVAILABLE:
        var_names = sorted(list(set([u for pair in Q_ext.keys() for u in pair])))
        mdl = Model(name="Ext_Binary_CPLEX")
        mdl.context.solver.log_output = False
        x = mdl.binary_var_dict(var_names, name="x")

        obj_terms = []
        for (u, v), w in Q_ext.items():
            if u == v:
                obj_terms.append(w * x[u])
            else:
                obj_terms.append(w * x[u] * x[v])
        mdl.minimize(mdl.sum(obj_terms))

        for _ in range(num_take):
            sol = mdl.solve()
            if not sol:
                break
            sample = {v: int(sol.get_value(x[v])) for v in var_names}
            decoded = decode_supervisor_sample(sample, stems, loop_indices, fixed_loops)
            if decoded:
                pairs_list, loop_assignment = decoded
                seq = generate_binary_sequence(target_structure, stems, pairs_list, loop_assignment)
                if seq not in seen:
                    seen.add(seq)
                    valid_sequences_with_energy.append({'seq': seq, 'score': sol.objective_value + offset})

            # Exclusion cut
            active = [x[v] for v in var_names if sol.get_value(x[v]) > 0.5]
            inactive = [x[v] for v in var_names if sol.get_value(x[v]) <= 0.5]
            mdl.add_constraint(mdl.sum(active) - mdl.sum(inactive) <= len(active) - 1)

    else:
        sampler = SimulatedAnnealingSampler() if sampler_type == "SA" else PathIntegralAnnealingSampler()
        num_reads = 3000
        if sampler_type == "SQA":
            sampleset = sampler.sample_qubo(Q_ext, num_reads=num_reads, num_trotter_slices=10)
        else:
            sampleset = sampler.sample_qubo(Q_ext, num_reads=num_reads)

        for sample, energy in sampleset.data(["sample", "energy"]):
            decoded = decode_supervisor_sample(sample, stems, loop_indices, fixed_loops)
            if decoded:
                pairs_list, loop_assignment = decoded
                seq = generate_binary_sequence(target_structure, stems, pairs_list, loop_assignment)
                if seq not in seen:
                    seen.add(seq)
                    valid_sequences_with_energy.append({'seq': seq, 'score': energy + offset})

    valid_sequences_with_energy.sort(key=lambda x: x['score'])
    generated = [x['seq'] for x in valid_sequences_with_energy[:num_take]]
    while len(generated) < num_take:
        generated.append("A" * len(target_structure))
    return generated

def generate_vienna_sequences(target_structure, num_output, timeout=5.0):
    """Uses a temporary script and subprocess to prevent Vienna hanging infinitely."""
    generated = []
    for _ in range(num_output):
        start_seq = "".join(random.choices("ACGU", k=len(target_structure)))
        
        script_code = f"""import RNA\nseq, dist = RNA.inverse_fold('{start_seq}', '{target_structure}')\nprint(seq)\n"""
        with open("temp_vienna.py", "w") as f:
            f.write(script_code)
            
        try:
            result = subprocess.run([sys.executable, "temp_vienna.py"], capture_output=True, text=True, timeout=timeout)
            seq = result.stdout.strip()
            if seq:
                generated.append(seq)
            else:
                generated.append(start_seq) # Use bad start_seq as failure
        except subprocess.TimeoutExpired:
            generated.append(start_seq) # Timed out, count as failure
            
    if os.path.exists("temp_vienna.py"):
        os.remove("temp_vienna.py")
    return generated

def generate_desirna_sequences(target_structure, num_output):
    input_file = "d.txt"
    seq_restr = "N" * len(target_structure)
    input_content = f">name\nDesign\n>seq_restr\n{seq_restr}\n>sec_struct\n{target_structure}\n"
    
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(input_content)
        
    # Use generic python path or specific if needed
    python_exe = sys.executable 
    desirna_script = r"DesiRNA/DesiRNA.py"
    
    cmd = [python_exe, desirna_script, "-f", input_file, "-t", "60", "-r", str(num_output)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    generated = []
    output_csvs = glob.glob("d_*/*_results.csv")
    if output_csvs:
        with open(output_csvs[0], 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = row.get("sequence", "")
                if seq:
                    generated.append(seq)
                    
    if os.path.exists(input_file):
        os.remove(input_file)
    for folder in glob.glob("d_*"):
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
            
    while len(generated) < num_output:
        generated.append("A" * len(target_structure))
        
    return generated[:num_output]

def evaluate_sequences(sequences, target):
    successes = 0
    final_sequences =[]
    fixed_count = 0
    fix_logs = []
    for seq in sequences:
        struct, _ = RNA.fold(seq)
        if struct == target:
            successes += 1
            final_sequences.append(seq)
        else:
            #try to rescue failed seq using post processing
            refined_seq,is_fixed = refine_sequence_targeted(seq,target)
            final_sequences.append(refined_seq)
            if is_fixed:
                successes +=1
                fixed_count += 1
                
                diff_pointers = "".join(['^' if c1 != c2 else ' ' for c1, c2 in zip(seq, refined_seq)])
                log_str = (
                    f"        Original: {seq}\n"
                    f"        Refined : {refined_seq}\n"
                    f"                  {diff_pointers}"
                )
                fix_logs.append(log_str)

            
    unique_seqs = list(set(final_sequences))
    unique_successes = 0
    for seq in unique_seqs:
        struct, _ = RNA.fold(seq)
        if struct == target:
            unique_successes += 1
            
    return len(final_sequences), successes, len(unique_seqs), unique_successes, fixed_count, fix_logs

def refine_sequence_targeted(seq,target_struct,max_mutations=3):
    """
    Attempts to fix a misfolded sequence by applying targeted mutations
    based on ensemble differential positions.
    """
    current_seq = list(seq)

    for _ in range(max_mutations):
        # 1. get the rival structure
        actual_struct,_ = RNA.fold("".join(current_seq))
        if actual_struct == target_struct:
            return "".join(current_seq),True # successfully fixed 
        # 2. Find differential positions
        diff_position  =[]
        for i ,(t_char,a_char) in enumerate(zip(target_struct,actual_struct)):
            if t_char != a_char:
                diff_position.append(i)
        # if the structure is completely different , it might be too far to fix 
        if len(diff_position) >15 or not diff_position:
            break

        # 3 Try single mutations only at diff_positions
        best_mutation =None
        best_energy_diff = float('inf')

        for pos in diff_position:
            original_base = current_seq[pos]
            for new_base in ['A','C','G','U']:
                if new_base == original_base:
                    continue
                # apply temporary mutation
                current_seq[pos] =new_base
                test_seq_str ="".join(current_seq)

                # fast eval : we wanr E_target to be lower than E_actual
                e_target = RNA.energy_of_struct(test_seq_str ,target_struct)
                e_actual = RNA.energy_of_struct(test_seq_str,actual_struct) # check the energy val with the rival struct
                energy_diff = e_target -e_actual

                if energy_diff <best_energy_diff:
                    best_energy_diff =energy_diff
                    best_mutation =(pos ,new_base)
                
                #revert temporary mutation
                current_seq[pos] = original_base
        #calculate current state's energy diff to ensure we aree improving
        current_seq_str = "".join(current_seq)

        # fast e_val : we want e_target to lower than e_actual
        current_e_target =RNA.energy_of_struct(current_seq_str,target_struct)
        current_e_actual =RNA.energy_of_struct(current_seq_str,actual_struct)
        curr_diff =current_e_target - current_e_actual
        # If we didn't find any mutation that improves the energy difference, stop early
        if best_mutation is None or best_energy_diff >=curr_diff:
            break

        # apply the best mutation permanently for this iternation
        current_seq[best_mutation[0]] =best_mutation[1] # i.e pos :best_mutation[0] ,value : best_mutation[1]
    
    final_seq = "".join(current_seq)
    final_struct,final_struct_energy =RNA.fold(final_seq)
    return final_seq,final_struct ==target_struct


def run_unified_benchmark():
    print("Pre-computing OLS coefficients...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    print("Pre-computing 2-bit loop coefficients for Extended Binary QUBO...")
    loop_coeffs = calculate_binary_loop_coeffs()
    
    csv_path = r"Structures/fmqa_paper_structures.csv"
    targets = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                name = f"{row['Category']} {i+1}"
                targets.append((name, row["Structure"]))
    else:
        print(f"Error: Could not find dataset at {csv_path}")
        return
    
    aggregate_results = defaultdict(lambda: defaultdict(lambda: {"raw_succ": [], "uniq_succ": [], "uniq_gen": [], "fixed": []}))
    
    for run in range(NUM_RUNS):
        print(f"\n{'#'*60}")
        print(f"--- STARTING RUN {run + 1} OF {NUM_RUNS} ---")
        print(f"{'#'*60}")
        
        for name, target in targets:
            print(f"\n[Run {run+1}] Testing {name}: {target}")
            
            def log_result(method_name, sequences):
                _, s, unq_gen, unq_s, fixed, fix_logs = evaluate_sequences(sequences, target)
                aggregate_results[name][method_name]["raw_succ"].append(s)
                aggregate_results[name][method_name]["uniq_succ"].append(unq_s)
                aggregate_results[name][method_name]["uniq_gen"].append(unq_gen)
                aggregate_results[name][method_name]["fixed"].append(fixed)
                print(f"  {method_name[:40]:<40} -> Raw: {s}/{TOTAL_SEQ_PER_METHOD} (Fixed: {fixed}) | Uniq: {unq_s}/{unq_gen}")
                if fix_logs:
                    print("      [Mutations that fixed sequences]:")
                    for log in fix_logs:
                        print(log)
                    print("")

            # 1. Baseline
            rand_pairs = get_random_pairs_for_experiment(target, NUM_QUBO_PAIRS)
            seqs = generate_classical_sequences(target, rand_pairs, "none", NUM_QUBO_PAIRS, LOOPS_PER_PAIR)
            log_result("Baseline (Random)", seqs)
            
            def run_qubo_var(method_name, sampler, penalty, is_noisy):
                if is_noisy:
                    pairs = get_qubo_pairs_via_noisy_annealing_coaxial(target, c_coeffs, num_reads=500, sampler_type=sampler)
                else:
                    pairs = get_qubo_pairs_via_annealing_coaxial(target, c_coeffs, num_reads=500, sampler_type=sampler)
                
                # Padding fallback just in case
                while len(pairs) < NUM_QUBO_PAIRS:
                    pairs.append({"pairs_list": tuple(['AU'] * sum([len(s) for s in extract_stems(target)]))})

                seqs = generate_classical_sequences(target, pairs, penalty, NUM_QUBO_PAIRS, LOOPS_PER_PAIR)
                log_result(method_name, seqs)

            def run_ext_var(method_name, sampler, penalty):
                seqs = generate_extended_qubo_sequences(target, c_coeffs, num_reads=1000, num_take=TOTAL_SEQ_PER_METHOD, sampler_type=sampler, penalty_type=penalty)
                log_result(method_name, seqs)

            # --- SA Variations ---
            # SA - Loop Entropy Alone
            run_qubo_var("Normal QUBO (SA) + Loop Entropy", "SA", "entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SA) + Loop Entropy", "SA", "entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SA) + Loop Entropy", "SA", "entropy")

            # SA - Mirror Alone
            run_qubo_var("Normal QUBO (SA) + Mirror", "SA", "mirror", is_noisy=False)
            run_qubo_var("Noisy QUBO (SA) + Mirror", "SA", "mirror", is_noisy=True)
            run_ext_var("Extended QUBO (SA) + Mirror", "SA", "mirror")

            # SA - Mirror & Entropy
            run_qubo_var("Normal QUBO (SA) + Mirror & Entropy", "SA", "mirror_entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SA) + Mirror & Entropy", "SA", "mirror_entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SA) + Mirror & Entropy", "SA", "mirror_entropy")
            
            # SA - Shifted Mirror
            run_qubo_var("Normal QUBO (SA) + Shifted Mirror", "SA", "shifted_mirror", is_noisy=False)
            run_qubo_var("Noisy QUBO (SA) + Shifted Mirror", "SA", "shifted_mirror", is_noisy=True)
            run_ext_var("Extended QUBO (SA) + Shifted Mirror", "SA", "shifted_mirror")

            # SA - Shifted Mirror & Entropy
            run_qubo_var("Normal QUBO (SA) + Shifted Mirror & Entropy", "SA", "shifted_mirror_entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SA) + Shifted Mirror & Entropy", "SA", "shifted_mirror_entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SA) + Shifted Mirror & Entropy", "SA", "shifted_mirror_entropy")
            
            # --- SQA Variations ---
            # SQA - Loop Entropy Alone
            run_qubo_var("Normal QUBO (SQA) + Loop Entropy", "SQA", "entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SQA) + Loop Entropy", "SQA", "entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SQA) + Loop Entropy", "SQA", "entropy")

            # SQA - Mirror Alone
            run_qubo_var("Normal QUBO (SQA) + Mirror", "SQA", "mirror", is_noisy=False)
            run_qubo_var("Noisy QUBO (SQA) + Mirror", "SQA", "mirror", is_noisy=True)
            run_ext_var("Extended QUBO (SQA) + Mirror", "SQA", "mirror")

            # SQA - Mirror & Entropy
            run_qubo_var("Normal QUBO (SQA) + Mirror & Entropy", "SQA", "mirror_entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SQA) + Mirror & Entropy", "SQA", "mirror_entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SQA) + Mirror & Entropy", "SQA", "mirror_entropy")
            
            # SQA - Shifted Mirror
            run_qubo_var("Normal QUBO (SQA) + Shifted Mirror", "SQA", "shifted_mirror", is_noisy=False)
            run_qubo_var("Noisy QUBO (SQA) + Shifted Mirror", "SQA", "shifted_mirror", is_noisy=True)
            run_ext_var("Extended QUBO (SQA) + Shifted Mirror", "SQA", "shifted_mirror")

            # SQA - Shifted Mirror & Entropy
            run_qubo_var("Normal QUBO (SQA) + Shifted Mirror & Entropy", "SQA", "shifted_mirror_entropy", is_noisy=False)
            run_qubo_var("Noisy QUBO (SQA) + Shifted Mirror & Entropy", "SQA", "shifted_mirror_entropy", is_noisy=True)
            run_ext_var("Extended QUBO (SQA) + Shifted Mirror & Entropy", "SQA", "shifted_mirror_entropy")

            # --- IBM CPLEX Exact Classical Variations ---
            if CPLEX_AVAILABLE:
                def run_cplex_var(method_name, penalty):
                    pairs = get_qubo_pairs_via_cplex(target, c_coeffs, num_pairs=NUM_QUBO_PAIRS)
                    seqs = generate_classical_sequences(target, pairs, penalty, NUM_QUBO_PAIRS, LOOPS_PER_PAIR)
                    log_result(method_name, seqs)

                run_cplex_var("Normal QUBO (CPLEX) + Loop Entropy", "entropy")
                run_cplex_var("Normal QUBO (CPLEX) + Mirror", "mirror")
                run_cplex_var("Normal QUBO (CPLEX) + Mirror & Entropy", "mirror_entropy")
                run_cplex_var("Normal QUBO (CPLEX) + Shifted Mirror", "shifted_mirror")
                run_cplex_var("Normal QUBO (CPLEX) + Shifted Mirror & Entropy", "shifted_mirror_entropy")

            # --- Extended Binary QUBO (Supervisor Methodology) ---
            def run_ext_bin_var(method_name, sampler):
                seqs = generate_extended_binary_sequences(target, c_coeffs, loop_coeffs, num_take=TOTAL_SEQ_PER_METHOD, sampler_type=sampler)
                log_result(method_name, seqs)

            run_ext_bin_var("Extended Binary QUBO (SA)", "SA")
            run_ext_bin_var("Extended Binary QUBO (SQA)", "SQA")
            if CPLEX_AVAILABLE:
                run_ext_bin_var("Extended Binary QUBO (CPLEX)", "CPLEX")
            
            # --- Traditional Comparative Tools ---
            # Vienna
            seqs = generate_vienna_sequences(target, num_output=TOTAL_SEQ_PER_METHOD, timeout=5.0)
            log_result("Vienna RNAinverse", seqs)
            
            # DesiRNA
            seqs = generate_desirna_sequences(target, num_output=TOTAL_SEQ_PER_METHOD)
            log_result("DesiRNA", seqs)

    # Averages
    print("\n\n" + "="*100)
    print(f"FINAL AVERAGED RESULTS OVER {NUM_RUNS} RUNS (Total {TOTAL_SEQ_PER_METHOD} sequences per target/method)")
    print("="*100)
    
    with open('final_benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target', 'Methodology', 'Avg_Raw_Success', 'Raw_Success_Rate_%', 'Avg_Fixed_by_Mut', 'Avg_Uniq_Success', 'Avg_Uniq_Generated', 'Uniq_Success_Rate_%'])
        
        for name, target in targets:
            print(f"\nTarget: {name}")
            print(f"{'Methodology':<40} | {'Raw Succ (Avg)':<15} | {'Fixed (Avg)':<12} | {'Uniq Succ/Gen (Avg)':<20}")
            print("-" * 95)
            for method, data in aggregate_results[name].items():
                avg_raw_s = np.mean(data["raw_succ"])
                avg_unq_s = np.mean(data["uniq_succ"])
                avg_unq_g = np.mean(data["uniq_gen"])
                avg_fixed = np.mean(data["fixed"])
                
                raw_rate = (avg_raw_s / TOTAL_SEQ_PER_METHOD) * 100
                unq_rate = (avg_unq_s / avg_unq_g * 100) if avg_unq_g > 0 else 0.0
                
                print(f"{method:<40} | {avg_raw_s:>5.1f} / {TOTAL_SEQ_PER_METHOD:<7} | {avg_fixed:>5.1f}        | {avg_unq_s:>5.1f} / {avg_unq_g:<5.1f} ({unq_rate:>5.1f}%)")
                writer.writerow([name, method, f"{avg_raw_s:.2f}", f"{raw_rate:.2f}", f"{avg_fixed:.2f}", f"{avg_unq_s:.2f}", f"{avg_unq_g:.2f}", f"{unq_rate:.2f}"])
        
if __name__ == "__main__":
    run_unified_benchmark()