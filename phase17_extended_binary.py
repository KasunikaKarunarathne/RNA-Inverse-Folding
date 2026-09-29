import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from collections import defaultdict
import RNA
import re
import csv
import os

from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase10_annealin import decode_bits_to_pairs
from phase2_turner_energy import get_turner_energy

# Supervisor's 2-bit loop encoding mapping (Matches loop_penalty.py exactly)
BASE_BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "U": (1, 1)}
BITS_BASE = {v: k for k, v in BASE_BITS.items()}

def calculate_structural_difficulty(target_structure, stems):
    num_unpaired = target_structure.count('.')
    num_stems = len(stems)
    total_length = len(target_structure)
    
    unpaired_ratio = num_unpaired / total_length if total_length > 0 else 0
    difficulty = 1.0
    if unpaired_ratio > 0.4: difficulty += 1.0
    if num_stems > 1: difficulty += 1.5
    for stem in stems:
        if len(stem) < 3: difficulty += 0.5
    return difficulty

def calculate_binary_loop_coeffs():
    """
    Supervisor's pure QUBO methodology:
    Fits a 37-term quadratic model for 4 loop bases (8 binary variables) using OLS regression.
    """
    from itertools import product
    
    num_states = 256
    num_coeffs = 37 # 1 bias + 8 linear + 28 quadratic interactions
    
    Phi = np.zeros((num_states, num_coeffs))
    E_true = np.zeros(num_states)
    
    combinations = list(product([0, 1], repeat=8))
    valid_pairs = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG']
    
    for row_i, bits in enumerate(combinations):
        b1 = BITS_BASE[(bits[0], bits[1])]
        b2 = BITS_BASE[(bits[2], bits[3])]
        b3 = BITS_BASE[(bits[4], bits[5])]
        b4 = BITS_BASE[(bits[6], bits[7])]
        
        pair1 = b1 + b4
        pair2 = b2 + b3
        
        # Calculate Exact Turner penalty for accidental loop stacks
        if pair1 in valid_pairs and pair2 in valid_pairs:
            energy = get_turner_energy(pair1, pair2)
            E_true[row_i] = max(0.0, -energy)
        else:
            E_true[row_i] = 0.0
            
        features = [1.0] # Bias
        features.extend(bits) # Linear terms
        for i in range(8):
            for j in range(i+1, 8):
                features.append(bits[i] * bits[j]) # Quadratic terms
        
        Phi[row_i, :] = features
        
    c_coeffs, _, _, _ = np.linalg.lstsq(Phi, E_true, rcond=None)
    return c_coeffs

def apply_loop_stack_penalty(Q_ext, offset, v_list, loop_coeffs, weight):
    """
    Applies the fitted 37-term 2nd-degree model directly to the 8 variables.
    No auxiliary variables needed!
    """
    offset += loop_coeffs[0] * weight # Bias
    
    for i in range(8):
        var = v_list[i]
        Q_ext[(var, var)] = Q_ext.get((var, var), 0) + loop_coeffs[1 + i] * weight # Linear
        
    idx = 9
    for i in range(8):
        for j in range(i+1, 8):
            var1, var2 = v_list[i], v_list[j]
            key = tuple(sorted([var1, var2]))
            Q_ext[key] = Q_ext.get(key, 0) + loop_coeffs[idx] * weight # Quadratic
            idx += 1
            
    return offset

def build_supervisor_qubo(target_structure, stems, stem_coeffs, loop_coeffs):
    # 1. Stems (Pure QUBO)
    Q_stem, offset = build_approx_qubo(stems, stem_coeffs)
    Q_ext = defaultdict(float)
    
    for (u, v), bias in Q_stem.items():
        key = (u, u) if u == v else tuple(sorted((u, v)))
        Q_ext[key] += bias

    # 2. Tetraloops
    tetraloop_stems = []
    fixed_loops = {}
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
        P_force = 1000.0
        Q_ext[(f"p_{left}_{right}_0", f"p_{left}_{right}_0")] -= P_force
        Q_ext[(f"p_{left}_{right}_1", f"p_{left}_{right}_1")] += P_force
        Q_ext[(f"p_{left}_{right}_2", f"p_{left}_{right}_2")] -= P_force

    paired_indices = set()
    for stem in stems:
        for (left, right) in stem:
            paired_indices.add(left)
            paired_indices.add(right)
            
    loop_indices = [i for i in range(len(target_structure)) 
                    if i not in paired_indices and i not in fixed_loops]
                    
    difficulty_multiplier = calculate_structural_difficulty(target_structure, stems)
    
    # 3. Binary Bias Tax
    # Tax mapping exactly matching supervisor: A=0, C=5, G=15, U=5
    # The algebraic equivalent for (b0, b1) mapping is: 15*b0 + 5*b1 - 15*b0*b1
    for idx in loop_indices:
        b0 = f"b_{idx}_0"
        b1 = f"b_{idx}_1"
        Q_ext[(b0, b0)] += 15.0
        Q_ext[(b1, b1)] += 5.0
        key = tuple(sorted((b0, b1)))
        Q_ext[key] -= 15.0

    # 4. Loop Stack Entropy Penalties (Supervisor method)
    # Penalize valid Turner stacks (4 bases) instead of just single pairs
    P_anti = 50.0 * difficulty_multiplier
    
    regions = []
    start = None
    for i, char in enumerate(target_structure):
        if char == '.' and start is None: start = i
        elif char != '.' and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None: regions.append((start, len(target_structure) - 1))
    
    for (r_start, r_end) in regions:
        length = r_end - r_start + 1
        for i in range(1, max(1, length - 5)):
            for j in range(i + 6, length + 1):
                abs_i = r_start + i - 1
                abs_j = r_start + j - 1
                
                if (abs_i in loop_indices and abs_i+1 in loop_indices and 
                    abs_j-1 in loop_indices and abs_j in loop_indices):
                    
                    v_list = [
                        f"b_{abs_i}_0", f"b_{abs_i}_1",
                        f"b_{abs_i+1}_0", f"b_{abs_i+1}_1",
                        f"b_{abs_j-1}_0", f"b_{abs_j-1}_1",
                        f"b_{abs_j}_0", f"b_{abs_j}_1"
                    ]
                    offset = apply_loop_stack_penalty(Q_ext, offset, v_list, loop_coeffs, P_anti)

    # 5. Mirror Stack Penalty (Supervisor method)
    P_mirror = 100.0 * difficulty_multiplier
    
    for stem in stems:
        left, right = stem[-1]
        L = right - left - 1
        m = L // 2
        
        for d in range(1, m): # Step inward
            curr_l = left + d
            curr_r = right - d
            
            if curr_l >= curr_r - 1: break # We need 4 bases to form a stack!
            
            if (curr_l in loop_indices and curr_l+1 in loop_indices and 
                curr_r-1 in loop_indices and curr_r in loop_indices):
                
                w_d = 1.0 - 0.5 * ((d - 1) / (m - 1)) if m > 1 else 1.0
                
                v_list = [
                    f"b_{curr_l}_0", f"b_{curr_l}_1",
                    f"b_{curr_l+1}_0", f"b_{curr_l+1}_1",
                    f"b_{curr_r-1}_0", f"b_{curr_r-1}_1",
                    f"b_{curr_r}_0", f"b_{curr_r}_1"
                ]
                offset = apply_loop_stack_penalty(Q_ext, offset, v_list, loop_coeffs, P_mirror * w_d)

    return dict(Q_ext), offset, loop_indices, fixed_loops

def decode_supervisor_sample(sample, stems, loop_indices, fixed_loops):
    pairs_list = decode_bits_to_pairs(sample, stems)
    if pairs_list is None:
        return None
        
    loop_assignment = {}
    for idx in loop_indices:
        b0 = sample.get(f"b_{idx}_0", 0)
        b1 = sample.get(f"b_{idx}_1", 0)
        loop_assignment[idx] = BITS_BASE[(b0, b1)]
        
    for idx, b in fixed_loops.items():
        loop_assignment[idx] = b
        
    return pairs_list, loop_assignment

def generate_sequence(target_structure, stems, pairs_list, loop_assignment):
    seq = [""] * len(target_structure)
    pair_idx = 0
    for stem in stems:
        for (left, right) in stem:
            seq[left] = pairs_list[pair_idx][0]
            seq[right] = pairs_list[pair_idx][1]
            pair_idx += 1
    for idx, b in loop_assignment.items():
        seq[idx] = b
    return "".join(seq)

def run_fmqa_benchmark():
    csv_path = r"d:\Academic UOP\Internship\simulation\Implementation\NN - Copy\Structures\fmqa_paper_structures.csv"
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

    print("Pre-computing OLS coefficients for Turner energy (Stems)...")
    stem_coeffs = calculate_qubo_coeffs(method="ols")
    
    print("Pre-computing OLS coefficients for Accidental Stacks (Loops)...")
    loop_coeffs = calculate_binary_loop_coeffs()
    
    results_table = []
    top10_file = open("phase17_supervisor_binary_top10.txt", "w")
    
    for name, target in targets:
        stems = extract_stems(target)
        print(f"\n==============================================")
        print(f"Testing {name}: {target}")
        print(f"==============================================")
        top10_file.write(f"\nTarget: {name} | {target}\n{'='*60}\n")
        
        print("Building Pure QUBO (Supervisor Methodology)...")
        Q_ext, offset, loop_indices, fixed_loops = build_supervisor_qubo(target, stems, stem_coeffs, loop_coeffs)
        
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q_ext, num_reads=5000)
        
        seen = set()
        valid_sequences_with_energy = []
        
        for sample, energy in sampleset.data(["sample", "energy"]):
            decoded = decode_supervisor_sample(sample, stems, loop_indices, fixed_loops)
            if decoded:
                pairs_list, loop_assignment = decoded
                seq = generate_sequence(target, stems, pairs_list, loop_assignment)
                if seq not in seen:
                    seen.add(seq)
                    valid_sequences_with_energy.append({'seq': seq, 'energy': energy + offset})
                    
        valid_sequences_with_energy.sort(key=lambda x: x['energy'])
        
        successes = 0
        top10_seqs = []
        
        for item in valid_sequences_with_energy[:100]:
            seq = item['seq']
            struct, mfe = RNA.fold(seq)
            if struct == target:
                successes += 1
                if len(top10_seqs) < 10:
                    top10_seqs.append(f"{seq} (Energy: {item['energy']:.2f}, MFE: {mfe:.2f})")
                    
        tot = max(1, len(valid_sequences_with_energy))
        results_table.append((name, tot, successes))
        top10_file.write(f"Top Sequences:\n" + "\n".join(top10_seqs) + "\n\n")
        print(f"  -> {successes}/{tot} valid unique reads ({(successes/tot*100):.1f}%) folded to target")

    top10_file.close()

    print("\n\nFINAL FMQA BENCHMARK RESULTS (SUPERVISOR METHODOLOGY)")
    print("-" * 60)
    print(f"{'Target':<20} | {'Total Unique':<12} | {'Successes':<10} | {'Rate %'}")
    print("-" * 60)
    for name, tot, succ in results_table:
        print(f"{name:<20} | {tot:<12} | {succ:<10} | {(succ/tot*100):.1f}%")

if __name__ == "__main__":
    run_fmqa_benchmark()
