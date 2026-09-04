from collections import defaultdict
from phase17_extended import build_extended_qubo, decode_extended_sample, generate_sequence

from numpy import random
# pyrefly: ignore [missing-import]
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import PathIntegralAnnealingSampler
import RNA
from collections import defaultdict
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase10_annealin import decode_bits_to_pairs
import re

def calculate_structural_difficulty(target_structure, stems):
    num_unpaired = target_structure.count('.')
    num_stems = len(stems)
    total_length = len(target_structure)
    
    unpaired_ratio = num_unpaired / total_length if total_length > 0 else 0
    
    # Base difficulty
    difficulty = 1.0
    
    # If high proportion of unpaired bases (lots of loops), kissing loops are highly likely
    if unpaired_ratio > 0.4:
        difficulty += 1.0
        
    # If multiple stems, kissing loops are possible across stems
    if num_stems > 1:
        difficulty += 1.5
        
    # Extra difficulty for very short stems (length < 3)
    for stem in stems:
        if len(stem) < 3:
            difficulty += 0.5
            
    return difficulty

def build_extended_qubo(target_structure, stems, c_coeffs):
    # 1. Start with the standard stem QUBO
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)
    Q_ext = defaultdict(float, Q_dict)
    
    # 2. Identify Tetraloops and Force C-G Closing Pairs (Positive Design)
    tetraloop_stems = []
    fixed_loops = {}
    
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
        # Force the QUBO to pick CG (1, 0, 1) for the closing pair.
        # Penalty equation: P * (1 - p0) + P * p1 + P * (1 - p2)
        # Simplifies to: -P*p0 + P*p1 - P*p2  (ignoring the constant 2P)
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
            
    # Exclude paired indices AND our fixed tetraloop indices
    loop_indices = [i for i in range(len(target_structure)) 
                    if i not in paired_indices and i not in fixed_loops]
    
    bases = ["A", "C", "G", "U"]
    
    # Weights for the penalties dynamically scaled by difficulty
    difficulty_multiplier = calculate_structural_difficulty(target_structure, stems)
    P_onehot = 100.0 * max(1.0, difficulty_multiplier * 0.5)
    P_anti = 50.0 * difficulty_multiplier
    
    # 4. One-Hot Penalty and Linear Bias
    # We add a small tax to C, G, U to strongly encourage the annealer to pick A for loops,
    # preventing loops from pairing with GC-heavy stems.
    bias_tax = {"A": 0.0, "C": 5.0, "G": 15.0, "U": 5.0}
    
    for idx in loop_indices:
        for b in bases:
            var = f"x_{idx}_{b}"
            Q_ext[(var, var)] += (-P_onehot + bias_tax[b])
            
        for i in range(len(bases)):
            for j in range(i+1, len(bases)):
                var1 = f"x_{idx}_{bases[i]}"
                var2 = f"x_{idx}_{bases[j]}"
                # Sort keys to prevent duplicate undirected edges
                key = tuple(sorted([var1, var2]))
                Q_ext[key] += 2 * P_onehot

    # 5. Anti-Pairing Penalty (Negative Design)
    # Penalize valid pairs between ANY remaining loops >= 4 apart
    valid_pairs = [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]


    for i in range(len(loop_indices)):
        for j in range(i+1, len(loop_indices)):
            idx1 = loop_indices[i]
            idx2 = loop_indices[j]
            distance = abs(idx1-idx2)
            
            # RNA needs 3 unpaired bases to make a hairpin bend (>= 4).
            if distance >= 4:
                # --- Dynamic Decomposition (Probabilistic Bound) ---
                probabilistic_weight = P_anti * (4.0 / distance) 
                
                # --- Modulo-Separability (Anti-Isolation) ---
                has_inner_neighbor = (idx1 + 1 in loop_indices) and (idx2 - 1 in loop_indices)
                has_outer_neighbor = (idx1 - 1 in loop_indices) and (idx2 + 1 in loop_indices)
                is_isolated = not (has_inner_neighbor or has_outer_neighbor)
                
                penalty = probabilistic_weight * (2.0 if is_isolated else 1.0)
                
                for b1, b2 in valid_pairs:
                    var1 = f"x_{idx1}_{b1}"
                    var2 = f"x_{idx2}_{b2}"
                    key = tuple(sorted([var1, var2]))
                    Q_ext[key] += penalty
                    
    return dict(Q_ext), offset, loop_indices, fixed_loops

def decode_extended_sample(sample, stems, loop_indices, fixed_loops):
    """Decodes the massive binary state into a full RNA sequence"""
    # 1. Decode stems using existing function
    pairs_list = decode_bits_to_pairs(sample, stems)
    if pairs_list is None:
        return None # Invalid stem encoding (e.g., both p0 and p1 are 0)
        
    # 2. Decode loops
    loop_assignment = {}
    bases = ["A", "C", "G", "U"]
    for idx in loop_indices:
        active_bases = []
        for b in bases:
            var = f"x_{idx}_{b}"
            if sample.get(var, 0) == 1:
                active_bases.append(b)
                
        # If the annealer failed the one-hot constraint, this sequence is invalid
        if len(active_bases) != 1:
            return None 
        loop_assignment[idx] = active_bases[0]
        
    # 3. Add fixed loops (Tetraloops)
    for idx, b in fixed_loops.items():
        loop_assignment[idx] = b
        
    return pairs_list, loop_assignment

def generate_sequence(target_structure, stems, pairs_list, loop_assignment):
    seq = [""] * len(target_structure)
    
    # Fill stems
    pair_idx = 0
    for stem in stems:
        for (left, right) in stem:
            seq[left] = pairs_list[pair_idx][0]
            seq[right] = pairs_list[pair_idx][1]
            pair_idx += 1
            
    # Fill loops
    for idx, b in loop_assignment.items():
        seq[idx] = b
        
    return "".join(seq)

def evaluate_extended_qubo(target_structure, num_reads=1000):
    stems = extract_stems(target_structure)
    print("Fitting OLS coefficients for Turner energy...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    print("Building Extended QUBO...")
    Q_ext, offset, loop_indices, fixed_loops = build_extended_qubo(target_structure, stems, c_coeffs)
    
    print(f"Extended QUBO built with {len(Q_ext)} interactions.")
    print(f"Simulating Quantum Annealing ({num_reads} reads)...")
    
    sampler = SimulatedAnnealingSampler()
    # sampler = PathIntegralAnnealingSampler()
    sampleset = sampler.sample_qubo(Q_ext, num_reads=num_reads)
    
    results = []
    seen = set()
    
    for sample, energy in sampleset.data(["sample", "energy"]):
        decoded = decode_extended_sample(sample, stems, loop_indices, fixed_loops)
        if decoded:
            pairs_list, loop_assignment = decoded
            seq = generate_sequence(target_structure, stems, pairs_list, loop_assignment)
            if seq not in seen:
                seen.add(seq)
                results.append((seq, energy + offset))
                
    # Sort by the final quantum energy
    results.sort(key=lambda x: x[1])
    
    print(f"\nEvaluating Top 10 unique sequences out of {len(results)} valid quantum reads:")
    print("=" * 70)
    success_count = 0
    top_10 = []
    for i, (seq, energy) in enumerate(results[:10]):
        struct, mfe = RNA.fold(seq)
        match = (struct == target_structure)
        if match: success_count += 1
        print(f"Rank {i+1}: {seq}")
        print(f"  Q-Energy: {energy:.2f} | MFE: {mfe:.2f} | Match: {match}")
        if not match:
            print(f"  Got Fold: {struct}")
        print("-" * 70)
        
        top_10.append({
            "rank": i+1,
            "seq": seq,
            "q_energy": energy,
            "mfe": mfe,
            "match": match,
            "got_fold": struct
        })
            
    print(f"\nTotal Successful Folds (Top 10): {success_count}/10")
    return success_count, len(results), top_10




def build_approx_qubo(stems,c_coeffs,penalty_weight=100):
    """
    Constructs the 2-local QUBO dictionary including 5'-3' Asymmetry and Coaxial Stacking.
    stems: List of stems. Each stem is a list of base pairs, e.g. [(i, j), (i+1, j-1), ...]
    c_coeffs: The 22 weights calculated in Phase 3.
    """
    Q = defaultdict(float)
    constant_offset =0
    # helper function to apply the 22 QUBO coeffs to any tow pairs 
    def apply_stacking_qubo(pair1,pair2,coeffs):
        q_vars = [
            f"p_{pair1[0]}_{pair1[1]}_0", f"p_{pair1[0]}_{pair1[1]}_1", f"p_{pair1[0]}_{pair1[1]}_2",
            f"p_{pair2[0]}_{pair2[1]}_0", f"p_{pair2[0]}_{pair2[1]}_1", f"p_{pair2[0]}_{pair2[1]}_2"
        ]
        local_offset = coeffs[0]
        coeff_i =1

        # add c_a
        for a in range(6):
            var = q_vars[a]
            Q[(var,var)] += coeffs[coeff_i]
            coeff_i +=1
        
        #add c_ab
        for a in range(6):
            for b in range(a+1,6):
                key = tuple(sorted([q_vars[a],q_vars[b]]))
                Q[key] += coeffs[coeff_i]
                coeff_i +=1
        return local_offset

    # invalid state penalties and intra stem stacking 
    for stem in stems:
        for pair in stem:
            p0 =f"p_{pair[0]}_{pair[1]}_0"
            p1 =f"p_{pair[0]}_{pair[1]}_1"

            constant_offset +=penalty_weight
            Q[(p0,p0)]-= penalty_weight
            Q[(p1,p1)] -= penalty_weight
            Q[tuple(sorted([p0,p1]))] += penalty_weight
        for k in range(len(stem)-1):
            pair1= stem[k]
            pair2 = stem[k+1]
            constant_offset += apply_stacking_qubo(pair1,pair2,c_coeffs)

     # 2. INTER-STEM COAXIAL STACKING (New Logic)
    # We check all pairs of stems to see if they are 'flush' (adjacent in 3D).
    # Specifically, we check if the innermost pair of Stem A is adjacent to the outermost pair of Stem B.
    for i in range(len(stems)):
        for j in range(len(stems)):
            if i ==j:
                continue
            stem_A = stems[i]
            stem_B = stems[j]

            # Get the terminal pairs (the outer edge and inner edge of each stem)
            terminals_A = [stem_A[0], stem_A[-1]]
            terminals_B = [stem_B[0], stem_B[-1]]

            is_flush = False
            flush_pair_A = None
            flush_pair_B = None
            # Check if any terminal pair of A is exactly adjacent to any terminal pair of B
            for tA in terminals_A:
                for tB in terminals_B:
                    # Are they adjacent on the left or the right?
                    if (tB[0] == tA[1] + 1) or (tB[1] == tA[0] - 1) or (tA[0] == tB[1] + 1) or (tA[1] == tB[0] - 1):
                        is_flush = True
                        flush_pair_A = tA
                        flush_pair_B = tB
                        break
                if is_flush:
                    break
            if is_flush:
                print(f"Coaxial stacking detected between Stem {i} {flush_pair_A} and Stem {j} {flush_pair_B}")
                # We apply the EXACT SAME 22-term QUBO mapping between the two independent stems!
                constant_offset += apply_stacking_qubo(flush_pair_A, flush_pair_B, c_coeffs)
    return dict(Q), constant_offset
