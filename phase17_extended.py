import neal
import RNA
from collections import defaultdict
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase10_annealin import decode_bits_to_pairs

import re

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
        fixed_loops[idx] = 'G'
        fixed_loops[idx+1] = 'C'
        fixed_loops[idx+2] = 'A'
        fixed_loops[idx+3] = 'A'
        
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
    
    # Weights for the penalties
    P_onehot = 100.0  # Force exactly 1 base per position
    P_anti = 50.0     # Penalize unwanted loop-loop pairs
    
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
            
            # RNA needs 3 unpaired bases to make a hairpin bend
            if abs(idx1 - idx2) >= 4:
                for b1, b2 in valid_pairs:
                    var1 = f"x_{idx1}_{b1}"
                    var2 = f"x_{idx2}_{b2}"
                    key = tuple(sorted([var1, var2]))
                    Q_ext[key] += P_anti
                    
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
    
    sampler = neal.SimulatedAnnealingSampler()
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

if __name__ == "__main__":
    target = ".(((((........)((((....))))..))))......."
    print(f"TARGET: {target}")
    evaluate_extended_qubo(target, num_reads=10000)



