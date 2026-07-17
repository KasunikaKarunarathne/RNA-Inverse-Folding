import os 
import random
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import spearmanr
import RNA

from phase1_rules import extract_stems, PAIR_ENCODING,ALLOWED_PAIRS
from phase2_turner_energy import calculate_true_turner_energy
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo

# def get_vienna_energy(sequence, structure):
#     """
#     Calculates the exact free energy of a sequence folded into a specific
#     structure using the official vienna packge
#     """
#     fc = RNA.fold_compound(sequence)
#     return fc.eval_structure(structure)

def generate_random_sequences(target_structure , num_samples):
    """
    Generate N random RNA sequences that physcally fir into the target structure 
    fill loops with dummy As
    """
    stems = extract_stems(target_structure)
    seq_length = len(target_structure)
    samples =[]

    for i in range(num_samples):
        # start with blank sequence of As 
        seq = ["A"] * seq_length
        for stem in stems:
            for pair in stem:
                random_pair = random.choice(ALLOWED_PAIRS)
                seq[pair[0]] = random_pair[0]
                seq[pair[1]] = random_pair[1]

        samples.append("".join(seq))

    # remove duplicates 
    return list(set(samples))

def evaluate_sequence_in_qubo(sequence , stems,Q_dict , offset):
    """
    Translates the sequence into binary bits, runs it through the
    Q_dict matrix,returns the QUBO predicted energy 
    """
    # Figure out which binary binary variables are active(equal to 1)
    active_vars = set()

    for stem in stems:
        for pair in stem:
            base1 = sequence[pair[0]]
            base2 = sequence[pair[1]]
            pair_string = f"{base1}{base2}"

            if pair_string not in PAIR_ENCODING:
                continue
            # get the 3 bit binary code ex GC = 1,0,0
            bits = PAIR_ENCODING[pair_string]

            # is the bit is 1 then add it into active var set
            if bits[0] == 1: active_vars.add(f"p_{pair[0]}_{pair[1]}_0")
            if bits[1] == 1: active_vars.add(f"p_{pair[0]}_{pair[1]}_1")
            if bits[2] == 1: active_vars.add(f"p_{pair[0]}_{pair[1]}_2")

    # calculate the energy by multiplying the active variables against the matrix
    qubo_energy = offset
    for (var1,var2) , weight in Q_dict.items():
        # the weight only applied if both binary variables are turned on 
        if var1 in active_vars and var2 in active_vars:
            qubo_energy += weight 

    return qubo_energy

def run_statistical_sampling(target_structure, num_samples,math_method="ols"):
    """
    Pipeline 
    Generate sequences , scores them both ways (HUBO , QUBO) and
    plots the correlation plots
    """
    print(f"----Analysis for target structure : {target_structure}----")
    print(f"Mathematical method used : {math_method}")

    stems = extract_stems(target_structure)

    # 1. generate the coeffs and build the matrix
    c_coeffs = calculate_qubo_coeffs(method= math_method)
    Q_dict , offset = build_approx_qubo(stems, c_coeffs)

    # 2. Generate random sequences 
    sequences = generate_random_sequences(target_structure, num_samples)

    print(f"Generated {len(sequences)} unique valid sequences.")
    combined_results = []
    true_energies =[]
    qubo_energies = []

    # 3. Score every sequence
    print("Scoring sequences with HUBO and QUBO.")
    for seq in sequences:
        # score HUBO
        t_energy = calculate_true_turner_energy(stems,seq)
        # t_energy = get_vienna_energy(seq, target_structure)
        true_energies.append(t_energy)

        # Score QUBO
        q_energy = evaluate_sequence_in_qubo(seq, stems, Q_dict,offset)
        qubo_energies.append(q_energy)
        combined_results.append({'seq': seq, 'true':t_energy,'qubo':q_energy})

    # QUARTILE AND BOTTOM 10%
    # sort everything by QUBO energy (real quantum comp. read out the lowest qubo first)

    qubo_sorted = sorted(combined_results, key= lambda x: x['qubo'])

    n_seq = len(qubo_sorted)
    q_size = n_seq // 4 
    ten_percent_idx = int(n_seq * 0.10)

    # Extract overall arrays 
    all_true = [x['true'] for x in qubo_sorted]
    all_qubo = [x['qubo'] for x in qubo_sorted]
    overall_corr , _ = spearmanr(all_true, all_qubo)

    print("\n ------ QUARTILE BREAKDOWN -------")
    print(f"Overall Full Landscape Correlation : {overall_corr: .4f}")

    for i in range(4):
        chunk = qubo_sorted[i*q_size:(i+1)*q_size]
        c_true = [x['true'] for x in chunk]
        c_qubo = [x['qubo'] for x in chunk]
        corr,_ = spearmanr(c_true,c_qubo)
        print(f"Quartile {i+1} (Qubo energies {c_qubo[0]:.2f} to {c_qubo[-1]:.2f}):spearman = {corr:.4f}")

    # Isolate bottom 10%
    bottom_10_chunk = qubo_sorted[:ten_percent_idx]
    b10_true = [x['true'] for x in bottom_10_chunk]
    b10_qubo = [x['qubo'] for x in bottom_10_chunk]
    b10_corr,_ = spearmanr(b10_true,b10_qubo)
    print("\n--- THE 10% GROUND STATE TEST ---")
    print(f"Spearman Correlation (Lowest 10% of QUBO): {b10_corr:.4f}")

    # Tail overlap 
    # Are the sequences in the lowest 10% of QUBO space the exact same sequences in the lowest 10% of biological space
    true_sorted = sorted(combined_results,key= lambda x : x['true'])

    bottom_10_qubo_seq = set([x['seq'] for x in qubo_sorted[:ten_percent_idx]])
    bottom_10_true_seq = set([x['seq'] for x in true_sorted[:ten_percent_idx]])
    
    # print(bottom_10_qubo_seq)
    # print(bottom_10_true_seq)

    overlap = bottom_10_qubo_seq.intersection(bottom_10_true_seq)
    overlap_pct = (len(overlap)/ten_percent_idx)*100

    print("\n--- TAIL OVERLAP ANALYSIS ---")
    print(f"Sequences in Top 10% QUBO set : {ten_percent_idx}")
    print(f"Sequences in Top 10% True set : {ten_percent_idx}")
    print(f"Exact Matches in both tails   : {len(overlap)}")
    print(f"Tail Overlap Percentage       : {overlap_pct:.1f}%")
    print("="*60 + "\n")
    
    # =================================================================
    # MISSED VALUES ANALYSIS (Non-Overlapping Sequences)
    # =================================================================
    # 1. Sequences in True Top 10% but missed by QUBO (False Negatives)
    missed_by_qubo = bottom_10_true_seq - bottom_10_qubo_seq 

    # 2. Sequences in QUBO top 10% but not in true top 10% (False Positives)
    false_positives_qubo = bottom_10_qubo_seq - bottom_10_true_seq

    # create a lookup to get energies by sequence
    seq_lookup = {item['seq']:item for item in combined_results}

    # Find the boundaries for the Top 10% sets
    qubo_min = qubo_sorted[0]['qubo']
    qubo_max = qubo_sorted[ten_percent_idx - 1]['qubo']
    qubo_mid = (qubo_min + qubo_max) / 2.0
    
    true_min = true_sorted[0]['true']
    true_max = true_sorted[ten_percent_idx - 1]['true']
    true_mid = (true_min + true_max) / 2.0

    print("\n--- Top 10% Boundaries ---")
    print(f"True (HUBO) 10% Range: {true_min:.4f} to {true_max:.4f} (Midpoint: {true_mid:.4f})")
    print(f"QUBO 10% Range       : {qubo_min:.4f} to {qubo_max:.4f} (Midpoint: {qubo_mid:.4f})")

    print("\n--- Sequences in True Top 10% but missed by QUBO (False Negatives) ---")
    print("These are good sequences that QUBO failed to include. Where do they rank in True?")
    print(f"Total missed: {len(missed_by_qubo)}")
    print(f"{'Sequence':<35} | {'True (HUBO)':<12} | {'QUBO':<12} | {'Error (Q-T)':<12} | {'True Position'}")
    print("-" * 96)
    
    missed_by_qubo_sorted = sorted([seq_lookup[seq] for seq in missed_by_qubo], key=lambda x: x['true'])
    for info in missed_by_qubo_sorted:
        error = info['qubo'] - info['true']
        # For false negatives, we care about where they sat in the True distribution
        position = "Closer to Lower (Highly Optimal)" if info['true'] < true_mid else "Closer to Upper (Borderline)"
        print(f"{info['seq']:<35} | {info['true']:<12.4f} | {info['qubo']:<12.4f} | {error:<12.4f} | {position}")


    print("\n--- Sequences in QUBO Top 10% but NOT in True Top 10% (False Positives) ---")
    print("These are bad sequences that QUBO incorrectly included. Where do they rank in QUBO?")
    print(f"Total false positives: {len(false_positives_qubo)}")
    print(f"{'Sequence':<35} | {'True (HUBO)':<12} | {'QUBO':<12} | {'Error (Q-T)':<12} | {'QUBO Position'}")
    print("-" * 96)

    fp_qubo_sorted = sorted([seq_lookup[seq] for seq in false_positives_qubo], key=lambda x: x['qubo'])
    for info in fp_qubo_sorted:
        error = info['qubo'] - info['true']
        # For false positives, we care about where they sat in the QUBO distribution
        position = "Closer to Lower" if info['qubo'] < qubo_mid else "Closer to Upper (Borderline)"
        print(f"{info['seq']:<35} | {info['true']:<12.4f} | {info['qubo']:<12.4f} | {error:<12.4f} | {position}")
    
    print("="*60 + "\n")

    # =================================================================
    # PLOTTING
    # =================================================================
    save_dir = "correlation_plots"
    os.makedirs(save_dir, exist_ok=True)

    # PLOT 1: The 10% Zoom-in
    plt.figure(figsize=(10, 8))
    plt.scatter(b10_true, b10_qubo, color='#ff7f0e', alpha=0.7, s=50, edgecolor='black')
    
    # Trendline for visual aid
    z = np.polyfit(b10_true, b10_qubo, 1)
    p = np.poly1d(z)
    plt.plot(b10_true, p(b10_true), "r--", alpha=0.8, label=f"Trendline")

    plt.title(f"Ground State Focus: Lowest 10% QUBO Energies ({math_method.upper()})\nSpearman Correlation: {b10_corr:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Turner Energy (True) [kcal/mol]", fontsize=12)
    plt.ylabel("QUBO Approximated Energy [kcal/mol]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"Bottom10_{target_structure}_{math_method}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 2: Tail Overlap Histogram
    plt.figure(figsize=(10, 6))
    
    # We plot the True energies of the sequences the QUBO *chose* as its bottom 10%
    # against the True energies of the actual biological bottom 10%.
    actual_best_energies = [x['true'] for x in true_sorted[:ten_percent_idx]]
    qubo_chosen_energies = [x['true'] for x in bottom_10_chunk]
    
    plt.hist(actual_best_energies, bins=20, alpha=0.5, color='green', label=f'True Biological Top 10% (Ideal)')
    plt.hist(qubo_chosen_energies, bins=20, alpha=0.5, color='purple', label=f'QUBO Selected Top 10% (Achieved)')
    
    plt.title(f"Tail Overlap Validation ({math_method.upper()})\nOverlap Match: {overlap_pct:.1f}%", fontsize=14, fontweight='bold')
    plt.xlabel("True Free Energy (kcal/mol)", fontsize=12)
    plt.ylabel("Number of Sequences", fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"TailOverlap_{target_structure}_{math_method}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots generated saved in '{save_dir}' folder!")


if __name__ == "__main__":
    target_list = ["..............(((((.....)))))", "(((...)))","...((((((.........))))))."]
    # true structure :CUCUUUAACAUUAAGCCCUGAAGAAGGGC
    #target_1 = "(((...)))" # true structure :GCCGUCGGC
    # target_1 = "...((((((.........))))))."
    # methods : ols , l1 , minimax , rank
    # new methods : huber, wls,margin_rank
    for i in target_list:
        run_statistical_sampling(i,num_samples=5000,math_method="ols")
        print("\n")