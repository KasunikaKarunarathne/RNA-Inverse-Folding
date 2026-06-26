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

    # 4. Statistical Analysis
    correlation ,_ = spearmanr(true_energies,qubo_energies)
    print(f"Spearman Rank correlation: {correlation:.4f}")

    # Print the top 10 most stable sequnces 
    combined_results = list(zip(sequences, true_energies, qubo_energies))
    # Sort the list based on true_energies (index 1 in the zip) from lowest to highest
    combined_results.sort(key=lambda x: x[1])
    
    print("\n" + "="*60)
    print(f"TOP 10 MOST STABLE SEQUENCES FOR {target_structure}")
    print("="*60)
    print(f"{'Rank':<5} | {'Sequence':<15} | {'True (kcal)':<12} | {'QUBO Pred':<12} | {'Error'}")
    print("-" * 60)
    
    for i, (seq, t_eng, q_eng) in enumerate(combined_results[:10]):
        error = abs(t_eng - q_eng)
        print(f"{i+1:<5} | {seq:<15} | {t_eng:>12.3f} | {q_eng:>12.3f} | {error:>6.3f}")
    print("="*60 + "\n")
    # ---------------------------------------------------------

    # 5. Plotting
    save_dir = "correlation_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(true_energies,qubo_energies,color = '#2ca02c', 
                alpha =0.5,s=30, edgecolor='black', linewidth=0.5)

    # draw the perfect y = x diagonal line for ref 
    min_val = min(min(true_energies), min(qubo_energies))
    max_val = max(max(true_energies), max(qubo_energies))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label="Perfect Match (y=x)")
    
    plt.title(f"Stage 2: Whole Structure Correlation ({math_method.upper()})\nTarget: {target_structure}\nSpearman Correlation: {correlation:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("ViennaRNA Energy [kcal/mol]", fontsize=12)
    plt.ylabel("QUBO Approximated Energy [kcal/mol]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    filename = f"Correlation_{target_structure}_{math_method}_{num_samples}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully to: {filepath}")
    print("="*60)


if __name__ == "__main__":
    target_1 = "(((...)))" # true structure :GCCGUCGGC
    # target_1 = "...(((...)))" # true structure :GCCGUCGGC
    # target_1 = "...(((...)))" # true structure :GCCGUCGGC
    # methods : ols , l1 , minimax , rank
    # new methods : huber, wls,margin_rank
    run_statistical_sampling(target_1,num_samples=100,math_method="margin_rank")