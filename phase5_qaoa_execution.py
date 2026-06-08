# phase4_qaoa_execution.py
import os
import json
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
# Sometimes SciPy warnings subclass differently, this blanket catches them:
warnings.filterwarnings("ignore")

# Import our previous phases
from phase1_rules import extract_stems, PAIR_ENCODING
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo
from phase2_turner_energy import calculate_true_turner_energy

REVERSE_ENCODING = {tuple(bits):pair for pair,bits in PAIR_ENCODING.items()}

def run_full_pipeline(target_structure,reps=2, maxiter=100):
    print(f"--- Starting Quantum Pipeline for Target: {target_structure} ---")
    # 1. Preperation
    stems = extract_stems(target_structure)
    seq_length = len(target_structure)

    # calculate 22 physical weights (phase 3)
    c_coeffs = calculate_qubo_coeffs()

    # build the matrix and get the offset
    Q_dict , constant_offset = build_approx_qubo(stems,c_coeffs)

    # 2. Build qiskit model
    qp = QuadraticProgram("Turner_RNA_Folding")

    # we only add binary variables for the pair that actually exist in the stems 
    for stem in stems:
        for pair in stem:
            qp.binary_var(f"p_{pair[0]}_{pair[1]}_0")
            qp.binary_var(f"p_{pair[0]}_{pair[1]}_1")
            qp.binary_var(f"p_{pair[0]}_{pair[1]}_2")
    linear , quadratic ={} ,{}
    for (var1, var2), coef in Q_dict.items():
        if var1 == var2:
            linear[var1] = coef
        else:
            quadratic[(var1,var2)] =coef
    qp.minimize(linear = linear, quadratic= quadratic)

    # 3. Execute
    counts, values =[],[]
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        # Add the offset back to the raw Qiskit score to get real kcal/mol
        values.append(mean + constant_offset)
        print(f"QAOA Iteration {eval_count}: Energy = {mean + constant_offset:.2f} kcal/mol")

    optimizer = COBYLA(maxiter=maxiter)
    sampler = Sampler()
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, callback=store_intermediate_result)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    print("Executing QAOA Circuit... (Optimizing quantum angles)")
    result = qaoa_optimizer.solve(qp)
    
    # --- STEP 4: DECODE THE QUANTUM BITSTRING ---
    # We start with a blank sequence. We fill unpaired loop bases with 'A' 
    # to prevent them from accidentally forming structures.
    final_sequence = ['A'] * seq_length 
    
    for stem in stems:
        for pair in stem:
            # Extract the 3 bits the quantum computer chose for this pair
            b0 = result.variables_dict[f"p_{pair[0]}_{pair[1]}_0"]
            b1 = result.variables_dict[f"p_{pair[0]}_{pair[1]}_1"]
            b2 = result.variables_dict[f"p_{pair[0]}_{pair[1]}_2"]
            
            bit_tuple = (int(b0), int(b1), int(b2))
            
            # If the optimizer output an invalid state (like 0,0,0), we flag an error.
            # The +100 penalty should prevent this from ever happening.
            if bit_tuple not in REVERSE_ENCODING:
                print(f"WARNING: Optimizer chose invalid state {bit_tuple} at pair {pair}.")
                assigned_pair = "AA" 
            else:
                assigned_pair = REVERSE_ENCODING[bit_tuple] # e.g., 'GC'
                
            # Place the letters into the sequence array
            final_sequence[pair[0]] = assigned_pair[0]
            final_sequence[pair[1]] = assigned_pair[1]
            
    rna_string = "".join(final_sequence)
    true_energy = calculate_true_turner_energy(stems, rna_string)

    # --- STEP 5: SAVE RESULTS TO FOLDER ---
    save_results(target_structure, rna_string, true_energy, counts, values, maxiter, reps)

def save_results(target, sequence, energy, counts, values, maxiter, reps):
    """Creates the /results directory and saves the plot and data."""
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # UPDATED: Create a unique filename base
    file_base = f"{target}_iters{maxiter}_reps{reps}"
        
    # 1. Save the Plot
    plt.figure(figsize=(10, 6))
    plt.plot(counts, values, color='#1f77b4', linewidth=2.5, label="QAOA Energy")
    plt.axhline(y=energy, color='red', linestyle='--', alpha=0.7, label="True Ground State")
    plt.title(f"QAOA Convergence for RNA Target:\n{target}", fontsize=14, fontweight='bold')
    plt.xlabel("Optimizer Iterations", fontsize=12)
    plt.ylabel("System Energy (kcal/mol)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()
    
    # Apply dynamic filename
    plot_path = os.path.join("results", f"{file_base}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Save the Data Report
    report = {
        "Target_Structure": target,
        "Optimal_Sequence": sequence,
        "Final_Energy_kcal_mol": round(energy, 2),
        "Sequence_Length": len(sequence),
        "QAOA_Reps": reps,
        "Total_Iterations": len(counts)
    }
    
    # Apply dynamic filename
    json_path = os.path.join("results", f"{file_base}.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n" + "="*40)
    print("SUCCESS: Pipeline Complete.")
    print(f"Generated Sequence: {sequence}")
    print(f"Final Energy:       {energy:.2f} kcal/mol")
    print(f"Files saved in the 'results/' directory as '{file_base}'.")
    print("="*40)

# --- RUN THE PIPELINE ---
if __name__ == "__main__":
    test_target = "((....))"
    # run_full_pipeline(test_target, reps=2, maxiter=100)
    stems = extract_stems(test_target)
    run_full_pipeline(test_target, reps=4, maxiter=100)