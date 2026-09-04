import RNA
import csv
import os
from scipy.stats import spearmanr
from phase3_coef_fitter import calculate_qubo_coeffs
from phase17_extended import build_extended_qubo, decode_extended_sample, generate_sequence
from phase1_rules import extract_stems
from phase24_advanced_qaoa import SimpleQAOAExecutor
from qiskit_optimization import QuadraticProgram

# We need a dynamic builder because the Extended QUBO uses string variables like "x_5_A"
def build_extended_qp_from_dict(Q_dict, name="Extended_RNA_QUBO"):
    qp = QuadraticProgram(name=name)
    added_vars = set()
    
    linear, quadratic = {}, {}
    for (v1, v2), coef in Q_dict.items():
        if v1 not in added_vars:
            qp.binary_var(v1)
            added_vars.add(v1)
        if v2 not in added_vars:
            qp.binary_var(v2)
            added_vars.add(v2)
            
        if v1 == v2:
            linear[v1] = coef
        else:
            quadratic[(v1, v2)] = coef
            
    qp.minimize(linear=linear, quadratic=quadratic)
    return qp

def evaluate_vienna_metrics(seq, target_structure):
    fc = RNA.fold_compound(seq)
    struct_mfe, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    struct_pf, free_energy_ensemble = fc.pf()
    ensemble_defect = fc.ensemble_defect(target_structure)
    diversity = fc.mean_bp_distance()
    return mfe, ensemble_defect, diversity

def run_extended_benchmark():
    csv_path = r"Structures\fmqa_paper_structures.csv"
    targets = []
    if os.path.exists(csv_path):
        with open(csv_path,'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(row['Structure'])
    else:
        targets = ["(((...)))"]

    print("Loading OLS coefficients...")
    c_coeffs = calculate_qubo_coeffs(method="ols")
    
    for target_structure in targets:
    
        print(f"\n==============================================")
        print(f"Testing Target: {target_structure}")
        print(f"==============================================")
        
        stems = extract_stems(target_structure)
        if not stems:
            print("No stems found, skipping...")
            continue
    
        print("\n--- 1. Constructing EXTENDED QUBO ---")
        
        # Build the FULL quantum matrix (stems + one-hot encoded loops)
        Q_ext, offset_ext, loop_indices, fixed_loops = build_extended_qubo(target_structure, stems, c_coeffs)
        qp_ext = build_extended_qp_from_dict(Q_ext, name="Extended_QUBO")
        print(f"Qubits required: {qp_ext.get_num_vars()}")

        # You increased the limit to 25 Qubits! 
        if qp_ext.get_num_vars() > 25:
            print(f"WARNING: {qp_ext.get_num_vars()} Qubits is too large for local Aer Simulator. Skipping to next structure...")
            continue
        
        print("\n--- 2. Running QAOA ---")
        # USING YOUR NEW FIXES: reps=1, maxiter=500
        executor = SimpleQAOAExecutor(reps=1, maxiter=500)
        
        # USING YOUR NEW FIXES: top_n=50
        top_samples = executor.run_qaoa_with_samples(qp_ext, top_n=50)
        
        print(f"\n--- 3. Decoding & Evaluating Top {len(top_samples)} Sequences ---")
        
        success_count = 0
        total_defect = 0.0
        total_diversity = 0.0
        
        q_energies = []
        v_energies = []
    
        for rank, (x_n, e_n, prob) in enumerate(top_samples):
            # Extract variables using the EXACT string names from the Extended QUBO
            var_dict = {var.name: int(val) for var, val in zip(qp_ext.variables, x_n)}
            
            # Use the Extended Decoder to read stems AND loops simultaneously from QAOA!
            decoded = decode_extended_sample(var_dict, stems, loop_indices, fixed_loops)
            if not decoded:
                continue
                
            pairs_list, loop_assignment = decoded
            final_seq = generate_sequence(target_structure, stems, pairs_list, loop_assignment)
            
            folded_struct, mfe = RNA.fold(final_seq)
            is_match = (folded_struct == target_structure)
            
            q_energy = e_n + offset_ext
            
            print(f"\nRank {rank+1} (Prob: {prob:.4f}, Q-Energy: {q_energy:.2f}, Vienna MFE: {mfe:.2f})")
            print(f"Seq:   {final_seq}")
            print(f"Match: {'YES' if is_match else 'NO'}")
            
            q_energies.append(q_energy)
            v_energies.append(mfe)
            
            if is_match:
                success_count += 1
                _, defect, diversity = evaluate_vienna_metrics(final_seq, target_structure)
                total_defect += defect
                total_diversity += diversity
                print(f"Metrics -> Defect: {defect:.2f}, Diversity: {diversity:.2f}")

        print("\n==============================================")
        print("FINAL QAOA SUMMARY")
        print("==============================================")
        print(f"Success Rate:    {success_count}/{len(top_samples)} ({(success_count/len(top_samples))*100:.1f}%)")
        if success_count > 0:
            print(f"Avg Defect:      {total_defect/success_count:.2f}")
            print(f"Avg Diversity:   {total_diversity/success_count:.2f}")
            
        if len(q_energies) > 1:
            corr, _ = spearmanr(v_energies, q_energies)
            print(f"Spearman Corr (Q-Energy vs MFE): {corr:.4f}")
        else:
            print("Spearman Corr: Not enough valid samples.")

if __name__ == "__main__":
    run_extended_benchmark()
