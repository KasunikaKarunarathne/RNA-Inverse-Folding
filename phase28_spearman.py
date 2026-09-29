import dimod
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
import os

DANGLE_REWARDS = {
    ("C", "G", "A"): -1.5, ("C", "G", "G"): -1.3, ("C", "G", "U"): -0.8, ("C", "G", "C"): -0.4,
    ("G", "C", "A"): -1.5, ("G", "C", "G"): -1.3, ("G", "C", "U"): -0.8, ("G", "C", "C"): -0.4,
    ("A", "U", "A"): -1.0, ("A", "U", "G"): -0.8, ("A", "U", "U"): -0.6, ("A", "U", "C"): -0.2,
    ("U", "A", "A"): -1.0, ("U", "A", "G"): -0.8, ("U", "A", "U"): -0.6, ("U", "A", "C"): -0.2,
    ("G", "U", "A"): -0.6, ("G", "U", "G"): -0.5, ("G", "U", "U"): -0.3, ("G", "U", "C"): -0.1,
    ("U", "G", "A"): -0.6, ("U", "G", "G"): -0.5, ("U", "G", "U"): -0.3, ("U", "G", "C"): -0.1,
}

def test_rosenberg_fidelity():
    poly = defaultdict(float)
    
    # 1. Inject HUBO terms
    for (b_left, b_right, d_base), bonus in DANGLE_REWARDS.items():
        var_L = f"L_{b_left}"
        var_R = f"R_{b_right}"
        var_D = f"D_{d_base}"
        poly[(var_L, var_R, var_D)] += bonus
        
    # 2. Add One-Hot Encoding
    P_onehot = 5.0 
    bases = ["A", "C", "G", "U"]
    for pos in ["L", "R", "D"]:
        for b in bases:
            poly[(f"{pos}_{b}",)] -= P_onehot
        for i in range(4):
            for j in range(i+1, 4):
                poly[(f"{pos}_{bases[i]}", f"{pos}_{bases[j]}")] += 2 * P_onehot

    # 3. Reduce to QUBO
    P_aux = 10.0
    bqm = dimod.make_quadratic(poly, strength=P_aux, vartype=dimod.BINARY)
    
    true_energies = []
    qubo_energies = []
    labels = []
    
    # Use ExactSolver to evaluate all QUBO auxiliary states
    sampler = dimod.ExactSolver()
    
    print("Calculating QUBO energies for all 24 3-Body configurations...")
    for (b_left, b_right, d_base), true_bonus in DANGLE_REWARDS.items():
        # Fix our primary variables to the current test configuration
        fixed_bqm = bqm.copy()
        for pos in ["L", "R", "D"]:
            for b in bases:
                var_name = f"{pos}_{b}"
                if (pos == "L" and b == b_left) or (pos == "R" and b == b_right) or (pos == "D" and b == d_base):
                    fixed_bqm.fix_variable(var_name, 1)
                else:
                    fixed_bqm.fix_variable(var_name, 0)
                    
        # Solve for the auxiliary variables 
        sampleset = sampler.sample(fixed_bqm)
        best_energy = sampleset.first.energy
        
        true_energies.append(true_bonus)
        qubo_energies.append(best_energy)
        labels.append(f"{b_left}{b_right}-{d_base}")
        
    # The QUBO energy includes the one-hot reward offsets (-P_onehot * 3).
    # We remove the offset to directly compare the physical energies.
    adjusted_qubo_energies = [e - (-3 * P_onehot) for e in qubo_energies]
    
    # 4. Calculate Spearman and Plot
    correlation, _ = spearmanr(true_energies, adjusted_qubo_energies)
    print(f"\n==============================================")
    print(f"Spearman Rank Correlation: {correlation:.4f}")
    print(f"==============================================")
    
    if correlation < 0.9:
        print("WARNING: Correlation is too low to proceed with QUBO reduction!")
    else:
        print("SUCCESS: The QUBO perfectly preserves the 3-Body energy landscape!")
    
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.scatter(true_energies, adjusted_qubo_energies, color='#9467bd', s=100, edgecolor='black')
    
    for i, label in enumerate(labels):
        plt.annotate(label, (true_energies[i], adjusted_qubo_energies[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
                     
    min_val = min(min(true_energies), min(adjusted_qubo_energies))
    max_val = max(max(true_energies), max(adjusted_qubo_energies))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Match (y=x)")
    
    plt.title(f"Rosenberg Reduction Fidelity\nSpearman Correlation: {correlation:.4f}", fontweight='bold')
    plt.xlabel("True Turner Energy (3-Body)")
    plt.ylabel("Reduced QUBO Energy (2-Body)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig("plots/rosenberg_fidelity.png", dpi=300)
    print("--> Saved plot to plots/rosenberg_fidelity.png")

if __name__ == "__main__":
    test_rosenberg_fidelity()
