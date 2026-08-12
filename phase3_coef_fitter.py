import os
import numpy as np
from itertools import product
from phase2_turner_energy import get_turner_energy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.optimize import minimize, linprog

# 1. Base pair encoding 
# mapping the 6 allowed types to their [p0,p1,p2] binary signatures
PAIR_ENCODING = {
    'AU': [1, 1, 1],
    'UA': [1, 1, 0],
    'CG': [1, 0, 1],
    'GC': [0, 1, 0],
    'GU': [0, 1, 1],
    'UG': [1, 0, 0]
}

ALLOWED_PAIRS = ['AU','UA','CG','GC','GU','UG']

def build_qvector(pair1,pair2):
    """
    Combines the 3 bits of pair1 and 3 bits of pair2 into the 6-bit q vector.
    """
    return PAIR_ENCODING[pair1] + PAIR_ENCODING[pair2]

def calculate_qubo_coeffs(method="ols"):
    """
    Solves for the 22 QUBO coefficients using different objective functions.
    Available methods: 'ols', 'l1', 'minimax', 'rank'
    """
    num_samples = 36
    num_coeffs = 22

    Phi = np.zeros((num_samples,num_coeffs))
    E_true = np.zeros(num_samples)

    # generate all 36 stack combinations
    combinations = list(product(ALLOWED_PAIRS,ALLOWED_PAIRS))

    for row_i , (pair1,pair2) in enumerate(combinations):
        # Get the targert energy
        E_true[row_i] = get_turner_energy(pair1,pair2)

        # get the 6 bit vector for this combinantion
        q = build_qvector(pair1,pair2)

        # populate the design mat phi for this row
        col_i =0
        Phi[row_i,col_i] = 1 # c0 bias term always 1
        col_i +=1

        # c_a : linear terms
        for a in range(6):
            Phi[row_i,col_i] = q[a]
            col_i +=1
        
        # c_ab : quadratic cross terms
        for a in range(6):
            for b in range(a+1,6):
                Phi[row_i,col_i] = q[a] *q[b]
                col_i +=1
        
    # --- Solve Regression based on selected method ---
    if method == "ols":
        c_coeffs , residuals, _,_ = np.linalg.lstsq(Phi,E_true,rcond=None)
        if len(residuals) >0:
            mse = residuals[0]/num_samples
            print(f"[{method.upper()}] Mean Squared Error of approximation: {mse:.4f}")
            
    elif method == "l1":
        def l1_loss(c):
            return np.sum(np.abs(np.dot(Phi, c) - E_true))
        c0, _, _, _ = np.linalg.lstsq(Phi, E_true, rcond=None)
        res = minimize(l1_loss, c0, method='BFGS')
        c_coeffs = res.x
        
    elif method == "minimax":
        c_bounds = [(None, None)] * num_coeffs
        t_bound = [(0, None)]
        bounds = c_bounds + t_bound
        c_obj = np.zeros(num_coeffs + 1)
        c_obj[-1] = 1.0 
        
        A_ub = np.vstack((
            np.hstack((Phi, -np.ones((num_samples, 1)))),
            np.hstack((-Phi, -np.ones((num_samples, 1))))
        ))
        b_ub = np.concatenate((E_true, -E_true))
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        c_coeffs = res.x[:-1]
        
    elif method == "rank":
        def rank_loss(c):
            E_pred = np.dot(Phi, c)
            loss = 0
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    diff_true = E_true[i] - E_true[j]
                    diff_pred = E_pred[i] - E_pred[j]
                    if diff_true * diff_pred < 0:
                        loss += np.abs(diff_pred)
            return loss + 0.01 * np.sum(c**2)
            
        c0, _, _, _ = np.linalg.lstsq(Phi, E_true, rcond=None)
        res = minimize(rank_loss, c0, method='Nelder-Mead')
        c_coeffs = res.x

    elif method == "quantum_anneal":
        import neal
        # 1. Setup Binary Expansion Parameters (6 bits over [-1.5, 1.5] matches actual physical domain with 0.047 step resolution)
        bits_per_coeff = 6 
        c_min = -1.5
        c_max = 1.5
        step_size = (c_max - c_min) / ((2**bits_per_coeff) - 1)
        
        sampler = neal.SimulatedAnnealingSampler()

        # We formulate 3 distinct Hamiltonian weighting regimes and select the best-performing ground state
        # (This avoids spin-glass frustration from dense pairwise terms while explicitly optimizing Spearman rank)
        weight_regimes = {
            "Uniform MSE": np.ones(num_samples),
            "Boltzmann Stability": np.exp(-(E_true - np.max(E_true)) / 1.5), # Prioritize stable native structures
            "Dense-Core Tie-Breaker": 1.0 + 3.0 * np.exp(-((E_true + 1.8) / 0.7)**2) # Prioritize crowded energy cluster to prevent rank flips
        }

        best_c_coeffs = None
        best_spearman = -1.0

        print("Evaluating Quantum Annealing weighting regimes...")
        for regime_name, weights in weight_regimes.items():
            Q_meta = {}
            # Build clean sample-wise QUBO (avoids spin-glass landscape traps)
            for s in range(num_samples):
                w = weights[s]
                P_s = np.sum(Phi[s] * c_min) - E_true[s]
                for i in range(num_coeffs):
                    if Phi[s, i] == 0: continue
                    for k_i in range(bits_per_coeff):
                        var_i = f"c_{i}_bit_{k_i}"
                        A_i = Phi[s, i] * step_size * (2**k_i)
                        if (var_i, var_i) not in Q_meta: Q_meta[(var_i, var_i)] = 0.0
                        Q_meta[(var_i, var_i)] += w * ((A_i**2) + (2 * P_s * A_i))

                        for j in range(i, num_coeffs):
                            if Phi[s, j] == 0: continue
                            start_k_j = k_i + 1 if i == j else 0
                            for k_j in range(start_k_j, bits_per_coeff):
                                var_j = f"c_{j}_bit_{k_j}"
                                A_j = Phi[s, j] * step_size * (2**k_j)
                                pair = tuple(sorted((var_i, var_j)))
                                if pair not in Q_meta: Q_meta[pair] = 0.0
                                Q_meta[pair] += w * (2 * A_i * A_j)

            sampleset = sampler.sample_qubo(Q_meta, num_reads=3000, num_sweeps=1000)
            best_sample = sampleset.first.sample
            
            cand_coeffs = np.zeros(num_coeffs)
            for i in range(num_coeffs):
                coeff_val = c_min 
                for k in range(bits_per_coeff):
                    if best_sample.get(f"c_{i}_bit_{k}", 0) == 1:
                        coeff_val += step_size * (2 ** k)
                cand_coeffs[i] = coeff_val
                
            cand_pred = np.dot(Phi, cand_coeffs)
            cand_spearman, _ = spearmanr(E_true, cand_pred)
            print(f"  [Regime: {regime_name:<22}] Spearman Rank: {cand_spearman:.4f}")
            
            if cand_spearman > best_spearman:
                best_spearman = cand_spearman
                best_c_coeffs = cand_coeffs

        print(f"\nSelected Optimal Quantum Annealing Regime with Spearman Score: {best_spearman:.4f}")
        c_coeffs = best_c_coeffs
        
    else:
        raise ValueError(f"Unknown method '{method}'")

    return c_coeffs


def evaluate_qubo_fit(c_coeffs, method="ols"):
    """
    Takes the calculated coefficients and compares the QUBO predicted
    energies against the true Turner 2004 dictionary energies.
    """
    num_samples = 36
    num_coeffs = 22

    # We quickly rebuild the matrices here to keep the functions perfectly decoupled
    Phi = np.zeros((num_samples,num_coeffs))
    E_true = np.zeros(num_samples)
    combinations = list(product(ALLOWED_PAIRS,ALLOWED_PAIRS))

    for row_i , (pair1,pair2) in enumerate(combinations):
        E_true[row_i] = get_turner_energy(pair1,pair2)
        q = build_qvector(pair1,pair2)

        col_i = 0
        Phi[row_i,col_i] = 1 
        col_i += 1
        for a in range(6):
            Phi[row_i,col_i] = q[a]
            col_i += 1
        for a in range(6):
            for b in range(a+1,6):
                Phi[row_i,col_i] = q[a] * q[b]
                col_i += 1

    print("="*60)
    print(f"QUBO APPROXIMATION ({method.upper()})")
    print("="*60)

    # Calculate predicted energies using the passed coefficients
    E_pred = np.dot(Phi, c_coeffs)
    errors = E_pred - E_true

    # Print the detailed comparison table
    print(f"{'Stack':<10} | {'Turner (True)':<15} | {'QUBO (Predicted)':<18} | {'Error'}")
    print("-" * 60)
    
    for i, (pair1, pair2) in enumerate(combinations):
        stack_name = f"{pair1}/{pair2}"
        print(f"{stack_name:<10} | {E_true[i]:>13.3f}   | {E_pred[i]:>16.3f}   | {errors[i]:>6.3f}")
        
    print("-" * 60)
    
    # Calculate and print overall metrics
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(np.abs(errors))
    
    print(f"\nOverall RMSE      : {rmse:.4f} kcal/mol")
    print(f"Max Absolute Error: {max_err:.4f} kcal/mol")
    print("="*60)

    # --- STAGE 1: MONOTONICITY PLOT (36 STACKS) ---
    # Create folder for plots
    save_dir = "plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate Spearman Rank Correlation (1.0 is perfect ordering)
    correlation, _ = spearmanr(E_true, E_pred)
    print(f"Spearman Rank Correlation: {correlation:.4f}\n")

    plt.figure(figsize=(12, 9)) # Made wider to fit text labels
    plt.scatter(E_true, E_pred, color='#9467bd', s=80, alpha=0.8, edgecolor='black')

    # Add text labels (pair names) to each dot
    for i, (pair1, pair2) in enumerate(combinations):
        label = f"{pair1}/{pair2}"
        plt.annotate(
            label, 
            (E_true[i], E_pred[i]),
            textcoords="offset points", 
            xytext=(5, 5), 
            ha='left',
            fontsize=8,
            alpha=0.75
        )

    # Draw the perfect y=x diagonal line for reference
    min_val = min(min(E_true), min(E_pred))
    max_val = max(max(E_true), max(E_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label="Perfect Match (y=x)")

    plt.title(f"Stage 1: Turner Stacks Monotonicity ({method.upper()})\nSpearman Rank Correlation: {correlation:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("True Turner Energy (HUBO) [kcal/mol]", fontsize=12)
    plt.ylabel("QUBO Approximated Energy [kcal/mol]", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # Save with dynamic filename
    filename = f"TurnerVSQUBO_{method}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close() # Closes the figure so it doesn't overlap the next one in the loop
    
    print(f"Stage 1 Monotonicity plot saved as '{filepath}'\n")


if __name__ == "__main__":
    # Loop through all 4 methods to test and plot them back-to-back
    methods_to_test = ["ols", "l1", "minimax", "rank","quantum_anneal"]
    
    for method in methods_to_test:
        print(f"\n--- Testing Method: {method.upper()} ---")
        
        # 1. calculate the weights
        coefficients = calculate_qubo_coeffs(method=method)
        
        print(f"Regression Complete for {method.upper()}.")
        print(f"Calculated 22 coefficients: {np.round(coefficients, 3)}\n")
        
        # 2. Pass those weights into the diagnostic function to see the error margins and plot
        evaluate_qubo_fit(coefficients, method=method)
