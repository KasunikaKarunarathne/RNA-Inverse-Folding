import os
import numpy as np
from itertools import product
from phase2_turner_energy import get_turner_energy
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# The 6 valid RNA pairs
ALLOWED_PAIRS = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG']
BASES = ['A', 'C', 'G', 'U']
NUM_BASES = 4

def get_target_penalty(b1, b2, b3, b4):
    """
    Calculates the exact thermodynamic penalty for a 4-base loop state.
    b1, b2 are the top/bottom left bases (i, i+1)
    b3, b4 are the bottom/top right bases (j-1, j)
    So the pairs are: (b1, b4) and (b2, b3)
    """
    pair1 = b1 + b4
    pair2 = b2 + b3
    
    # If it forms a valid stack
    if pair1 in ALLOWED_PAIRS and pair2 in ALLOWED_PAIRS:
        energy = get_turner_energy(pair1, pair2)
        # We only penalize stabilizing (negative) energies.
        # e.g., if energy is -3.42, the penalty is +3.42 to cancel it.
        return max(0.0, -energy)
    
    # If it is an invalid/non-canonical stack, it doesn't form, so no thermodynamic penalty
    return 0.0

def build_feature_vector(b1, b2, b3, b4):
    """
    Constructs the 113-term One-Hot feature vector for this 4-base state.
    """
    bases = [b1, b2, b3, b4]
    
    # 1. Bias term
    features = [1.0]
    
    # 2. Linear terms (16 terms: 4 positions * 4 bases)
    linear_vars = []
    for pos_base in bases:
        for b in BASES:
            val = 1.0 if pos_base == b else 0.0
            linear_vars.append(val)
            features.append(val)
            
    # 3. Quadratic Cross-Terms (96 terms)
    # We pair up the 4 positions (6 possible pairs)
    # For each pair of positions, we cross-multiply their 4 variables (16 cross terms)
    # total 6 * 16 = 96 quadratic terms
    for p_idx_A in range(4):
        for p_idx_B in range(p_idx_A + 1, 4):
            # Extract the 4 variables for position A
            vars_A = linear_vars[p_idx_A * 4 : (p_idx_A + 1) * 4]
            # Extract the 4 variables for position B
            vars_B = linear_vars[p_idx_B * 4 : (p_idx_B + 1) * 4]
            
            # Cross multiply
            for vA in vars_A:
                for vB in vars_B:
                    features.append(vA * vB)
                    
    return np.array(features)

def calculate_loop_qubo_coeffs():
    """
    Fits the 113-term QUBO surrogate to the 256 possible 4-base states using OLS.
    """
    num_states = 256
    num_coeffs = 113
    
    Phi = np.zeros((num_states, num_coeffs))
    E_true = np.zeros(num_states)
    
    combinations = list(product(BASES, BASES, BASES, BASES))
    
    for row_i, (b1, b2, b3, b4) in enumerate(combinations):
        E_true[row_i] = get_target_penalty(b1, b2, b3, b4)
        Phi[row_i, :] = build_feature_vector(b1, b2, b3, b4)
        
    # Solve OLS
    c_coeffs, residuals, _, _ = np.linalg.lstsq(Phi, E_true, rcond=None)
    
    if len(residuals) > 0:
        mse = residuals[0] / num_states
        print(f"[OLS] Mean Squared Error of approximation: {mse:.4f}")
        
    return c_coeffs, Phi, E_true, combinations

def evaluate_fit():
    c_coeffs, Phi, E_true, combinations = calculate_loop_qubo_coeffs()
    E_pred = np.dot(Phi, c_coeffs)
    errors = E_pred - E_true
    
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(np.abs(errors))
    
    print(f"Overall RMSE      : {rmse:.4f} kcal/mol")
    print(f"Max Absolute Error: {max_err:.4f} kcal/mol")
    
    # Check how many invalid states got accidental negative penalties (rewards)
    invalid_rewards = 0
    for i, (b1, b2, b3, b4) in enumerate(combinations):
        if E_true[i] == 0.0 and E_pred[i] < -0.01:
            invalid_rewards += 1
            
    print(f"Invalid states receiving negative penalty (< -0.01): {invalid_rewards} out of 220")
    
    # Save the coefficients for Phase 22 to use
    np.save("loop_penalty_coeffs.npy", c_coeffs)
    print("Coefficients saved to 'loop_penalty_coeffs.npy'")

if __name__ == "__main__":
    evaluate_fit()
