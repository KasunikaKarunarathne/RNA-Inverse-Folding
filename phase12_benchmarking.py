import time
import dimod
import neal
from docplex.mp.model import Model
from phase1_rules import extract_stems
from phase3_coef_fitter import calculate_qubo_coeffs
from phase4_qubo_builder import build_approx_qubo

def decode_bits_to_pairs(sample_dict, stems):
    """Converts binary variable values back into sequence base pairs."""
    bit_to_pair = {
        (1,1,1):'AU', (1,1,0):'UA',
        (1,0,1):'CG', (0,1,0):'GC',
        (0,1,1):'GU', (1,0,0):'UG'
    }
    pair_assignment = []
    for stem in stems:
        for (left, right) in stem:
            b0 = sample_dict.get(f"p_{left}_{right}_0", 0)
            b1 = sample_dict.get(f"p_{left}_{right}_1", 0)
            b2 = sample_dict.get(f"p_{left}_{right}_2", 0)
            bit_tuple = (b0, b1, b2)
            if bit_tuple not in bit_to_pair:
                return None
            pair_assignment.append(bit_to_pair[bit_tuple])
    return tuple(pair_assignment)

def solve_qubo_with_cplex(Q_dict):
    """
    Benchmarks your QUBO dictionary using IBM CPLEX Classical Optimization.
    Returns the optimal binary assignment dictionary and true minimum energy.
    """
    # 1. Initialize the classical CPLEX model
    mdl = Model(name="RNA_QUBO_Classical_Benchmark")
    
    # 2. Extract unique variable names (e.g., 'p_0_28_0', etc.) from Q_dict
    var_names = sorted(list(set([u for pair in Q_dict.keys() for u in pair])))
    x = mdl.binary_var_dict(var_names, name="x")
    
    # 3. Construct the polynomial objective directly from Q_dict
    objective_terms = []
    for (u, v), weight in Q_dict.items():
        if u == v:
            objective_terms.append(weight * x[u])         # Linear term: w * x_i
        else:
            objective_terms.append(weight * x[u] * x[v])  # Quadratic cross-term: w * x_i * x_j
            
    mdl.minimize(mdl.sum(objective_terms))
    
    # 4. Run CPLEX Branch-and-Bound solver
    solution = mdl.solve()
    
    # 5. Decode the exact ground state
    if solution:
        best_energy = solution.objective_value
        best_sample = {var: int(solution.get_value(x[var])) for var in var_names}
        return best_sample, best_energy
    else:
        print("CPLEX failed to converge to a solution.")
        return None, None


def run_solver_benchmark(target_structure, c_coeffs=None):
    """
    Directly benchmarks D-Wave Neal (Quantum Annealing emulator) vs IBM CPLEX (Exact Classical Solver)
    on a given RNA target structure.
    """
    print(f"\n=======================================================")
    print(f"BENCHMARKING STRUCTURE: {target_structure}")
    print(f"=======================================================")
    
    stems = extract_stems(target_structure)
    if not stems:
        print("No paired stems found to benchmark.")
        return
        
    if c_coeffs is None:
        print("Calculating QUBO coefficients...")
        c_coeffs = calculate_qubo_coeffs(method="ols")
        
    Q_dict, offset = build_approx_qubo(stems, c_coeffs)
    num_vars = len(set([u for pair in Q_dict.keys() for u in pair]))
    print(f"QUBO generated with {num_vars} binary variables and {len(Q_dict)} interaction terms.\n")
    
    # --- 1. SOLVE WITH D-WAVE NEAL (ANNEALING) ---
    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q_dict, num_reads=1000)
    neal_time = time.time() - start_time
    
    neal_best = sampleset.first.sample
    neal_energy = sampleset.first.energy + offset
    neal_pairs = decode_bits_to_pairs(neal_best, stems)
    
    # --- 2. SOLVE WITH IBM CPLEX (CLASSICAL BRANCH & BOUND) ---
    start_time = time.time()
    cplex_sample, cplex_raw_energy = solve_qubo_with_cplex(Q_dict)
    cplex_time = time.time() - start_time
    
    if cplex_sample:
        cplex_energy = cplex_raw_energy + offset
        cplex_pairs = decode_bits_to_pairs(cplex_sample, stems)
    else:
        cplex_energy, cplex_pairs = None, "CPLEX Failed"
        
    # --- 3. COMPARISON RESULTS TABLE ---
    print(f"{'Solver':<20} | {'Runtime (sec)':<15} | {'Min QUBO Energy':<18} | {'Decoded Pairs'}")
    print("-" * 80)
    print(f"{'D-Wave Neal (SA)':<20} | {neal_time:<15.4f} | {neal_energy:<18.4f} | {str(neal_pairs)}")
    if cplex_sample:
        print(f"{'IBM CPLEX (Exact)':<20} | {cplex_time:<15.4f} | {cplex_energy:<18.4f} | {str(cplex_pairs)}")
    print("-" * 80)
    
    # Check if annealing found the global minimum
    if cplex_sample and abs(neal_energy - cplex_energy) < 1e-4:
        print("[SUCCESS] D-Wave Neal successfully found the EXACT classical ground state!")
    elif cplex_sample:
        print(f"[INFO] D-Wave Neal energy difference from exact minimum: {abs(neal_energy - cplex_energy):.4f} kcal/mol")


if __name__ == "__main__":
    # IMPORTANT SPEED OPTIMIZATION: Compute coefficients just ONCE here before looping!
    print("Pre-computing QUBO coefficients once for all benchmarks...")
    coeffs = calculate_qubo_coeffs(method="ols") # Using OLS as baseline for fast solver comparisons
    
    # Test across benchmark RNA hairpins of increasing complexity
    test_structures = [
        "(((...)))",
        "((((((((....))))))))",
        "((((((((..........))))))))",
        "(((((((((((((((....)))))))))))))))"
    ]
    
    for struct in test_structures:
        run_solver_benchmark(struct, c_coeffs=coeffs)