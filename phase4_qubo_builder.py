from collections import defaultdict

def build_approx_qubo(stems , c_coeffs, penalty_weight=100):
    """
    Constructs the 2-local QUBO dictionary using the fitted coefficients.
    stems: The output from Phase 1.
    c_coeffs: The 22 weights calculated in Phase 3A.
    """
    Q = defaultdict(float)
    constant_offset =0 # Tracks c0 and penalty offsets

    # 1. APPLY INVALID STATE PENALTY
    # Our 3 bits give 8 possible states, but there are only 6 valid RNA pairs.
    # The invalid states are [0,0,0] and [0,0,1]. Both have p_0 = 0 and p_1 = 0.
    # We penalize this using the math: Penalty = W * (1 - p_0) * (1 - p_1)

    for stem in stems:
        for pair in stem:
            p0 = f"p_{pair[0]}_{pair[1]}_0"
            p1 = f"p_{pair[0]}_{pair[1]}_1"

            constant_offset  += penalty_weight
            Q[(p0,p0)] -= penalty_weight
            Q[(p1,p1)] -= penalty_weight

            # ensure the dictionary keys are sorted 
            key = tuple(sorted([p0,p1]))
            Q[key] += penalty_weight

    # 2. APPLY THE 22 TURNER REGRESSION WEIGHTS
    for stem in stems:
        # we only apply nn stacking if the stem has atleast 2 pairs
        for k in range(len(stem) -1):
            pair1 = stem[k]
            pair2 = stem[k+1]

            # create 6 var list (q0 to q5)
            q_vars = [
                f"p_{pair1[0]}_{pair1[1]}_0", f"p_{pair1[0]}_{pair1[1]}_1", f"p_{pair1[0]}_{pair1[1]}_2",
                f"p_{pair2[0]}_{pair2[1]}_0", f"p_{pair2[0]}_{pair2[1]}_1", f"p_{pair2[0]}_{pair2[1]}_2"
            ]
            coeff_i =0
            # add c_0
            constant_offset += c_coeffs[coeff_i]
            coeff_i +=1

            # add c_a
            for a in range(6):
                var = q_vars[a]
                Q[(var,var)] += c_coeffs[coeff_i]
                coeff_i +=1

            # add c_ab
            for a in range(6):
                for b in range(a+1,6):
                    var1 = q_vars[a]
                    var2 = q_vars[b]
                    key = tuple(sorted([var1,var2]))
                    Q[key] += c_coeffs[coeff_i]
                    coeff_i +=1
    return dict(Q), constant_offset

#---------------------------------- Test ----------------------------------------
# Q_dict, offset = build_approx_qubo(stems, coefficients)
# print(f"QUBO generated with {len(Q_dict)} interaction terms.")
# print(f"Global Energy Offset: {offset:.2f}")
