# This is the 36 stacking combinantions 
# ( 6 combinations in each pair * another pair(6 combinations))
# Format: (Pair1, Pair2) -> Energy in kcal/mol
# Pair1 is the 5'-3' pair, Pair2 is the adjacent 5'-3' pair in the stem.
# Example: Pair1='AU', Pair2='GC' means A is next to G on the left strand, 
# and U is next to C on the right strand.

TURNER_2004 = {

    # =========================
    # AU row
    # =========================
    ('AU', 'AU'): -0.930,
    ('AU', 'UA'): -1.100,
    ('AU', 'CG'): -2.110,
    ('AU', 'GC'): -2.240,
    ('AU', 'GU'): -0.550,
    ('AU', 'UG'): -1.360,

    # =========================
    # UA row
    # =========================
    ('UA', 'AU'): -1.100,
    ('UA', 'UA'): -0.930,
    ('UA', 'CG'): -2.110,
    ('UA', 'GC'): -1.360,
    ('UA', 'GU'): -1.440,
    ('UA', 'UG'): -0.550,

    # =========================
    # CG row
    # =========================
    ('CG', 'AU'): -2.110,
    ('CG', 'UA'): -1.360,
    ('CG', 'CG'): -3.260,
    ('CG', 'GC'): -2.360,
    ('CG', 'GU'): -1.360,
    ('CG', 'UG'): -2.110,

    # =========================
    # GC row
    # =========================
    ('GC', 'AU'): -2.360,
    ('GC', 'UA'): -2.240,
    ('GC', 'CG'): -3.420,
    ('GC', 'GC'): -3.260,
    ('GC', 'GU'): -1.440,
    ('GC', 'UG'): -2.510,

    # =========================
    # GU row
    # =========================
    ('GU', 'AU'): -1.270,
    ('GU', 'UA'): -1.010,
    ('GU', 'CG'): -2.510,
    ('GU', 'GC'): -2.110,
    ('GU', 'GU'): -0.500,
    ('GU', 'UG'): -1.270,

    # =========================
    # UG row
    # =========================
    ('UG', 'AU'): -1.010,
    ('UG', 'UA'): -1.270,
    ('UG', 'CG'): -2.110,
    ('UG', 'GC'): -1.360,
    ('UG', 'GU'): -1.270,
    ('UG', 'UG'): -0.500,
}
def get_turner_energy(pair1,pair2):
    """
    Safely get the energy for a 4-nucleotide stack.
    Returns 0 if the pair combination hasn't been added to the dictionary yet.
    """
    return TURNER_2004.get((pair1,pair2),0.0)

def calculate_true_turner_energy(stems, sequence):
    """
    Calculates the exact 6-local biological energy for an entire RNA sequence.
    """
    total_energy =0.0
    for stem in stems:
        for k in range(len(stem)-1):
            # stem must have atlest a stack
            # get two adjacent pairs 
            left1, right1 = stem[k]
            left2, right2 = stem[k+1]

            # extract actual letters from the sequence
            p1_str = sequence[left1] + sequence[right1]
            p2_str = sequence[left2] + sequence[right2]

            # lookup the physical energy and add to the total
            total_energy += get_turner_energy(p1_str,p2_str)

    return total_energy

# ----------------------     TEST   ----------------------------------
# from phase1_rules import extract_stems

# # 1. Define the test variables
# target = "((((....))))"
# stems = extract_stems(target)
# test_sequence = "GCAUAAAAAUGC"

# # 2. Run the function
# calculated_energy = calculate_true_turner_energy(stems, test_sequence)

# # 3. results
# print(f"Target Structure: {target}")
# print(f"Test Sequence:    {test_sequence}")
# print("-" * 30)
# print(f"Calculated Energy: {calculated_energy:.2f} kcal/mol")
# print(f"Expected Energy:   -6.63 kcal/mol")

# if round(calculated_energy, 2) == -6.63:
#     print("SUCCESS: Phase 2 is tracking the physical stacks perfectly!")
# else:
#     print("ERROR: Something went wrong in the stack mapping.")