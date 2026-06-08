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
    ('AU', 'AU'): -0.9,
    ('AU', 'CG'): -1.8,
    ('AU', 'GC'): -2.3,
    ('AU', 'UA'): -1.1,
    ('AU', 'GU'): -0.5,
    ('AU', 'UG'): -0.7,

    # =========================
    # CG row
    # =========================
    ('CG', 'AU'): -2.1,
    ('CG', 'CG'): -2.9,
    ('CG', 'GC'): -3.4,
    ('CG', 'UA'): -2.3,
    ('CG', 'GU'): -1.5,
    ('CG', 'UG'): -1.5,

    # =========================
    # GC row
    # =========================
    ('GC', 'AU'): -1.7,
    ('GC', 'CG'): -2.0,
    ('GC', 'GC'): -2.9,
    ('GC', 'UA'): -1.8,
    ('GC', 'GU'): -1.3,
    ('GC', 'UG'):  1.9,

    # =========================
    # UA row
    # =========================
    ('UA', 'AU'): -0.9,
    ('UA', 'CG'): -1.7,
    ('UA', 'GC'): -2.1,
    ('UA', 'UA'): -0.9,
    ('UA', 'GU'): -0.7,
    ('UA', 'UG'): -0.5,

    # =========================
    # GU row
    # =========================
    ('GU', 'AU'): -0.9,
    ('GU', 'CG'): -1.7,
    ('GU', 'GC'): -2.1,
    ('GU', 'UA'): -0.9,
    ('GU', 'GU'): -0.5,
    ('GU', 'UG'): -0.5,

    # =========================
    # UG row
    # =========================
    ('UG', 'AU'): -0.9,
    ('UG', 'CG'): -1.7,
    ('UG', 'GC'): -2.1,
    ('UG', 'UA'): -0.9,
    ('UG', 'GU'):  0.6,
    ('UG', 'UG'): -0.5,
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