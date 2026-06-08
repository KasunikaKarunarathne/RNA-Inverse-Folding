# Allowed pairs (6 biologically valid RNA base pair types) and encoding 
ALLOWED_PAIRS = ['AU', 'UA', 'CG', 'GC', 'GU', 'UG']

# since there are 6 pair we need 3 bits to represent them 
# Each pair type is uniquely identified by three binary variables: p_0, p_1, p_2
# p_0: Watson-Crick vs wobble
# p_1: AU/GU-type
# p_2: Orientation
PAIR_ENCODING = {
    'AU': [1, 1, 1],
    'UA': [1, 1, 0],
    'CG': [1, 0, 1],
    'GC': [0, 1, 0],
    'GU': [0, 1, 1],
    'UG': [1, 0, 0]
}

def extract_stems(dot_bracket):
    """
    Parses a dot-bracket string and extracts Stems.
    A stem is defined as a continuous sequence of adjacent base pairs.
    Returns a list of stems, where each stem is a list of tuple pairs.
    """
    # 1. Find all pairs using a stack 
    pairs = []
    bracket_stack = []
    for i,char in enumerate(dot_bracket):
        if char == '(':
            bracket_stack.append(i)
        elif char == ')':
            # pop the most recent : ( and pair it with the current :)
            # j is poped opening bracket
            j = bracket_stack.pop()
            # i is the current element( closing bracket)
            pairs.append((j,i))
    # Sort pairs so they read left-to-right (5' to 3')
    pairs.sort()

    # 2. Group adjacent pairs into discrete stems
    stems =[]
    current_stem =[]

    for i in range(len(pairs)):
        if not current_stem:
            current_stem.append(pairs[i])
        else:
            prev_left, prev_right = current_stem[-1]
            curr_left , curr_right = pairs[i]

            # check whether the current pair belong to the current stem or another new stem 
            if curr_left == prev_left +1  and curr_right == prev_right -1:
                current_stem.append(pairs[i])
            # if it doent belong to the prev stem then start a new stem
            else:
                # only appended to the stems when breaking a pattern
                stems.append(current_stem)
                current_stem = [pairs[i]] 
    
    # append the final stem 
    if current_stem:
        stems.append(current_stem)
    return stems

# ----------------------     TEST   ----------------------------------
# target_structure = "(((...)))(((...)))"
# stems = extract_stems(target_structure)
# for s_idx, stem in enumerate(stems):
#     print(f"Stem {s_idx + 1}: {stem}")
#     print(f"Number of base pairs (m_s) in Stem {s_idx + 1}: {len(stem)}\n")