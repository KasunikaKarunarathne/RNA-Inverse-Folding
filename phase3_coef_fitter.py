import numpy as np
from itertools import product
from phase2_turner_energy import get_turner_energy

# 1. Base pair encoding 
# mapping the 6 allowed types to their [p0,p1,p2] binary signatures
PAIR_ENCODING = {
    'AU':[1,1,1],
    'UA':[1,1,0],
    'CG':[1,0,1],
    'GC':[1,0,0],
    'GU':[0,1,1],
    'UG':[0,1,0],
}

ALLOWED_PAIRS = ['AU','UA','CG','GC','GU','UG']

def build_qvector(pair1,pair2):
    """
    Combines the 3 bits of pair1 and 3 bits of pair2 into the 6-bit q vector.

    """
    return PAIR_ENCODING[pair1] + PAIR_ENCODING[pair2]

def calculate_qubo_coeffs():
    """
    Performs the Least-Squares fit: min || Phi * c - E ||^2
    Returns the 22 calculated coefficients.
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
        
    # Solve least square regression
    c_coeffs , residuals, _,_ = np.linalg.lstsq(Phi,E_true,rcond=None)
    print("Regression Complete.")
    print(f"Calculated 22 coefficients: {np.round(c_coeffs, 3)}")

    # if residuals exist it means the qubo is an apporx
    if len(residuals) >0:
        mse = residuals[0]/num_samples
        print(f"Mean Squared Error of approximation: {mse:.4f}")
    return c_coeffs

coefficients = calculate_qubo_coeffs()

