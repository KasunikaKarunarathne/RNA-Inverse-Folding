from dimod.core import composite
from numpy.random import beta
from numpy import dtype
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# =====================================================================
# 1. DEFINE BASIC QUANTUM OPERATORS (2x2 MATRICES)
# =====================================================================
I = np.eye(2,dtype=complex)
X = np.array([[0,1],[1,0]],dtype=complex)
Z = np.array([[1,0],[0,-1]],dtype=complex)

def tensor_product(operators):
    """
    Computes the Kronecker (tensor) product of a list of operators."""
    res = operators[0]
    for op in operators[1:]:
        res = np.kron(res,op)
    return res


# =====================================================================
# 2. BUILD THE 4-QUBIT GRAPH HAMILTONIANS (16x16 MATRICES)
# =====================================================================
n =4
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
# (A) Cost hamiltonian : C = sum_{(u, v)} 0.5 * (I - Z_u * Z_v)
"""
Node u	Node v	Are they in different groups?	Product (u * v)	Cut Score Wanted
+1	+1	NO (Same group)	(+1) * (+1) = +1	0
-1	-1	NO (Same group)	(-1) * (-1) = +1	0
+1	-1	YES (Different!)	(+1) * (-1) = -1	1
-1	+1	YES (Different!)	(-1) * (+1) = -1	1

Z * |0> = +1 * |0>
Z * |1> = -1 * |1>

C * |0101> = 4 * |0101>   (Because 4 edges are cut!)
C * |0000> = 0 * |0000>   (Because 0 edges are cut!)
"""
C = np.zeros((2**n,2**n),dtype=complex)
for u,v in edges:
    z_u = tensor_product([Z if i==u else I for i in range(n)])
    z_v = tensor_product([Z if i==v else I for i in range(n)])
    C += 0.5*(np.eye(2**n) -(z_u @ z_v)) #matrix multiplication -@
    #Zu​=Z⊗I⊗I⊗I.

# (B) Mixer Hamiltonian: B = sum_{j=0}^{3} X_j
B = np.zeros((2**n,2**n),dtype=complex)
for j in range(n):
    X_j = tensor_product([X if i==j else I for i in range(n)])
    B += X_j

# (C) Initial State: |s> = |+>^{\otimes 4} (equal superposition of all 16 states)
s = np.ones(2**n,dtype=complex)/np.sqrt(2**n)

# =====================================================================
# 3. DEFINE QAOA SIMULATION & OBJECTIVE FUNCTION
# =====================================================================
def qaoa_circuit(gamma,beta):
    """
    Applies 1 layer of QAOA:
    |psi(gamma, beta)> = exp(-i * beta * B) * exp(-i * gamma * C) * |s>
    """
    # 1. Build cost unitary manually: U_C = exp(-i * gamma * C)
    U_C =expm(-1j*gamma*C)
    # 2. Build mixer unitary manually: U_B = exp(-i * beta * B)
    U_B = expm(-1j*beta*B)
    # evolve the state vector
    state_after_cost = U_C @ s
    state_after_mixer = U_B @ state_after_cost
    return state_after_mixer

def objective(params):
    """Energy expectation value <psi| C |psi> (to maximize cut)."""
    gamma,beta = params
    psi =qaoa_circuit(gamma,beta)
    # expectation value : <psi|C|psi>
    # Exact equivalent of np.vdot(psi, C @ psi):
    probabilities = np.abs(psi)**2
    cut_scores = np.real(np.diag(C))
    expected_cut = np.sum(probabilities * cut_scores)
    # Scipy minimizes, so return negative cut to maximize it:
    return -expected_cut

# =====================================================================
# 4. OPTIMIZE WITH SCIPY (COBYLA)
# =====================================================================
initial_guess =[0.5,0.5]
res = minimize(objective,x0=initial_guess,method='COBYLA')
optimal_gamma,optimal_beta = res.x
max_cut = -res.fun


print("=" * 55)
print("QAOA FROM SCRATCH: 4-QUBIT MAXCUT RESULTS")
print("=" * 55)
print(f"Optimal gamma : {optimal_gamma:.4f} rad  (Analytical approx: pi/4 = 0.7854)")
print(f"Optimal beta  : {optimal_beta:.4f} rad  (Analytical approx: pi/8 = 0.3927)")
print(f"Max Expected Cut: {max_cut:.4f} / 4.0")
print("=" * 55)

# =====================================================================
# 5. MEASUREMENT PROBABILITIES
# =====================================================================
optimal_psi = qaoa_circuit(optimal_gamma,optimal_beta)
probabilities = np.abs(optimal_psi) **2

print("\nFinal Measurement Probabilities (States with > 5% probability):")
print(f"{'State':<10} | {'Cut Value':<10} | {'Probability':<12}")
print("-" * 38)
for x in range(2**n):
    bitstring = format(x, f'0{n}b')
    cut_val = int(np.real(C[x, x]))
    prob = probabilities[x] * 100
    if prob > 5.0:
        highlight = " <-- OPTIMAL MAX-CUT!" if cut_val == 4 else ""
        print(f"|{bitstring}>    | {cut_val:<10} | {prob:.2f}%{highlight}")