import numpy as np
from scipy.linalg import eigh

def calculate_hamming_distance(sample1,sample2):
    """Calculates the number of bit differences between two binary samples."""
    diff_count = 0
    # Assuming both samples have the same keys 
    for key in sample1.keys():
        if sample1.get(key,0) != sample2.get(key,0):
            diff_count +=1
    return diff_count

def run_subspace_diagonalization(samples_list,gamma=5.0):
    """
    Performs Subspace Exact Diagonalization (SQD).
    gamma: The strength of the transverse field (quantum tunneling).
    """
    N = len(samples_list)
    if N==0:
        return None
    #1. Initialize the Hamiltonian matrix (NxN)
    H = np.zeros((N,N))

    #2. Fill in the hamiltonian matrix
    for i in range(N):
        # Diaginal :classical QUBO energy
        H[i,i] = samples_list[i]['energy']

        # off-diagonal: Quantum tunnelling (transverse field)
        for j in range(i+1,N):
            dist = calculate_hamming_distance(samples_list[i]['sample'],samples_list[j]['sample'])
            # in one hot encoding, a single RNA mutations flips 2 bits
            # We allow quantum tunneling between states that are 1 mutation apart 
            if dist <=2:
                H[i,j] = -gamma
                H[j,i] = -gamma
            
            #3. Exact diagonalization (solves quantum system)
            eigenvalues, eigenvectors = eigh(H)

            #4. The ground state is the eigenvector of the lowest eigenvector of the lowest eigenvalue
            ground_state_vector = eigenvectors[:,0]
            #5. Find the sequence with the highest prob amplitude in the quantum superposition
            probabilities = ground_state_vector **2
            best_index = np.argmax(probabilities)
            best_candidate = samples_list[best_index]
            return best_candidate['seq'] ,eigenvalues[0]
            # returns the sequecne and the new quantum energy 

