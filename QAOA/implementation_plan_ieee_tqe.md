# Implementation Plan: IEEE Transactions on Quantum Engineering (TQE) Submission

## Objective
Finalize and upgrade the hybrid quantum-classical RNA inverse folding manuscript into a publication-grade submission targeting **IEEE Transactions on Quantum Engineering (TQE)**. The plan bridges biophysical modeling with quantum systems engineering, incorporating constraint-preserving Hadfield mixers, mathematical encoding proofs, circuit synthesis, constant-depth gate scheduling, and tensor network (MPS) scaling analyses.

---

## Phase 1: Manuscript Text & Mathematical Hardening
*Target Files:* `springer_journal_template/manuscript.tex`, `RNA_clone/bare_jrnl.tex`, `references.bib`

- [ ] **1.1 Integrate Inter-Stem Coaxial Stacking (Section II-B)**
  - Integrate the pedagogical formulation of coaxial stacking.
  - Define flush interface detection condition ($|i_B - j_A| = 1 \lor |i_A - j_B| = 1$) with the concrete dual-hairpin example `((..))((..))`.
  - Explain how the 22-term OLS stacking polynomial is applied across the two terminal base pairs, transforming the block-diagonal QUBO matrix into an interconnected topological network via 9 cross-stem couplers.

- [ ] **1.2 Integrate Extended QUBO & Encoding Proofs (Section II-E)**
  - Insert the global block-matrix structure:
    $$Q_{\text{ext}} = \begin{bmatrix} \mathbf{Q}_{\text{stem}} & \mathbf{0} \\ \mathbf{0} & \mathbf{Q}_{\text{loop}} \end{bmatrix}$$
  - Prove that the 22 stem OLS coefficients remain strictly invariant when loops are added.
  - Detail the algebraic expansion of the One-Hot penalty $(1 - \sum x_{i,b})^2$, showing why $x^2 = x$ reduces the expression to $-P_{\text{1H}}$ linear rewards and $+2P_{\text{1H}}$ quadratic penalties.
  - Formulate the Compact Extended Binary QUBO (2 qubits per loop base), proving the exact quadratic bias tax ($15 b_0 + 5 b_1 - 15 b_0 b_1$) and the 37-term 2nd-degree OLS loop model.

- [ ] **1.3 Integrate Gate-Based QAOA & Hadfield Ring Mixer (Section II-F)**
  - Add the hierarchical system architecture narrative (quantum global conformational search + classical thermodynamic loop completion).
  - Define the 6-cycle ring graph ($C_6$) over the feasible alphabet $\{AU, UA, CG, GC, GU, UG\}$.
  - Provide the exact eigendecomposition $\lambda_k = 2\cos(2\pi k / 6)$ and the circulant unitary matrix $U_M(\beta) = \exp(-i \beta M_6)$.
  - Include the mathematical proof for Warm-Start QAOA showing why $U_P(\gamma_0)$ on a single state produces an unobservable global phase, proving why $U_M(\beta_0)$ must act first.

- [ ] **1.4 Integrate Empirical Benchmark Results (Section III-F)**
  - Insert the unified benchmark results table across FMQA structures (`G-C Placement`, `Simple Hairpin`, `Shortie 4`, `stickshift`, `Small and Easy 6`).
  - Highlight the scientific breakthrough on `Shortie 4` (Classical SA collapsed at 0.0%; Cold QAOA achieved 20.0% by escaping classical kinetic traps).
  - Highlight the warm-start boost on `stickshift` (Classical 66.7% $\to$ Warm QAOA 80.0%, reaching theoretical ground state $-19.88$ kcal/mol).
  - Document optimizer performance: COBYLA converging in 75s vs. Powell at 150s.

- [ ] **1.5 Synchronize Bibliography (`references.bib` / `IEEEexample.bib`)**
  - Ensure all 13 citations resolve cleanly (`walter1994coaxial`, `hadfield2019quantum`, `perera2022battleship`, `mathews2004incorporating`, `turner2009nndb`, `lucas2014ising`, etc.).

---

## Phase 2: Quantum Engineering Enhancements (Tailored for IEEE TQE)

- [ ] **2.1 Hardware Resource & Architecture Trade-off Table**
  - Add an engineering comparison table contrasting:
    1. Unconstrained QAOA ($H_X$ mixer, 3 qubits/pair, $P_{\text{leak}} \approx 90\%$, heavy penalty $W=100$).
    2. Subspace Hadfield QAOA ($M_6$ mixer, 3 qubits/pair, $P_{\text{leak}} = 0\%$, penalty-free $W=0$).
    3. Continuous Quantum Annealing (D-Wave chimera/pegasus minor-embedding, chain strength).
    4. Compact Extended Binary QUBO (2 qubits/loop base, zero illegal states).

- [ ] **2.2 Commuting "Brickwork" Stacking Scheduling (Constant-Depth $O(1)$ Optimization)**
  - Formulate the 2-step parallel gate execution schedule:
    - Step 1 (Even Stacks): Execute non-overlapping stacks $(1,2), (3,4), (5,6)$ simultaneously.
    - Step 2 (Odd Stacks): Execute non-overlapping stacks $(2,3), (4,5), (6,7)$ simultaneously.
  - Prove that because non-overlapping stacks commute, the problem Hamiltonian circuit depth per QAOA layer is strictly constant ($O(1)$) regardless of stem length $K$.

- [ ] **2.3 Gate-Level Native Hardware Decomposition of $M_6$**
  - Detail the synthesis of the 6-state ring mixer into native 2-qubit gates ($CX$, $iSWAP$, or $XX+YY$ Givens rotations) on 3 physical qubits.
  - Calculate exact 2-qubit gate count and circuit depth per base-pair rung.

---

## Phase 3: Solving the Simulation Memory Bottleneck (Remaining 3 Structures)

- [ ] **3.1 Identify Classical vs. Quantum Scaling Limits**
  - Document the classical statevector memory wall:
    - $K = 8$ (`stickshift`): $1.68 \times 10^6$ states $\approx 25$ MB RAM.
    - $K = 11$ (`Corner bulge`): $3.62 \times 10^8$ states $\approx 5.8$ GB RAM.
    - $K = 14$ (`InfoRNA`): $7.84 \times 10^{10}$ states $\approx 1.25$ TB RAM.
  - Emphasize that on physical QPUs, memory does not scale exponentially: $K=14$ requires only $14 \times 3 = 42$ physical qubits (well within IBM's 127Q Eagle/Heron processors).

- [ ] **3.2 Implement Matrix Product State (MPS) Simulation in Qiskit Aer**
  - Develop a lightweight Qiskit simulation script using `AerSimulator(method='matrix_product_state')`.
  - Exploit the 1D nearest-neighbor entanglement topology of RNA helices to compress statevectors into 1D tensor trains with bounded bond dimension ($\chi$).
  - Execute simulated QAOA on `Corner bulge` ($K=11$), `Prion` ($K=10$), and `InfoRNA` ($K=14$) within standard workstation RAM (< 500 MB).

---

## Phase 4: Final Validation & Submission Packaging

- [ ] **4.1 Full LaTeX Compilation Verification**
  - Compile `manuscript.tex` under PDFLaTeX and BibTeX.
  - Verify zero undefined citations, zero missing figure references, and zero LaTeX syntax errors.

- [ ] **4.2 Review Against IEEE TQE Reviewer Criteria**
  - Verify clear differentiation from domain bioinformatics (focus on quantum software, encoding proofs, and hardware mapping).
  - Ensure all figures and tables strictly adhere to IEEE Transactions two-column format.
