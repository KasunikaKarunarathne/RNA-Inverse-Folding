# 🔷 Master Plan: QForge — Hybrid Quantum Computing Execution Platform
# 🧬 Application Domain: RNA Inverse Folding via QAOA

> **Project Identity:** *Computer Engineer specializing in Quantum Software, Compilers and Hybrid Quantum-HPC Systems*
> **Researcher:** Kasunika Karunarathne — Computer Engineering, University of Peradeniya
> **Last Updated:** 2026-09-19
> **Current Status:** Level 1 — RNA/QAOA foundation built, QAOA results need improvement, QForge not yet started

---

## 📌 How to Use This Plan

This document is the **single source of truth** for the entire project. If you lose a chat session, start a new one by referencing this file. The plan has **two parallel tracks** that converge:

| Track | Focus | Produces |
|:---|:---|:---|
| **Track A** — Research | RNA Inverse Folding via Quantum Computing | Research paper, scientific results |
| **Track B** — Engineering | QForge Execution Platform | Portfolio project, engineering skills |

After completing each sub-task, update the status markers:

| Marker | Meaning |
|:---:|:---|
| `[ ]` | Not started |
| `[/]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked / Needs decision |
| `[~]` | Skipped (documented why) |

---

## 🏗️ QForge Architecture Overview

```
                 ┌─────────────────────┐
                 │   User Application  │  ◄── Track A: RNA / MaxCut / QUBO
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Problem Translator  │  ◄── QUBO / Hamiltonian builder
                 └──────────┬──────────┘
                            │
                            ▼
              ╔══════════════════════════════╗
              ║      QUANTUM COMPILER        ║  ◄── Track B: IR, Passes, Routing
              ║  Parser → IR → Passes →      ║
              ║  Routing → Hardware Mapping  ║
              ╚══════════════╤═══════════════╝
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Quantum Runtime    │  ◄── Track B: Scheduler, Queue
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │    CPU     │ │    GPU     │ │    QPU     │  ◄── Track B: Backends
       │ Simulator  │ │ Simulator  │ │  Backend   │
       └────────────┘ └────────────┘ └────────────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Results / Benchmark │  ◄── Both Tracks converge here
                 └─────────────────────┘
```

---

## 📊 Current State: Where We Are Right Now

### What's Already Built (Track A — RNA Research)

| Component | Phase | Status | Notes |
|:---|:---|:---:|:---|
| Stem extraction & pair encoding | Phase 1 | ✅ Done | `phase1_rules.py` |
| Turner nearest-neighbor energy | Phase 2 | ✅ Done | `phase2_turner_energy.py` |
| OLS QUBO coefficient fitter | Phase 3 | ✅ Done | R²=0.890, RMSE=0.186 |
| QUBO matrix builder | Phase 4 | ✅ Done | `phase4_qubo_builder.py` |
| SA/SQA annealing pipeline | Phases 5-9 | ✅ Done | Strong baseline results |
| Loop penalty system | Phases 10-15 | ✅ Done | Mirror, entropy, terminal penalties |
| 7-way SA/SQA benchmark | Phases 17-18 | ✅ Done | Comprehensive comparison |
| Coaxial stacking QUBO | Phase 20-21 | ✅ Done | Extended QUBO formulation |
| Classical post-processing pipelines | Final | ✅ Done | `final_pipeline_with_post_processing.py`, `full_pipeline.py` (Best classical seeds) |
| HUBO / QAMOO (Phases 27, 30) | Phase 27, 30 | ❌ Excluded | Supervisor strictly rejected QAMOO; problem out of scope for QAMOO |
| Advanced QAOA executor (V2) | Phase 24 | ✅ Done | Native Qiskit V2 primitives |
| QAOA benchmarking | Phase 25 | ⚠️ Partial | Good on simple, fails on multi-stem |
| Extended QAOA (all-quantum) | Phase 26 | ⚠️ Limited | Concept proven, qubit explosion |

### Current QAOA Results (The Problem)

| Target Structure | Qubits | Success Rate | Spearman | Diagnosis |
|:---|:---:|:---:|:---:|:---|
| `((((...)))).` | 12 | **68.0%** | 0.83 | ✅ Good — simple hairpin |
| `(((((......)))))` | 15 | **94.0%** | 0.92 | ✅ Excellent — simple hairpin |
| `((....)).((.....))` | 12 | **0.0%** | -0.11 | ❌ Multi-stem total failure |
| `..((((((((.....)).))))))..` | 24 | **24.0%** | 0.87 | ⚠️ Degraded — large nested stem |
| `(((((.....))..((.........)))))` | 21 | **0.0%** | 0.48 | ❌ Multi-stem failure |
| 3 structures (30-41 qubits) | >25 | N/A | N/A | ❌ OOM — too many qubits |

### Root Cause Analysis

| Failure Mode | Affected Structures | Root Cause |
|:---|:---|:---|
| Multi-stem collapse | 3-way junction, nested multi-stem | p=1 QAOA has limited expressiveness; penalty imbalance drowns physics signal for disconnected stems |
| Qubit scalability wall | Structures needing >25 qubits | Statevector sim needs 2^n memory; MPS poor for dense QAOA entanglement |
| Optimizer stagnation | 24-qubit structure (24% success) | COBYLA trapped in local minima; single-run without restarts |
| Anti-correlation | `((....)).((.....))` (Spearman -0.11) | QUBO energy landscape fundamentally misleading QAOA for this topology |

### What's NOT Built Yet (Track B — QForge)

Everything in Track B is new. We start from scratch at Level 2.

---

## 📋 Target GitHub Repository Structure

```
qforge/
│
├── applications/              ◄── Level 1
│   ├── rna_inverse_folding/   (existing RNA work, refactored)
│   ├── maxcut/                (simple demo problem)
│   └── qaoa/                  (shared QAOA logic)
│
├── compiler/                  ◄── Level 2
│   ├── parser/                (QASM/circuit → IR)
│   ├── ir/                    (Quantum Intermediate Representation)
│   ├── passes/                (gate cancellation, rotation merging, depth opt)
│   ├── optimizer/             (pass manager, cost model)
│   ├── router/                (qubit routing, SWAP insertion)
│   └── hardware_model/        (topology definitions)
│
├── runtime/                   ◄── Level 3
│   ├── scheduler/             (FIFO, shortest-job-first, resource-aware)
│   ├── queue/                 (job queue management)
│   ├── resource_manager/      (backend allocation)
│   └── executor/              (backend dispatch)
│
├── backends/                  ◄── Level 3
│   ├── cpu/                   (NumPy statevector simulator)
│   ├── gpu/                   (CuPy → CUDA acceleration)
│   └── qiskit/                (IBM QPU adapter)
│
├── simulator/                 ◄── Level 3
│
├── benchmarks/                ◄── All Levels
│
├── tests/                     ◄── All Levels
│
├── dashboard/                 ◄── Level 3-4 (observability)
│
├── docs/
│
└── examples/
```

---

# 🟢 LEVEL 1 — Foundation

> **Goal:** Fix the RNA/QAOA pipeline + establish QForge's problem/application layer
> **Skills developed:** Quantum algorithms, Python, scientific computing, Qiskit, benchmarking
> **Timeline estimate:** 4-6 weeks

---

## 1.1 📚 Learn: Variational Quantum Algorithm Foundations

> [!NOTE]
> This learning phase builds the theoretical depth needed for BOTH tracks. Understanding why QAOA fails teaches you what the compiler/runtime must handle.

### 1.1.1 QAOA Theory Deep Dive
- [x] **Read:** Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
  - Understand cost Hamiltonian, mixer Hamiltonian, `p` layers, the role of `γ` and `β`
  - Why p=1 is fundamentally limited — can only explore states reachable by single alternation
- [x] **Read:** Hadfield et al., "From the QAOA to the Quantum Alternating Operator Ansatz" (2019)
  - Generalized/custom mixers that can enforce constraints natively in the circuit
  - XY-mixer for constrained subspaces (one-hot encoding preservation)
- [x] **Hands-on:** Implement a simple 4-qubit MaxCut QAOA from scratch (no Qiskit abstractions)
  - Build the cost unitary manually: `e^{-iγC}`
  - Build the mixer unitary manually: `e^{-iβB}`
  - Optimize with scipy.optimize.minimize
  - **Why:** This forces understanding of what QAOAAnsatz hides from you

### 1.1.2 Barren Plateaus & Trainability
- [ ] **Read:** McClean et al., "Barren Plateaus in Quantum Neural Network Training Landscapes" (2018)
  - Why random initialization → exponentially vanishing gradients as qubit count grows
  - How your `x0 = np.random.uniform(-0.01, 0.01, ...)` helps but doesn't fully solve it
- [ ] **Understand:** The relationship between circuit depth, entanglement, and trainability
  - More reps = better approximation ratio BUT harder to train

### 1.1.3 Warm-Starting Techniques
- [ ] **Read:** Egger et al., "Warm-starting quantum optimization" (IBM, 2021)
  - Use classical solutions to initialize quantum states or parameters
- [ ] **Read:** Tate et al., "Bridging classical and quantum with SDP initialized warm-started QAOA" (2023)
- [ ] **Key insight for your project:** You already have high-quality classical sequences from `final_pipeline_with_post_processing.py` and `full_pipeline.py` (Phase 20-21) — they can directly warm-start QAOA!

### 1.1.4 Qiskit Runtime V2 Mastery
- [ ] Complete: [IBM Quantum Learning](https://learning.quantum.ibm.com/) — QAOA/VQE sections
- [ ] **Understand deeply:** `EstimatorV2` vs `SamplerV2`, transpilation levels, ISA circuits
- [ ] **Practice:** Run the same circuit through optimization levels 0, 1, 2, 3 and compare gate counts/depth

### ✅ Checkpoint 1.1 — Learning Validation
> **Deliverable:** Write `summaries/level1_foundations.md` explaining in your own words:
> 1. Why p=1 QAOA is limited for multi-stem RNA structures
> 2. What barren plateaus are and how they affect your 24-qubit case
> 3. How warm-starting with SA/SQA solutions could help
> 4. The difference between EstimatorV2 and SamplerV2 and when to use each

---

## 1.2 🔬 Research & Diagnose: Why QAOA Fails on Multi-Stem

### 1.2.1 Multi-Stem Failure Analysis
- [ ] Extract the QUBO matrix for `((....)).((.....))` and visualize its structure
  - Compare sparsity pattern against single-stem `(((((......)))))` QUBO
  - Hypothesis: disconnected stem blocks create multiple degenerate ground states
- [ ] Run QAOA with `shots=8192` (not 1024) to get better probability estimates
- [ ] Plot the probability distribution of all measured bitstrings — is it peaked or flat?
  - Flat = QAOA isn't concentrating on any solution → landscape problem
  - Peaked on wrong solutions → penalty/encoding problem

### 1.2.2 Penalty Weight Sensitivity Study
- [ ] Sweep `penalty_weight` = [1, 2, 5, 10, 20, 50, 100] on ALL 5 computable structures
- [ ] Generate plots: Success Rate vs Penalty Weight, Spearman Correlation vs Penalty Weight
- [ ] Find structure-specific sweet spots
- [ ] **Decision point:** Should penalty_weight be dynamically set per structure?

### 1.2.3 QAOA Depth (Reps) Scaling Study
- [ ] Run with `reps` = [1, 2, 3, 5] on 12-qubit and 15-qubit structures
- [ ] Measure: convergence speed (iterations to plateau), final energy, success rate
- [ ] **Trade-off analysis:** More reps = better approximation BUT more parameters for optimizer

### 1.2.4 Classical Optimizer Comparison
- [ ] Test on the 15-qubit structure (where we have 94% baseline):
  - COBYLA (current) — gradient-free, simple
  - SPSA — designed for noisy quantum function evaluations
  - L-BFGS-B — gradient-based (via parameter shift rule)
  - Nelder-Mead — simplex method
- [ ] Compare: function evaluations to convergence, final energy, success rate
- [ ] Test multi-start strategy: 5-10 random restarts, keep best

### ✅ Checkpoint 1.2 — Diagnostic Report
> **Deliverable:** `implementation_plan/qaoa_diagnostic_report.md` with:
> - QUBO matrix visualizations (single-stem vs multi-stem)
> - Penalty sensitivity plots
> - Depth scaling plots
> - Optimizer comparison table
> - **Clear recommendation** for the Level 1.3 implementation path

---

## 1.3 🛠️ Implement: Fix QAOA + Build QForge Problem Layer

### Track A: QAOA Pipeline Improvements

#### 1.3.1 Warm-Start QAOA
- [ ] **Strategy A — Parameter warm-start:**
  - Solve with SA/SQA first (fast, already works)
  - Use the classical optimal bitstring to compute approximate initial `γ`, `β`
- [ ] **Strategy B — State warm-start:**
  - Prepare initial quantum state biased toward the SA/SQA solution
  - Replace uniform superposition `|+⟩` with a biased initial state
- [ ] Benchmark both strategies against cold-start QAOA

#### 1.3.2 Adaptive QAOA Depth
- [ ] Start with p=1, evaluate quality
- [ ] If success < threshold → increase to p=2 and re-optimize
- [ ] Implement "layerwise training" — fix layer 1 params, optimize layer 2 only
- [ ] Add parameter transfer: optimal p=1 params become first layer of p=2

#### 1.3.3 Improved Optimizer Loop
- [ ] Implement multi-start optimization (5-10 random restarts, keep best)
- [ ] Implement SPSA as primary optimizer
- [ ] Add early stopping if energy converges (save compute)
- [ ] Increase shots from 1024 → 4096 for better gradient estimates

#### 1.3.4 Structure-Aware Penalty Tuning
- [ ] Use `calculate_structural_difficulty()` from Phase 17 to auto-scale penalty_weight
- [ ] Multi-stem structures get lower penalty to avoid drowning physics signal
- [ ] Implement per-stem penalty scaling based on stem length

#### 1.3.5 Re-Benchmark Improved QAOA
- [ ] Run on all 8 structures with improved pipeline
- [ ] Generate comparison table: Original Phase 25 vs Improved
- [ ] Target: **70%+ success rate** on structures ≤ 25 qubits
- [ ] Generate publication-quality plots

### Track B: QForge Application Layer

#### 1.3.6 Refactor RNA Code into QForge Structure
- [ ] Create `qforge/applications/rna_inverse_folding/` directory
- [ ] Move & organize: `phase1_rules.py` → `qforge/applications/rna_inverse_folding/structure_parser.py`
- [ ] Move & organize: `phase2_turner_energy.py` → `qforge/applications/rna_inverse_folding/energy_model.py`
- [ ] Move & organize: `phase3_coef_fitter.py` → `qforge/applications/rna_inverse_folding/qubo_fitter.py`
- [ ] Move & organize: `phase4_qubo_builder.py` → `qforge/applications/rna_inverse_folding/qubo_builder.py`
- [ ] Create clean API: `RNAProblem(target_structure) → qubo_matrix`

#### 1.3.7 Add MaxCut as Second Problem
- [ ] Implement `qforge/applications/maxcut/` — simple graph → QUBO conversion
- [ ] This proves QForge isn't locked to one application
- [ ] MaxCut is the canonical QAOA benchmark — easy to validate correctness

#### 1.3.8 Abstract Problem Translator
- [ ] Create `qforge/problem_translator.py`
- [ ] Interface: `Problem → QUBO → Hamiltonian (SparsePauliOp)`
- [ ] Both RNA and MaxCut use the same downstream pipeline

### ✅ Checkpoint 1.3 — Level 1 Complete
> **Deliverables:**
> 1. Improved QAOA pipeline with benchmark results (Track A)
> 2. `qforge/applications/` with RNA + MaxCut + Problem Translator (Track B)
> 3. All tests passing: `pytest qforge/tests/test_applications.py`

---

## 1.4 📝 Document & Update

- [ ] Write `summaries/level1_complete.md` — what was learned, what was built, key results
- [ ] Update this master plan with actual results and decisions
- [ ] **Decision point:** Confirm Level 2 direction based on diagnostic findings

---

# 🟡 LEVEL 2 — Quantum Compiler

> **Goal:** Build QForge's quantum compiler: IR, optimization passes, hardware-aware routing
> **Skills developed:** Compiler design, data structures, graph algorithms, quantum architecture
> **Timeline estimate:** 6-8 weeks
> **This is where the project becomes substantially more impressive than a typical Qiskit project.**

---

## 2.1 📚 Learn: Compiler Fundamentals + Quantum Circuit Compilation

### 2.1.1 Classical Compiler Concepts (adapted to quantum)
- [ ] **Study:** What is an Intermediate Representation (IR)?
  - Why compilers don't go directly from source to machine code
  - SSA form, basic blocks, control flow graphs (classical analogues)
  - **Quantum analogue:** Circuit → IR → optimized IR → hardware-mapped circuit
- [ ] **Study:** Compiler passes — the concept of analysis + transformation passes
  - Each pass does ONE thing well
  - Passes compose: Pass1 → Pass2 → Pass3
- [ ] **Hands-on:** Read Qiskit's transpiler source code (it's a real quantum compiler!)
  - `qiskit.transpiler.passes` — study 3-4 passes to understand the pattern

### 2.1.2 Quantum Circuit Representation
- [ ] **Study:** How quantum circuits are represented as DAGs (Directed Acyclic Graphs)
  - Qiskit's `DAGCircuit` — nodes are gates, edges are qubit wires
  - Why DAGs are better than flat instruction lists for optimization
- [ ] **Study:** Standard quantum gate sets
  - Universal gate sets: {H, CNOT, T} or {RZ, SX, CNOT}
  - Native gate sets for IBM hardware: {CX, ID, RZ, SX, X}
  - Gate decomposition: any unitary → native gates

### 2.1.3 Qubit Routing & Mapping
- [ ] **Study:** The qubit mapping problem
  - Logical qubits (your algorithm) → Physical qubits (hardware topology)
  - Not all qubits are connected → need SWAP gates for non-adjacent CNOTs
- [ ] **Study:** Graph algorithms needed:
  - Shortest path (Dijkstra/BFS) on hardware topology
  - Graph coloring for qubit assignment
  - The SABRE routing algorithm (used by Qiskit)
- [ ] **Hands-on:** Implement BFS on a simple grid topology

### 2.1.4 Cost Models for Quantum Circuits
- [ ] **Study:** What metrics matter for quantum circuit quality:
  - Gate count (total, 2-qubit specifically)
  - Circuit depth (critical path length)
  - SWAP count (routing overhead)
  - Estimated execution time
  - Estimated error (based on gate fidelities)

### ✅ Checkpoint 2.1 — Learning Validation
> **Deliverable:** `summaries/level2_compiler_foundations.md` explaining:
> 1. Why quantum circuits need an IR (not just Qiskit QuantumCircuit)
> 2. How DAG representation enables optimization
> 3. The qubit routing problem and why it's NP-hard
> 4. What cost model metrics you'll implement

---

## 2.2 🔬 Research & Choose: Compiler Design Decisions

### 2.2.1 IR Design Decisions
- [ ] **Decision:** What does a QForge IR instruction look like?
  ```python
  # Option A: Simple tuple-based
  Instruction(opcode="CNOT", qubits=[0, 1])
  
  # Option B: Dataclass-based with metadata
  @dataclass
  class QInstruction:
      gate: str
      qubits: list[int]
      params: list[float]  # for RZ(theta), etc.
      depth_level: int
  ```
- [ ] **Decision:** What gates does the IR support?
  - Minimum: H, X, Y, Z, RX, RY, RZ, CNOT, CZ, MEASURE
  - Extended: SWAP, Toffoli, U3
- [ ] **Decision:** DAG-based or list-based IR?
  - DAG: more powerful optimization but more complex to implement
  - List: simpler, good enough for initial passes
  - **Recommendation:** Start with list-based, add DAG later

### 2.2.2 Hardware Topology Modeling
- [ ] **Decision:** What topologies to support?
  - Linear: `0—1—2—3—4`
  - Grid: IBM-like heavy-hex
  - All-to-all: ideal simulator
  - Custom: user-defined adjacency graph
- [ ] **Decision:** How to represent topology?
  ```python
  class HardwareTopology:
      adjacency: dict[int, list[int]]
      gate_errors: dict[tuple[int,int], float]  # optional
      gate_times: dict[str, float]               # optional
  ```

### 2.2.3 Benchmark Qiskit's Transpiler
- [ ] Run your RNA QAOA circuit through Qiskit transpilation levels 0-3
- [ ] Record: gate count, depth, 2Q gates, compilation time at each level
- [ ] This becomes the baseline your compiler needs to be competitive with (or at least comparable to)

### ✅ Checkpoint 2.2 — Design Document
> **Deliverable:** `docs/compiler_design.md` with all design decisions documented and justified

---

## 2.3 🛠️ Implement: The QForge Quantum Compiler

### 2.3.1 Quantum IR
- [ ] Implement `qforge/compiler/ir/instruction.py` — the QInstruction dataclass
- [ ] Implement `qforge/compiler/ir/circuit.py` — QCircuit (ordered list of QInstructions)
- [ ] Implement `qforge/compiler/parser/qiskit_parser.py` — convert Qiskit QuantumCircuit → QCircuit
- [ ] Implement `qforge/compiler/ir/emitter.py` — convert QCircuit → Qiskit QuantumCircuit (round-trip)
- [ ] **Test:** Parse a Qiskit circuit → QForge IR → emit back → verify identical

### 2.3.2 Optimization Passes
- [ ] **Pass 1 — Gate Cancellation:** `X·X = I`, `H·H = I`, `CNOT·CNOT = I`
  ```
  Before: X q0; X q0;     After: (nothing)
  ```
- [ ] **Pass 2 — Rotation Merging:** `RZ(a)·RZ(b) = RZ(a+b)`
  ```
  Before: RZ(0.3) q0; RZ(0.7) q0;     After: RZ(1.0) q0;
  ```
- [ ] **Pass 3 — Depth Optimization:** Parallelize independent gates
  ```
  Before: H q0; H q1; H q2; (depth 3)
  After:  H q0 || H q1 || H q2; (depth 1)
  ```
- [ ] **Pass 4 — Dead Gate Removal:** Remove gates on qubits that are never measured
- [ ] Implement `qforge/compiler/passes/pass_manager.py` — orchestrates passes in sequence
- [ ] **Test each pass:** Input circuit → pass → verify output is correct and shorter

### 2.3.3 Hardware-Aware Routing
- [ ] Implement `qforge/compiler/hardware_model/topology.py` — topology definitions
- [ ] Implement `qforge/compiler/router/basic_router.py`:
  - For each CNOT(i,j) where i,j are not adjacent:
    - Find shortest path from i to j on topology graph
    - Insert SWAP gates along the path
    - Update the logical-to-physical qubit mapping
- [ ] Implement `qforge/compiler/router/sabre_router.py` (stretch goal):
  - SABRE: a more sophisticated routing heuristic used by Qiskit

### 2.3.4 Cost Model
- [ ] Implement `qforge/compiler/optimizer/cost_model.py`:
  ```
  CostReport:
    total_gates: int
    two_qubit_gates: int
    circuit_depth: int
    swap_count: int
    estimated_error: float
    compilation_time_ms: float
  ```
- [ ] Print before/after reports for every compilation:
  ```
  Original circuit:  Depth: 42, 2Q gates: 81
  After optimization: Depth: 27, 2Q gates: 49
  After routing:     Depth: 34, 2Q gates: 63, SWAPs: 14
  ```

### 2.3.5 Integration: RNA QAOA Through QForge Compiler
- [ ] Take the Phase 24 QAOA circuit for a 15-qubit RNA problem
- [ ] Route it through: QForge Parser → IR → Passes → Router → Cost Model
- [ ] Compare QForge compilation vs Qiskit transpiler (levels 0-3)
- [ ] Document the comparison

### ✅ Checkpoint 2.3 — Level 2 Complete
> **Deliverables:**
> 1. Working quantum compiler with IR, 4+ passes, routing, cost model
> 2. Comparison benchmark: QForge compiler vs Qiskit transpiler
> 3. All tests passing: `pytest qforge/tests/test_compiler.py`
> 4. Cost report showing real optimizations on RNA QAOA circuits

---

## 2.4 📝 Document & Update

- [ ] Write `summaries/level2_compiler.md`
- [ ] Update this master plan with results
- [ ] **Decision point:** Prioritize GPU backend or scheduling next?

---

# 🟠 LEVEL 3 — HPC Runtime + Multi-Backend Execution

> **Goal:** Build the execution layer: CPU simulator, GPU acceleration, job scheduling, benchmarking
> **Skills developed:** HPC, CUDA/CuPy, parallel computing, runtime design, systems thinking
> **Timeline estimate:** 6-8 weeks
> **This is your Quantum-HPC identity.**

---

## 3.1 📚 Learn: Simulation, GPU Computing & Runtime Systems

### 3.1.1 Quantum Simulation Fundamentals
- [ ] **Study:** How statevector simulation works
  - State = complex vector of size 2^n
  - Gate application = matrix-vector multiplication (or tensor contraction)
  - Measurement = probabilistic sampling from |amplitude|²
- [ ] **Hands-on:** Implement a 4-qubit statevector simulator from scratch using only NumPy
  - Support: H, X, CNOT, RZ, MEASURE
  - Verify against Qiskit Aer for correctness

### 3.1.2 GPU Acceleration
- [ ] **Study:** Why GPUs are good for quantum simulation
  - Statevector operations are massively parallel matrix operations
  - Memory bandwidth is the bottleneck, not compute
- [ ] **Learn:** CuPy (NumPy API on GPU) — the easiest entry point
- [ ] **Learn:** CUDA basics (optional stretch) — kernels, threads, blocks, shared memory
- [ ] **Learn:** cuQuantum / CUDA-Q (optional stretch) — NVIDIA's quantum-specific GPU library

### 3.1.3 Runtime & Scheduling Concepts
- [ ] **Study:** Job queues (FIFO, priority queues)
- [ ] **Study:** Scheduling algorithms: FIFO, shortest-job-first, resource-aware
- [ ] **Study:** Backend abstraction pattern — how one API dispatches to multiple backends
- [ ] **Hands-on:** Build a simple Python task queue with `asyncio` or `concurrent.futures`

### 3.1.4 RNA Scalability Research (Track A)
- [ ] **Study:** Matrix Product States (MPS) for quantum simulation
  - Why MPS works for 1D entanglement but struggles with dense QAOA
  - Bond dimension truncation and accuracy trade-offs
- [ ] **Study:** Circuit cutting / quantum-centric supercomputing
  - Partition large circuits into smaller sub-circuits
  - Run independently, combine classically
- [ ] **Study:** QUBO decomposition — stem-by-stem QAOA, sliding window (SWIFT-FMQA)

### ✅ Checkpoint 3.1 — Learning Validation
> **Deliverable:** `summaries/level3_hpc_foundations.md` explaining:
> 1. How statevector simulation works (with your from-scratch implementation)
> 2. Why GPU acceleration helps and where the bottleneck is
> 3. Your chosen scheduling strategy and why
> 4. Which scalability approach (circuit cutting / decomposition / MPS) is best for RNA

---

## 3.2 🔬 Research & Choose: Runtime Architecture

### 3.2.1 Backend Strategy
- [ ] **Decision:** What's the CPU simulator interface?
  ```python
  class CPUBackend:
      def run(self, circuit: QCircuit, shots: int) -> ResultCounts
  ```
- [ ] **Decision:** GPU backend approach: CuPy (Level 2) vs CUDA (Level 3) vs cuQuantum (Level 4)?
  - **Recommendation:** Start with CuPy, benchmark speedup, upgrade if needed
- [ ] **Decision:** QPU backend — wrap Qiskit Runtime or direct IBM API?

### 3.2.2 Scheduler Design
- [ ] **Decision:** Scheduling policy selection:
  - Small circuits (< 15 qubits) → CPU (fast, no overhead)
  - Medium circuits (15-25 qubits) → GPU (memory-bound benefit)
  - Large circuits or hardware validation → QPU
- [ ] **Decision:** Job representation:
  ```python
  @dataclass
  class QuantumJob:
      job_id: str
      circuit: QCircuit
      shots: int
      backend: str  # "cpu" | "gpu" | "qpu"
      status: str   # "queued" | "running" | "done" | "failed"
      result: Optional[ResultCounts]
      metrics: JobMetrics  # runtime, memory, etc.
  ```

### 3.2.3 RNA Scalability Decision (Track A)
- [ ] Analyze QUBO graph structure for 30+ qubit RNA structures
- [ ] Prototype stem-by-stem QAOA decomposition on medium structures
- [ ] Compare: full QAOA vs decomposed QAOA vs classical SA/SQA
- [ ] **Decision:** Which decomposition strategy to integrate into QForge

### ✅ Checkpoint 3.2 — Architecture Document
> **Deliverable:** `docs/runtime_architecture.md` with all decisions

---

## 3.3 🛠️ Implement: QForge Runtime

### 3.3.1 CPU Backend (NumPy Statevector Simulator)
- [ ] Implement `qforge/backends/cpu/simulator.py`
  - Statevector initialization: |0...0⟩
  - Gate application via Kronecker product + matrix multiplication
  - Measurement: sample from probability distribution
- [ ] Support: H, X, Y, Z, RX, RY, RZ, CNOT, CZ, SWAP, MEASURE
- [ ] **Test:** Verify against Qiskit Aer on 10 test circuits

### 3.3.2 GPU Backend (CuPy Accelerated)
- [ ] Implement `qforge/backends/gpu/simulator.py`
  - Mirror CPU backend but use CuPy arrays instead of NumPy
  - GPU memory management: allocate/free statevector on GPU
- [ ] **Benchmark:** CPU vs GPU speedup across qubit counts (10, 15, 20, 25)
- [ ] Generate speedup plot — at what qubit count does GPU become faster?

### 3.3.3 QPU Backend (Qiskit Adapter)
- [ ] Implement `qforge/backends/qiskit/adapter.py`
  - Convert QForge IR → Qiskit QuantumCircuit
  - Submit via Qiskit Runtime V2
  - Collect results and convert back
- [ ] The QForge API stays the same:
  ```python
  qforge.submit(circuit, backend="ibm")
  ```

### 3.3.4 Job Scheduler & Queue
- [ ] Implement `qforge/runtime/scheduler.py`
  - FIFO scheduling (baseline)
  - Resource-aware scheduling (route based on qubit count)
- [ ] Implement `qforge/runtime/queue.py` — job queue with status tracking
- [ ] Implement `qforge/runtime/executor.py` — dispatches jobs to backends

### 3.3.5 Benchmarking System
- [ ] Implement `qforge/benchmarks/benchmark_runner.py`
  - Run same circuit on CPU, GPU, QPU
  - Collect: runtime, memory, accuracy, circuit depth, gate count
- [ ] Generate comparison table:
  ```
  Backend | Qubits | Runtime | Memory | Depth | Shots | Fidelity
  CPU     | 20     | 1.2s    | 16MB   | 34    | 4096  | 0.98
  GPU     | 20     | 0.3s    | 16MB   | 34    | 4096  | 0.98
  QPU     | 20     | 45s     | N/A    | 34    | 4096  | 0.87
  ```

### 3.3.6 RNA Scalability Implementation (Track A)
- [ ] Implement the chosen decomposition strategy from 3.2.3
- [ ] Run ALL 8 RNA structures through QForge (including previously OOM ones)
- [ ] Compare: QForge QAOA vs SA/SQA vs Vienna/DesiRNA

### 3.3.7 HPC Experiment (Track B)
- [ ] Run scaling experiment: 10, 20, 30, 40, 50 qubits on CPU vs GPU
- [ ] Measure: At what problem size does GPU acceleration matter?
- [ ] Measure: What is the compilation overhead?
- [ ] Measure: How much does routing increase circuit depth?
- [ ] **This becomes a proper experimental research question.**

### ✅ Checkpoint 3.3 — Level 3 Complete
> **Deliverables:**
> 1. Working CPU + GPU + QPU backends with identical API
> 2. Job scheduler with resource-aware routing
> 3. HPC scaling benchmark (CPU vs GPU vs QPU across qubit counts)
> 4. RNA pipeline running through QForge on all 8 structures
> 5. All tests passing: `pytest qforge/tests/`

---

## 3.4 📝 Document & Update

- [ ] Write `summaries/level3_hpc_runtime.md`
- [ ] Update master plan
- [ ] Add observability dashboard (stretch goal)

---

# 🔴 LEVEL 4 — Advanced Systems + Publication

> **Goal:** Real hardware, advanced algorithms, distributed runtime, publication
> **Skills developed:** Error mitigation, distributed systems, CUDA, C++/Rust (optional), FPGA (optional)
> **Timeline estimate:** 8-12 weeks
> **Even reaching Level 3 properly gives you a very strong portfolio. Level 4 is stretch.**

---

## 4.1 📚 Learn: Advanced Quantum & Systems Topics

### 4.1.1 Quantum Error Mitigation (for Real Hardware)
- [ ] **Study:** Zero-Noise Extrapolation (ZNE) — run at multiple noise levels, extrapolate to zero
- [ ] **Study:** Probabilistic Error Cancellation (PEC) — statistically reconstruct noiseless outcome
- [ ] **Study:** Clifford Data Regression (CDR) — learn noise model from easy-to-simulate circuits
- [ ] **Why this matters:** Even perfect algorithms fail on noisy hardware without mitigation

### 4.1.2 Advanced QAOA Variants (Track A)
- [ ] **Study:** Filtering VQE (F-VQE) — modified landscape for faster convergence
- [ ] **Study:** QAOA with XY-mixer — enforce one-hot constraints in circuit (not penalty)
- [ ] **Study:** VQE with hardware-efficient ansatz — more flexible than QAOA structure
- [ ] **Compare:** Which is most promising for RNA QUBO?

### 4.1.3 Distributed Systems (Track B, stretch)
- [ ] **Study:** FastAPI for REST APIs
- [ ] **Study:** Redis/RabbitMQ for job queues
- [ ] **Study:** Docker for containerization
- [ ] **Goal:** Client → API Gateway → Job Scheduler → {CPU, GPU, QPU}

### 4.1.4 FPGA / Verilog (Optional stretch)
- [ ] **Study:** How FPGAs can accelerate runtime scheduling or measurement post-processing
- [ ] **Implement:** Small Verilog module for fast bitstring → result processing
- [ ] **Warning:** Don't let this derail the core project

---

## 4.2 🔬 Research & Choose: Advanced Paths

### 4.2.1 Hardware Execution Strategy
- [ ] Compare IBM backends (qubit count, error rates, connectivity)
- [ ] Run noise model simulations matching target backend
- [ ] Determine if ZNE/PEC is needed for your problem size
- [ ] Design warm-start hardware pipeline: optimize locally → inject → sample on QPU

### 4.2.2 Advanced Algorithm Selection (Track A)
- [ ] Prototype QAOA with XY-mixer on 12-qubit multi-stem test
- [ ] Prototype VQE with problem-inspired ansatz
- [ ] Compare convergence, solution quality, qubit efficiency
- [ ] **Decision:** Which advanced algorithm to integrate

### 4.2.3 Distributed Runtime Design (Track B)
- [ ] Design API schema for QForge distributed mode
- [ ] Design job submission, status polling, result retrieval flow

---

## 4.3 🛠️ Implement: Advanced Features

### 4.3.1 Real QPU Execution with Error Mitigation
- [ ] Implement warm-start hardware pipeline in QForge:
  1. Optimize γ, β on local Aer simulator (fast)
  2. Apply error mitigation (ZNE via Qiskit Runtime options)
  3. Execute single measurement batch on QPU
- [ ] Run on real IBM hardware with sufficient iterations
- [ ] Compare: Simulator vs Noisy Simulator vs Real Hardware

### 4.3.2 Advanced QAOA for RNA (Track A)
- [ ] Implement chosen advanced algorithm
- [ ] Integrate into QForge pipeline
- [ ] Benchmark: Standard QAOA vs Advanced vs SA/SQA vs Classical

### 4.3.3 Distributed Runtime (Track B, stretch)
- [ ] Implement FastAPI-based QForge server
- [ ] Implement async job submission and result collection
- [ ] Add observability dashboard showing job metrics

### 4.3.4 Observability Dashboard (Track B)
- [ ] Implement `qforge/dashboard/`
  ```
  QForge Dashboard
  ─────────────────
  Jobs completed:     1,284
  CPU utilization:      71%
  GPU utilization:      84%
  Avg queue time:     2.3 s
  Avg compile time:  0.18 s
  Best backend:       GPU
  Largest circuit:  48 qubits
  ```

---

## 4.4 🛠️ Publication (Track A)

### 4.4.1 Final Benchmark Suite
- [ ] All algorithms × All structures × Multiple random seeds
- [ ] Statistical significance testing (confidence intervals)
- [ ] Compare against FMQA (Kikuchi & Tanaka, 2026), DesiRNA, Vienna inverse_fold

### 4.4.2 Paper Draft
- [ ] Write using IEEE/Springer template (already in repo)
- [ ] Structure: Introduction → Related Work → Method → QUBO Formulation → QAOA Pipeline → QForge Integration → Results → Conclusion
- [ ] Generate all publication plots (SVG/PDF)

### 4.4.3 Reproducible Code Package
- [ ] Clean up codebase for public release
- [ ] Write README with setup instructions
- [ ] Add example notebooks

### ✅ Checkpoint 4 — Project Complete
> **Deliverables:**
> 1. Real hardware results with error mitigation
> 2. Full QForge platform (compiler + runtime + 3 backends + benchmarks)
> 3. Research paper draft
> 4. Clean GitHub repository

---

# 📐 Skills Development Tracker

| Skill Area | Current | Target | Built In |
|:---|:---:|:---:|:---|
| **Python + Software Engineering** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | All levels |
| **Quantum Algorithms (QAOA/VQE)** | ⭐⭐ | ⭐⭐⭐⭐⭐ | Level 1, 4 |
| **Qiskit Runtime V2** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Level 1, 4 |
| **QUBO/Ising Formulation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Level 1 |
| **Compiler Design (IR, Passes)** | ⭐ | ⭐⭐⭐⭐ | Level 2 |
| **Graph Algorithms (Routing)** | ⭐ | ⭐⭐⭐⭐ | Level 2 |
| **Linear Algebra (Simulation)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Level 3 |
| **GPU / CUDA / CuPy** | ⭐ | ⭐⭐⭐⭐ | Level 3 |
| **HPC / Parallel Computing** | ⭐ | ⭐⭐⭐⭐ | Level 3 |
| **Runtime Systems / Scheduling** | ⭐ | ⭐⭐⭐⭐ | Level 3 |
| **Quantum Error Mitigation** | ⭐ | ⭐⭐⭐ | Level 4 |
| **Real Hardware Execution** | ⭐⭐ | ⭐⭐⭐⭐ | Level 4 |
| **Distributed Systems / APIs** | ⭐ | ⭐⭐⭐ | Level 4 |
| **RNA Bioinformatics (ViennaRNA)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | All |
| **Academic Writing** | ⭐⭐ | ⭐⭐⭐⭐ | Level 4 |
| **C++/Rust** | ⭐ | ⭐⭐⭐ | Level 4 (optional) |
| **FPGA / Verilog** | ⭐ | ⭐⭐ | Level 4 (optional) |

---

# 🔑 Key Files Reference (Current Codebase)

| File | Purpose |
|:---|:---|
| [`phase1_rules.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase1_rules.py) | Stem extraction, base pair encoding |
| [`phase2_turner_energy.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase2_turner_energy.py) | Turner nearest-neighbor energy lookup |
| [`phase3_coef_fitter.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase3_coef_fitter.py) | OLS coefficient fitting for QUBO |
| [`phase4_qubo_builder.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase4_qubo_builder.py) | QUBO matrix construction |
| [`phase24_advanced_qaoa.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase24_advanced_qaoa.py) | Advanced QAOA executor (Native V2) |
| [`phase25_qaoa_benchmarking.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/phase25_qaoa_benchmarking.py) | QAOA benchmarking pipeline |
| [`full_pipeline.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/full_pipeline.py) | SA/SQA full benchmark pipeline (up to Phase 21) |
| [`final_pipeline_with_post_processing.py`](file:///d:/Academic%20UOP/Internship/simulation/Implementation/NN%20-%20Copy/final_pipeline_with_post_processing.py) | End-to-end classical pipeline with loop post-processing (Best classical sequences source) |

---

# 📋 Decision Log

> Major decisions and reasoning — critical for new chat continuity.

| Date | Decision | Reasoning |
|:---|:---|:---|
| Pre-Phase 25 | OLS (not rank-aware) for QUBO coefficients | Best absolute energy fit: RMSE=0.186, R²=0.890 |
| Phase 24 | Native V2 primitives, not MinimumEigenOptimizer | Legacy wrapper fails on hardware ISA mapping |
| Phase 25 | Reduced penalty_weight 100→10 for QAOA | SA handles 100 fine, but drowns physics signal for QAOA |
| Phase 25 | reps=1 (p=1 QAOA) | Fewer parameters = easier for COBYLA. Trade-off: limited expressiveness |
| Phase 25 | Statevector over MPS for < 30 qubits | MPS too slow for dense QAOA entanglement |
| Phase 26 | RQAOA rejected | Variable elimination violates soft penalty balance → 0% success |
| Phase 26 | Extended QAOA (all nucleotides) concept proven | Qubit count explodes (24→64). Impractical at scale but validates formulation |
| 2026-09-22 | Strictly Exclude QAMOO & HUBO (Phases 27, 30) | Supervisor directive: problem is unsuitable for QAMOO; focus on QUBO up to Phase 21 + final classical pipelines |
| 2026-09-22 | Source best classical seeds from final pipelines | `final_pipeline_with_post_processing.py` and `full_pipeline.py` provide best classical sequences for warm-starting |
| 2026-09-19 | Adopt QForge platform architecture | Transforms RNA research into portfolio-grade engineering project |
| TBD | Level 1.3 optimizer choice | Pending diagnostic results from 1.2 |
| TBD | Level 2 IR format (list vs DAG) | Pending learning phase 2.1 |
| TBD | Level 3 GPU approach (CuPy vs CUDA) | Pending benchmarking |
| TBD | Level 3 RNA scalability strategy | Pending analysis of QUBO graph structure |

---

# 🎯 Career Identity

```
             Computer Engineering
                       │
       ┌───────────────┼───────────────┐
       │               │               │
   Software         Compiler          HPC
       │               │               │
    Python/C++       IR/Graph       CUDA/GPU
       │               │               │
       └───────────────┼───────────────┘
                       │
                Quantum Systems
                       │
             ┌─────────┴─────────┐
             │                   │
        QAOA/RNA             QPU Runtime
             │                   │
             └─────────┬─────────┘
                       │
                     QForge
```

> **Target identity:** *Computer Engineer specializing in Quantum Software, Compilers and Hybrid Quantum-HPC Systems.*

---

> [!IMPORTANT]
> **Next Immediate Action:** Start with **Task 1.1.1** — Read the original QAOA paper. Then proceed through the learning phase before touching any code. Every piece of learning produces a visible artifact (summary document), and every implementation produces testable code.
