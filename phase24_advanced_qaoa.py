import numpy as np
from scipy.optimize import minimize
from contextlib import nullcontext

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram
from qiskit.primitives import BaseSamplerV2, BackendSamplerV2

class TranspilingSamplerV2(BaseSamplerV2):
    def __init__(self, backend):
        # We must dynamically import SamplerV2 from runtime because this is the IBM hardware path
        from qiskit_ibm_runtime import SamplerV2
        self.sampler = SamplerV2(mode=backend)
        self.pm = generate_preset_pass_manager(optimization_level=1, target=backend.target)
    def run(self, pubs, **kwargs):
        transpiled_pubs = []
        for pub in pubs:
            t_circuit = self.pm.run(pub[0])
            transpiled_pubs.append((t_circuit, *pub[1:]))
        return self.sampler.run(transpiled_pubs, **kwargs)

class SimpleQAOAExecutor:
    def __init__(self, reps=1, maxiter=500, use_rqaoa=False, use_ibm_hardware=False, ibm_token=None):
        """
        Initialize QAOA solver.
        - use_rqaoa: If True, wraps QAOA in RecursiveMinimumEigenOptimizer (RQAOA).
        - use_ibm_hardware: If True, runs the final circuit on a real IBM QPU.
        """
        self.reps = reps
        self.maxiter = maxiter
        self.use_rqaoa = use_rqaoa
        self.use_ibm_hardware = use_ibm_hardware
        
        # 1. Classical optimizer
        self.optimizer = COBYLA(maxiter=maxiter)
        
        if self.use_ibm_hardware:
            print("Connecting to IBM Quantum (This may take a minute)...")
            if ibm_token:
                QiskitRuntimeService.save_account(
                    channel="ibm_cloud",
                    token=ibm_token,
                    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/0824fef28e49483fb1c85f49f4969651:48d741c8-bbcc-4954-aff0-3aedeacf459e::",
                    overwrite=True)
            self.service = QiskitRuntimeService(channel="ibm_cloud")
            self.backend = self.service.least_busy(operational=True, simulator=False)
            print(f"--> Selected REAL Quantum Hardware: {self.backend.name}")
            self.sampler = TranspilingSamplerV2(self.backend)
        else:
            # Use statevector for < 30 qubits, MPS is too slow for dense entanglement!
            self.backend = AerSimulator(method="statevector")
            self.sampler = BackendSamplerV2(backend=self.backend)

        def print_progress(eval_count, parameters, mean, std):
            print(f"   ... QAOA Iteration {eval_count}: Current Energy = {mean:.2f}")

        # QAOA instance for RQAOA fallback
        self.qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.optimizer,
            reps=reps,
            callback=print_progress,
            transpiler=None
        )
        self.min_eigen_optimizer = MinimumEigenOptimizer(self.qaoa)
        
        if self.use_rqaoa:
            print("--> Wrapping QAOA in RecursiveMinimumEigenOptimizer (RQAOA)...")
            self.optimizer_algo = RecursiveMinimumEigenOptimizer(
                self.min_eigen_optimizer, 
                min_num_vars=1, 
                min_num_vars_optimizer=self.min_eigen_optimizer
            )
        else:
            self.optimizer_algo = self.min_eigen_optimizer

    def run_qaoa_with_samples(self, qp: QuadraticProgram, top_n: int = 10):
        print(f"--> Running QAOA and extracting Top {top_n} candidate samples...")
        
        if not self.use_rqaoa:
            if self.use_ibm_hardware:
                print("--> [NATIVE V2 PATH] Using Session + IBM EstimatorV2 for extreme hardware performance!")
            else:
                print("--> [NATIVE V2 PATH] Using Aer EstimatorV2 (MPS) for efficient large-scale simulation!")
            
            # 1. Convert QUBO to Ising Operator
            operator, offset = qp.to_ising()
            
            # 2. Build Ansatz and Transpile
            ansatz = QAOAAnsatz(operator, reps=self.reps)
            
            if self.use_ibm_hardware:
                ansatz_samp = ansatz.copy()
                ansatz_samp.measure_all()
                pm = generate_preset_pass_manager(optimization_level=1, target=self.backend.target)
                isa_circuit_samp = pm.run(ansatz_samp)
                isa_circuit_est = pm.run(ansatz)
                isa_operator = operator.apply_layout(isa_circuit_samp.layout)
            else:
                ansatz_samp = ansatz.copy()
                ansatz_samp.measure_all()
                isa_circuit_samp = ansatz_samp.decompose(reps=3)
                isa_circuit_est = ansatz.decompose(reps=3)
                isa_operator = operator
            
            # 3. Smart initialization (near zero) to avoid Barren Plateaus at large scale
            x0 = np.random.uniform(-0.01, 0.01, size=isa_circuit_est.num_parameters)
            
            if self.use_ibm_hardware:
                from qiskit_ibm_runtime import EstimatorV2, SamplerV2
                # IBM Open Plan disabled Session mode. We must use standard job mode.
                session_context = nullcontext()
            else:
                session_context = nullcontext()
                
            with session_context as session:
                if self.use_ibm_hardware:
                    estimator = EstimatorV2(mode=self.backend)
                    sampler_v2 = SamplerV2(mode=self.backend)
                else:
                    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2, SamplerV2 as AerSamplerV2
                    opts = {'backend_options': {'method': 'statevector'}}
                    estimator = AerEstimatorV2(options=opts)
                    sampler_v2 = AerSamplerV2(options=opts)
                
                iteration_count = [0]
                
                def cost_func(params, ansatz_c, hamiltonian, est):
                    pub = (ansatz_c, [hamiltonian], [params])
                    result = est.run(pubs=[pub]).result()
                    energy = result[0].data.evs[0]
                    iteration_count[0] += 1
                    print(f"   ... QAOA Iteration {iteration_count[0]}: Current Energy = {energy:.2f}")
                    return energy
                
                print("--> Starting COBYLA Optimization...")
                res = minimize(
                    cost_func, 
                    x0, 
                    args=(isa_circuit_est, isa_operator, estimator), 
                    method="COBYLA",
                    options={'maxiter': self.maxiter}
                )
                
                print("--> Optimization complete! Sampling optimal parameters...")
                pub = (isa_circuit_samp, [res.x], 1024)
                sample_result = sampler_v2.run([pub]).result()
                
            # 5. Process Bitstrings
            pub_result = sample_result[0]
            counts = pub_result.data.meas.get_counts()
            
            sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            top_samples = []
            
            for bitstr, count in sorted_counts[:top_n]:
                rev_bitstr = bitstr[::-1]
                int_array = np.array([int(b) for b in rev_bitstr])
                prob = count / 1024.0
                val = qp.objective.evaluate(int_array)
                top_samples.append((int_array, val, prob))
                
            return top_samples
            
        else:
            print("--> [FALLBACK PATH] Running QAOA using MinimumEigenOptimizer...")
            result = self.optimizer_algo.solve(qp)
            sorted_samples = sorted(result.samples, key=lambda s: s.probability, reverse=True)
            best_samples = sorted_samples[:top_n]
            return [(s.x, s.fval, s.probability) for s in best_samples]
