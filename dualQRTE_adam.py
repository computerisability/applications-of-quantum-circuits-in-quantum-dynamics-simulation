# Import the relevant libraries
from qiskit import Aer, transpile, execute, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector, QuantumCircuit, Parameter
from qiskit.opflow import StateFn, CircuitSampler, PauliExpectation, Gradient, PauliSumOp, CircuitStateFn
from qiskit.quantum_info import Statevector, SparsePauliOp, state_fidelity
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Estimator
from qiskit.utils import QuantumInstance
from qiskit.algorithms.gradients import ParamShiftEstimatorGradient
import numpy as np
from utils import *
import time
from qiskit.algorithms.gradients import DerivativeType, LinCombQGT, LinCombEstimatorGradient 
from qiskit.algorithms.time_evolvers.variational.variational_principles import RealMcLachlanPrinciple 
from qiskit.algorithms.optimizers import COBYLA, ADAM, NELDER_MEAD, NFT, QNSPSA, SPSA, L_BFGS_B, SciPyOptimizer
from collections import deque
from scipy.optimize import minimize
from scipy import optimize

backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)

class DualQRTE:
    """ Class implementing the dual QRTE algorithm """

    def __init__(
            self,
            ansatz,
            maxiter,
            gradient_tolerance,
            expectation,
            quantum_instance
    ):
        """
        Args:
            ansatz (Ansatz class): contains the initial conditions and initial ansatz (if provided)
            maxiter(int): maximum number of optimization steps
            gradient_tolerance (float): gradient tolerance (\delta) to stop the minimization routine
            expectation (ExpectationBase): expectation converter to evaluate expectation values
            quantum_instance (QuantumInstance): quantum backend to evaluate the circuits
        """
        self.ansatz = ansatz
        self.maxiter = maxiter
        self.gradient_tolerance = gradient_tolerance
        self.expectation = expectation
        self.quantum_instance = quantum_instance
        self.n_qubits = len(self.ansatz.circuit.qubits)
        self.sampler = CircuitSampler(quantum_instance)

    def get_fidelity_gradient(
            self,
            circuit_ansatz,
            var_parameters
    ):
        """ Evaluate the fidelity and it's gradient given the current parameters and the circuit ansatz.

        Args:
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current numerical values for variational parameters.

        Returns:
            A function to evaluate the fidelity and a function to evaluate the gradient
        """

        ansatz_old = circuit_ansatz.assign_parameters(var_parameters)
        
        shift = ParameterVector("dw", circuit_ansatz.num_parameters)
        shifted_ansatz = circuit_ansatz.assign_parameters(var_parameters + shift)

        state = StateFn(shifted_ansatz & ansatz_old.inverse())
        overlap = StateFn(projector_zero_global(self.n_qubits),is_measurement=True) @ state
        overlap = self.expectation.convert(overlap)
        
        def fidelity(shift_values: np.array):
            if isinstance(shift_values, list):
                shift_values = np.asarray(shift_values)
                value_dict = {x_i: shift_values[:, i].tolist() for i,x_i in enumerate(shift)}
            else:
                value_dict = dict(zip(shift, shift_values))

            sampled_overlap = self.sampler.convert(overlap, params=value_dict)
            return np.real(sampled_overlap.eval())
        
        def gradient(shift_values: np.array):
            dim = shift_values.size
            plus_shifts = (shift_values + np.pi/2 * np.identity(dim)).tolist()
            minus_shifts = (shift_values - np.pi/2 * np.identity(dim)).tolist()
            evaluated = fidelity(plus_shifts + minus_shifts)
            return (evaluated[:dim] - evaluated[dim:])/2
       
        return fidelity, gradient


    def minimization_routine(
            self,
            circuit_ansatz,
            var_parameters,
            shift_init_guess,
            b,
            dtau
    ):
       
        fidelity, gradient = self.get_fidelity_gradient(circuit_ansatz, var_parameters)

        def loss_function(dtheta):
            fid = fidelity(dtheta)
            loss = (1-fid)/2 - dtau * np.dot(b, dtheta)
            return loss

        def loss_grad_function(dtheta):
            grad = gradient(dtheta)
            grad = -0.5 * grad - dtau * b
            return grad 


        updated_shift = np.copy(shift_init_guess)
        loss = loss_function(updated_shift)
        grad_norm = max(np.abs(loss_grad_function(updated_shift)))  # infinite norm

        # Initialize the 1st and 2nd moment vectors of Adam
        m = np.zeros(len(var_parameters))
        v = np.zeros(len(var_parameters))

        count = 0  # to count the number of iterations of Adam
        
        
        while grad_norm > self.gradient_tolerance and count < self.maxiter:  
            count += 1

            grad = loss_grad_function(updated_shift)
            grad_norm = max(np.abs(grad))

            updated_shift, m, v = adam_step(updated_shift, count, m, v, grad)

            # Get the optimized fidelity
            loss = loss_function(updated_shift)
            
            # Print statement
            if count % 20 == 0:
                print(f'Iteration: {count}   Fidelity: {loss}    Gradient: {grad_norm}')
        
        return updated_shift, loss

        #return updated_shift, result.fun

    def one_time_step(
            self,
            hamiltonian,
            circuit_ansatz,
            var_parameters,
            time_step,
            shift_init_guess
    ):
        """ Advance the time evolution by one time step dt in time.

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current variational parameters.
            time_step (float): small time step.
            shift_init_guess (np.array): guess for the classical optimization

        Returns:
            the new variational parameters 
        """

        estimator = Estimator()
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
        variational_principle = RealMcLachlanPrinciple(gradient=gradient)
        b = variational_principle.evolution_gradient(hamiltonian, circuit_ansatz, var_parameters)
        #print(b)


#        from scipy.optimize import minimize_scalar
#
#        iteration_data = {'x': [], 'f': [], 'step': [], 'success': False}
#
#        def callback(xk):
#            current_f = objective_function(xk)
#            current_step = len(iteration_data['x']) + 1
#            iteration_data['x'].append(xk[0])
#            iteration_data['f'].append(current_f)
#            iteration_data['step'].append(current_step)
#            print(f"迭代 {current_step}: x = {xk[0]:.6f}")
#            #print(f"迭代 {current_step}: x = {xk[0]:.6f}, f(x) = {current_f:.6f}")
# 
#        def objective_function(dtau):
#            updated_shift, L = self.minimization_routine(circuit_ansatz,var_parameters,shift_init_guess,b,dtau)
#            return L/(dtau**2) 
#
#        result = minimize(objective_function, x0=[0.01], bounds=[(0.001, 0.1)],  method='L-BFGS-B', options={'disp': True})
#        #result = minimize(objective_function, x0=[0.01], bounds=[(0.001, 0.1)],  method='L-BFGS-B', callback=callback, options={'disp': True})
#        dtau_best = result.x
#        updated_shift_best, L_best = self.minimization_routine(circuit_ansatz,var_parameters,shift_init_guess,b,dtau_best)
#        

        #result = minimize_scalar(objective_function, bounds=(0.001, 0.1), method='bounded')
#        dtau = result.x

#
#        Ld2_best = 1e10
#        dtau_best = 0.01
#        updated_shift_best = 0
#        #dtau_list = [0.01]
#        #dtau_list = np.linspace(0.001, 0.01, 10).tolist()
#        #dtau_list = np.linspace(0.001, 0.015, 10).tolist()
#        dtau_list = np.linspace(0.01, 0.02, 10).tolist()
#        for dtau in dtau_list:
#            updated_shift, L = self.minimization_routine(circuit_ansatz,var_parameters,shift_init_guess,b,dtau)
#            Ld2 = L/(dtau**2) 
#            if Ld2 < Ld2_best: 
#               Ld2_best = Ld2 
#               dtau_best = dtau
#               updated_shift_best = updated_shift
#            
#            print(f"dtau = {dtau:.6f}, Ld2 = {Ld2:.6f}")
# 
#        print(f"dtau_best = {dtau_best:.6f}, Ld2_best = {Ld2_best:.6f}")
#
#        return var_parameters + (time_step/dtau_best)*updated_shift_best
#
        dtau = 0.01
        updated_shift, L = self.minimization_routine(circuit_ansatz,var_parameters,shift_init_guess,b,dtau)
        return var_parameters + (time_step/dtau)*updated_shift

    def evolve(
            self,
            hamiltonian,
            num_time_steps,
            final_time,
            initial_parameters,
            shift_init_guess,
            observables
    ):
        """ Main function performing the actual time evolution.

        Args:
            hamiltonian (functools.partial): Hamiltonian used for the time evolution.
            num_time_steps (int): number of time steps for the time evolution.
            final_time (float): final time of the evolution
            initial_parameters (np.array): initial parameters of the circuit ansatz.
            shift_init_guess (np.array): guess for the classical optimization of the infidelity
            observables (List[PauliOp]): list of Pauli strings to evaluate

        Returns:
            data_log (dict): contains all the information/results of the algorithm.
        """

        evaluate_observables = get_observable_evaluator(self.ansatz.circuit, observables, self.expectation, self.sampler)
        
        decomposed_circ = self.ansatz.circuit.decompose(reps=3)
        #print(f'current circuit: \n{decomposed_circ}')
        depth = decomposed_circ.depth()
        print(f'Depth    :{depth}')
        print(decomposed_circ.count_ops())
        cnots = decomposed_circ.count_ops().get('cx', 0)
        
        time_step = final_time / num_time_steps
        times = np.linspace(0, final_time, num_time_steps + 1).tolist()  # +1 to include t=0
        
        # Initialize the data log at t=0
        data_log = {"times": times,
                    "parameters": [initial_parameters],
                    
                    "Depth":[1],
                    "CNOTS":[0],
                    "parameter_number":[0],
                    
                    "observables values": [evaluate_observables(initial_parameters)],
                    "evolved state": []
                    }
        evolved_state = self.ansatz.circuit.bind_parameters(initial_parameters)
        data_log["evolved state"].append(evolved_state)

        # Loop over all time steps (to reach the final time)
        for tt in times[1:]:
            print(f"Time : {tt:<10.3f}")
            next_parameters = self.one_time_step(
                hamiltonian,
                self.ansatz.circuit,
                data_log["parameters"][-1],
                time_step,
                shift_init_guess
            )
            #shift_init_guess = next_parameters - data_log["parameters"][-1]
            
            gate_counts = self.ansatz.circuit.count_ops()

# 提取目标门数量（如果不存在则返回0）
            x_count = gate_counts.get('rx', 0)
            y_count = gate_counts.get('ry', 0)
            z_count = gate_counts.get('rz', 0)
            xx_count = gate_counts.get('rxx', 0)  # 或者 'rxx'（取决于库的实现）
            yy_count = gate_counts.get('ryy', 0)  # 或者 'ryy'
            zz_count = gate_counts.get('rzz', 0)  # 或者 'rzz'

            print(f"X gates: {x_count}")
            print(f"Y gates: {y_count}")
            print(f"Z gates: {z_count}")
            print(f"XX gates: {xx_count}")
            print(f"YY gates: {yy_count}")
            print(f"ZZ gates: {zz_count}")
            
            
            
            decomposed_circ = self.ansatz.circuit.decompose(reps=3)
            print(f'current circuit: \n{self.ansatz.circuit}')
            depth = decomposed_circ.depth()
            print(f'Depth    :{depth}')
            
            print(decomposed_circ.count_ops())
            
            cnots = decomposed_circ.count_ops().get('cx', 0)
            
            
            shift_init_guess = np.zeros(len(next_parameters))
            evolved_state = self.ansatz.circuit.bind_parameters(next_parameters)

            #print("parameters = ", data_log["parameters"][-1])        
            # Compute and store the relevant quantities
            data_log["parameters"].append(next_parameters)
            
            data_log["Depth"].append(depth)
            data_log["CNOTS"].append(cnots)
            data_log["parameter_number"].append(len(next_parameters))
            
            data_log["observables values"].append(evaluate_observables(next_parameters))
            data_log["evolved state"].append(evolved_state)
            #print("next parameters = ", data_log["parameters"][-1])        
            #break    
            print("Z0 = ", data_log["observables values"][-1])        
            #if tt >= 0.05: break    
            #if tt >= times[20]: break    
        #print("Z0 = ", data_log["observables values"][-1])        
        return data_log
