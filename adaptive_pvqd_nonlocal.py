# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:09:38 2024

@author: 83464
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 22:38:28 2024

@author: 83464
"""
import json
# Import the relevant libraries
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector
from qiskit.opflow import StateFn, CircuitSampler
import numpy as np
from utils import *
import time


class AdaptivePVQD:
    """ Class implementing the Adaptive pVQD algorithm """

    def __init__(
            self,
            ansatz,
            maxiter,
            fidelity_tolerance,
            gradient_tolerance,
            expectation,
            quantum_instance,
            #pool_type
    ):
        """
        Args:
            ansatz (Ansatz class): contains the initial conditions and initial ansatz (if provided)
            maxiter(int): maximum number of optimization steps
            fidelity_tolerance (float): fidelity tolerance (\varepsilon) to stop the minimization routine
            gradient_tolerance (float): gradient tolerance (\delta) to stop the minimization routine
            expectation (ExpectationBase): expectation converter to evaluate expectation values
            quantum_instance (QuantumInstance): quantum backend to evaluate the circuits
        """
        self.ansatz = ansatz
        self.maxiter = maxiter
        self.fidelity_tolerance = fidelity_tolerance
        self.gradient_tolerance = gradient_tolerance
        self.expectation = expectation
        self.quantum_instance = quantum_instance
        #self.pool_type = pool_type

        self.n_qubits = len(self.ansatz.circuit.qubits)
        self.sampler = CircuitSampler(quantum_instance)

    def get_loss(
            self,
            hamiltonian,
            circuit_ansatz,
            var_parameters,
            time_step
    ):
        """ Evaluate the infidelity given the current parameters and the circuit ansatz.

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current numerical values for variational parameters.
            time_step (float): small time step.

        Returns:
            A function to evaluate the infidelity and a function to evaluate the gradient
        """

        # Get the time evolved circuit
        trotterized_ansatz = circuit_ansatz.assign_parameters(var_parameters)
        evolution_gate = PauliEvolutionGate(hamiltonian, time=time_step)  # Lie-Trotter with a single decomposition
        trotterized_ansatz.append(evolution_gate, circuit_ansatz.qubits)

        # Get the shifted circuit in parameter space
        shift = ParameterVector("dw", circuit_ansatz.num_parameters)
        shifted_ansatz = circuit_ansatz.assign_parameters(var_parameters + shift)

        # Get the overlap/fidelity between the time evoled and the shifted circuits
        # state = StateFn(shifted_ansatz + trotterized_ansatz.inverse())
        state = StateFn(shifted_ansatz & trotterized_ansatz.inverse())
        overlap = StateFn(projector_zero_local(self.n_qubits), is_measurement=True) @ state
        overlap = self.expectation.convert(overlap)

        def infidelity(shift_values: np.array):
            """ Returns the infidelity value given some numerical values for the small shift parameters. """

            # Create the dictionary of "parameter object: value"
            if isinstance(shift_values, list):
                shift_values = np.asarray(shift_values)
                value_dict = {x_i: shift_values[:, i].tolist() for i, x_i in enumerate(shift)}
            else:
                value_dict = dict(zip(shift, shift_values))

            sampled_overlap = self.sampler.convert(overlap, params=value_dict)
            return 1 - np.real(sampled_overlap.eval())

        def gradient(shift_values: np.array):
            """ Returns the gradient of the infidelity using the parameter-shift rule. """
            dim = shift_values.size

            plus_shifts = (shift_values + np.pi/2 * np.identity(dim)).tolist()
            minus_shifts = (shift_values - np.pi/2 * np.identity(dim)).tolist()

            evaluated = infidelity(plus_shifts + minus_shifts)
            return (evaluated[:dim] - evaluated[dim:])/2

        def gradient_last_comp(shift_values):
            """ Computes only the last entry of the gradient. """

            last_comp_unit_vec = np.zeros(len(shift_values))
            last_comp_unit_vec[-1] = 1

            plus_shift = shift_values + np.pi/2 * last_comp_unit_vec
            minus_shift = shift_values - np.pi/2 * last_comp_unit_vec

            return (infidelity(plus_shift) - infidelity(minus_shift))/2

        return infidelity, gradient, gradient_last_comp

    def adaptive_step(
            self,
            hamiltonian,
            var_parameters,
            time_step,
            shift_init_guess
    ):
        """ Add a gate (from an operator pool) to the current circuit ansatz. Choose the gate that
        maximizes the gradient  (to converge faster to the minimum of the infidelity).

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            var_parameters (np.array): current variational parameters.
            time_step (float): small time step.
            shift_init_guess (np.array): guess for the classical optimization.

        Returns:
            new_circuit_ansatz (QuantumCircuit): circuit with the added gate
        """

        print('--------------------------')
        print(f'Start of an adaptive/block step')

        # Create the operator pool
        operator_pool = self.ansatz.create_op_pool_nonlocal(['x','y','z'], ['xx','yy','zz'])
        #直接用穷举法，构造算子池
        #operator_pool =[('x', [0]), ('x', [1]), ('x', [2]), ('x', [3]),# ('x', [4]), ('x', [5]), ('x', [6]), ('x', [7]),#('x', [8]), ('x', [9]), ('x', [10]), ('x', [11]),
        #                ('y', [0]), ('y', [1]), ('y', [2]), ('y', [3]),# ('y', [4]), ('y', [5]), ('y', [6]), ('y', [7]),#('y', [8]), ('y', [9]), ('y', [10]), ('y', [11]),
        #                ('z', [0]), ('z', [1]), ('z', [2]), ('z', [3]),# ('z', [4]), ('z', [5]), ('z', [6]), ('z', [7]),#('z', [8]), ('z', [9]), ('z', [10]), ('z', [11]),
        #                ('xx', [0, 1]),('xx', [0, 2]),('xx', [0, 3]),#('xx', [0, 4]),('xx', [0, 5]),('xx', [0, 6]),('xx', [0, 7]),#('xx', [0, 8]),('xx', [0, 9]),('xx', [0, 10]),('xx', [0, 11]),
        #                ('xx', [1, 2]),('xx', [1, 3]),#('xx', [1, 4]),('xx', [1, 5]),('xx', [1, 6]),('xx', [1, 7]),#('xx', [1, 8]),('xx', [1, 9]),('xx', [1, 10]),('xx', [1, 11]), 
        #                ('xx', [2, 3]),#('xx', [2, 4]),('xx', [2, 5]),('xx', [2, 6]),('xx', [2, 7]),#('xx', [2, 8]),('xx', [2, 9]),('xx', [2, 10]),('xx', [2, 11]),
                        #('xx', [3, 4]),('xx', [3, 5]),('xx', [3, 6]),('xx', [3, 7]),#('xx', [3, 8]),('xx', [3, 9]),('xx', [3, 10]),('xx', [3, 11]),
                        #('xx', [4, 5]),('xx', [4, 6]),('xx', [4, 7]),#('xx', [4, 8]),('xx', [4, 9]),('xx', [4, 10]),('xx', [4, 11]),
                        #('xx', [5, 6]),('xx', [5, 7]),#('xx', [5, 8]),('xx', [5, 9]),('xx', [5, 10]),('xx', [5, 11]),
                        #('xx', [6, 7]),#('xx', [6, 8]),('xx', [6, 9]),('xx', [6, 10]),('xx', [6, 11]),
                        #('xx', [7, 8]),('xx', [7, 9]),('xx', [7, 10]),('xx', [7, 11]),
                        #('xx', [8, 9]),('xx', [8, 10]),('xx', [8, 11]),
                        #('xx', [9, 10]),('xx', [9, 11]),
                        #('xx', [10, 11]),
        #                ('yy', [0, 1]),('yy', [0, 2]),('yy', [0, 3]),#('yy', [0, 4]),('yy', [0, 5]),('yy', [0, 6]),('yy', [0, 7]),#('yy', [0, 8]),('yy', [0, 9]),('yy', [0, 10]),('yy', [0, 11]),
        #                ('yy', [1, 2]),('yy', [1, 3]),#('yy', [1, 4]),('yy', [1, 5]),('yy', [1, 6]),('yy', [1, 7]),#('yy', [1, 8]),('yy', [1, 9]),('yy', [1, 10]),('yy', [1, 11]), 
        #                ('yy', [2, 3]),#('yy', [2, 4]),('yy', [2, 5]),('yy', [2, 6]),('yy', [2, 7]),#('yy', [2, 8]),('yy', [2, 9]),('yy', [2, 10]),('yy', [2, 11]),
                        #('yy', [3, 4]),('yy', [3, 5]),('yy', [3, 6]),('yy', [3, 7]),#('yy', [3, 8]),('yy', [3, 9]),('yy', [3, 10]),('yy', [3, 11]),
                        #('yy', [4, 5]),('yy', [4, 6]),('yy', [4, 7]),#('yy', [4, 8]),('yy', [4, 9]),('yy', [4, 10]),('yy', [4, 11]),
                        #('yy', [5, 6]),('yy', [5, 7]),#('yy', [5, 8]),('yy', [5, 9]),('yy', [5, 10]),('yy', [5, 11]),
                        #('yy', [6, 7]),#('yy', [6, 8]),('yy', [6, 9]),('yy', [6, 10]),('yy', [6, 11]),
                        #('yy', [7, 8]),('yy', [7, 9]),('yy', [7, 10]),('yy', [7, 11]),
                        #('yy', [8, 9]),('yy', [8, 10]),('yy', [8, 11]),
                        #('yy', [9, 10]),('yy', [9, 11]),
                        #('yy', [10, 11]),
        #                ('zz', [0, 1]),('zz', [0, 2]),('zz', [0, 3]),#('zz', [0, 4]),('zz', [0, 5]),('zz', [0, 6]),('zz', [0, 7]),#('zz', [0, 8]),('zz', [0, 9]),('zz', [0, 10]),('zz', [0, 11]),
        #                ('zz', [1, 2]),('zz', [1, 3]),#('zz', [1, 4]),('zz', [1, 5]),('zz', [1, 6]),('zz', [1, 7]),#('zz', [1, 8]),('zz', [1, 9]),('zz', [1, 10]),('zz', [1, 11]), 
        #                ('zz', [2, 3])]#('zz', [2, 4]),('zz', [2, 5]),('zz', [2, 6]),('zz', [2, 7]),#('zz', [2, 8]),('zz', [2, 9]),('zz', [2, 10]),('zz', [2, 11]),
                        #('zz', [3, 4]),('zz', [3, 5]),('zz', [3, 6]),('zz', [3, 7]),#('zz', [3, 8]),('zz', [3, 9]),('zz', [3, 10]),('zz', [3, 11]),
                        #('zz', [4, 5]),('zz', [4, 6]),('zz', [4, 7]),#('zz', [4, 8]),('zz', [4, 9]),('zz', [4, 10]),('zz', [4, 11]),
                        #('zz', [5, 6]),('zz', [5, 7]),#('zz', [5, 8]),('zz', [5, 9]),('zz', [5, 10]),('zz', [5, 11]),
                        #('zz', [6, 7])]#('zz', [6, 8]),('zz', [6, 9]),('zz', [6, 10]),('zz', [6, 11]),
                        #('zz', [7, 8]),('zz', [7, 9]),('zz', [7, 10]),('zz', [7, 11]),
                        #('zz', [8, 9]),('zz', [8, 10]),('zz', [8, 11]),
                        #('zz', [9, 10]),('zz', [9, 11]),
                        #('zz', [10, 11])]
        #print(operator_pool)
        # Define the array where the gradients will be stored
        new_gradients = np.zeros(len(operator_pool))

        # Loop over all the trial operators in the pool
        for i, operator in enumerate(operator_pool):

            # Returns a trial circuit without modifying self.ansatz.circuit
            trial_circuit_ansatz = self.ansatz.add_ops([operator], update_count=False)

            # Update the variational parameters and the shift vector by appending zero(s)
            # (because each trial operator has a single parameter, only one zero is appended, i.e. num_new_params=1)
            num_new_params = trial_circuit_ansatz.num_parameters - len(var_parameters)
            new_var_parameters = np.append(var_parameters, np.full(num_new_params, 0))
            new_shift_init_guess = np.append(shift_init_guess, np.full(num_new_params, 0))

            # Get the gradient function for this trial circuit (only for the newly added parameter(s), here we assume
            # that there is only one new parameter)
            _,_, gradient_last_comp = self.get_loss(hamiltonian, trial_circuit_ansatz, new_var_parameters, time_step)
            
            # The only entries in the gradient vector that change are the ones corresponding to the
            # added variational parameters (again, here there is only one new parameter)
            new_gradients[i] = np.abs(gradient_last_comp(new_shift_init_guess))

        # Make a dictionary to pair each trial operator to its gradient
        operator_pool_keys = [op + '_' + str(pos) for op, pos in operator_pool]	
        ops_and_grad = dict(zip(operator_pool_keys, new_gradients))	
        ops_and_grad_cp = dict(ops_and_grad)  # make a copy of the dictionary
        
        # Initialize a list to store the operators to add to the current parametrized circuit
        operators = []

        # Loop until all the qubit indices have been exhausted (i.e. until the pool is empty)
        while len(ops_and_grad_cp) > 0:
            # Select the operator in the current pool that maximizes the gradient
            op_max_grad = max(ops_and_grad_cp, key=ops_and_grad_cp.get).split('_')
            op_max_grad[1] = eval(op_max_grad[1])
            operators.append(tuple(op_max_grad))
            
            # Remove all operators in the pool that act on qubit indices already acted on by op_max_grad
            [ops_and_grad_cp.pop(op) for op in ops_and_grad_cp.copy().keys() if set(eval(op.split('_')[1])).isdisjoint(op_max_grad[1]) == False]

        # Sort the operators according to the qubit(s) on which they act and add them to the circuit
        sorted_idx = np.argsort([op[1][0] for op in operators])
        operators = [operators[i] for i in sorted_idx]
        new_circuit_ansatz = self.ansatz.add_ops(operators, update_count=True)
        print(f'Operators added to the circuit: {operators}')
        #print(f'New circuit ansatz: \n{new_circuit_ansatz}')                      #打印量子电路
        
        #以下几行，是我自己加的代码
        decomposed_circ = self.ansatz.circuit.decompose(reps=3)
        depth = decomposed_circ.depth()
        #cnots = decomposed_circ.count_ops()["cx"]           没有CNOT时，会报错
        cnots = decomposed_circ.count_ops().get('cx', 0) #已经很久了，还没有CNOT，感觉不对!原来是大小写
        #print(f'currnet circuit； \n{decomposed_circ}')
        print(f'Depth:{depth}')
        print(f'CNOTS:{cnots}')
        print(decomposed_circ.count_ops())
        #以上几行，是我自己增加的代码

        # # Add a Trotter step to the parametrized circuit
        # self.ansatz.add_a_trotter_step()
        # print(f"A Trotter step has been added to the circuit ansatz.")
        # print(self.ansatz.circuit)
        # new_circuit_ansatz = self.ansatz.circuit

        self.ansatz.circuit = new_circuit_ansatz  # Overwrite the current circuit
        return new_circuit_ansatz

    def minimization_routine(
            self,
            var_parameters,
            shift_init_guess,
            hamiltonian,
            circuit_ansatz,
            time_step
    ):
        """ Find the small parameters dw* that minimize the infidelity.

        Args:
            var_parameters (np.array): current variational parameters.
            shift_init_guess (np.array): guess for the classical optimization
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            time_step (float): small time step.

        Returns:
            approximate minimum of the infidelity
        """

        print('-------------------------------')
        print('Start optimizing the infidelity')

        # Get the infidelity and gradient functions
        infidelity, gradient,_ = self.get_loss(hamiltonian, circuit_ansatz, var_parameters, time_step)

        updated_shift = np.copy(shift_init_guess)
        fid = 1 - infidelity(updated_shift)
        grad_norm = max(np.abs(gradient(updated_shift)))  # infinite norm

        # Initialize the 1st and 2nd moment vectors of Adam
        m = np.zeros(len(var_parameters))
        v = np.zeros(len(var_parameters))

        count = 0  # to count the number of iterations of Adam
        while grad_norm > self.gradient_tolerance and count < self.maxiter:  
            count += 1

            grad = gradient(updated_shift)
            grad_norm = max(np.abs(grad))

            updated_shift, m, v = adam_step(updated_shift, count, m, v, grad)

            # Get the optimized fidelity
            fid = 1 - infidelity(updated_shift)
            
            # Print statement
            if count % 20 == 0:
                print(f'Iteration: {count}   Fidelity: {fid}    Gradient: {grad_norm}')

        return updated_shift, fid

    def one_time_step(
            self,
            hamiltonian,
            circuit_ansatz,
            var_parameters,
            time_step,
            shift_init_guess,
    ):
        """ Advance the time evolution by one time step dt in time.

        Args:
            hamiltonian (PauliSumOp): Hamiltonian used for the time evolution.
            circuit_ansatz (QuantumCircuit): parametrized/variational circuit.
            var_parameters (np.array): current variational parameters.
            time_step (float): small time step.
            shift_init_guess (np.array): guess for the classical optimization

        Returns:
            A tuple with the new variational parameters and the fidelity
        """

        if circuit_ansatz.num_parameters == 0:
            # When there are no parameters in the circuit (at t=0 when no ansatz is provided)  
            updated_shift = np.copy(shift_init_guess)
            fidelity = 0  # set the fidelity to any arbitrary value below "self.fidelity_tolerance"
        else: 
            # For t>0 (when there is at least one parameter in the circuit).
            # Find the current best small shift parameters dw* that minimize the infidelity
            updated_shift, fidelity = self.minimization_routine(
                var_parameters,
                shift_init_guess,
                hamiltonian,
                circuit_ansatz,
                time_step
            )

        # Adaptive part (comment out the while loop if adaptive part should not be used):
        updated_var_parameters = np.copy(var_parameters)
        while fidelity < self.fidelity_tolerance:
            
            # Perform and adaptive step
            new_circuit_ansatz = self.adaptive_step(hamiltonian, updated_var_parameters, time_step, updated_shift)
    
            # Update the parameters w and the small shift parameters dw by appending zero(s)
            total_num_new_params = new_circuit_ansatz.num_parameters - len(var_parameters)
            num_new_params = new_circuit_ansatz.num_parameters - len(updated_var_parameters)
            updated_var_parameters = np.append(var_parameters, np.full(total_num_new_params, 0))
            updated_shift = np.append(updated_shift, np.full(num_new_params, 0))
    
            # Find the current best small shift parameters dw* that minimize the infidelity
            updated_shift, fidelity = self.minimization_routine(
                updated_var_parameters,
                updated_shift,
                hamiltonian,
                new_circuit_ansatz,
                time_step
            )

        return updated_var_parameters + updated_shift, fidelity

    def evolve(
            self,
            hamiltonian,
            num_time_steps,
            final_time,
            #filename      = 'data/trial_results.dat',
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

        start_time = time.time()
        #print(f'Hamiltonian:\n{hamiltonian(time=0)}')
        print(f'Initial circuit:\n{self.ansatz.circuit}')

        #以下几行是我自己添加
        decomposed_circ = self.ansatz.circuit.decompose(reps=3)
        #print(f'current circuit: \n{decomposed_circ}')
        depth = decomposed_circ.depth()
        print(f'Depth    :{depth}')
        print(decomposed_circ.count_ops())
        cnots = decomposed_circ.count_ops().get('cx', 0)
        print(f'CNOTS:{cnots}')
        #以上几行是我自己添加        













        # Get the function to evaluate the expectation value of the observables of interest given
        # the current parametrized circuit
        evaluate_observables = get_observable_evaluator(self.ansatz.circuit, observables, self.expectation, self.sampler)
                                  #这个函数在模块utils里面
        print(evaluate_observables)
        # Time related stuff
        time_step = final_time / num_time_steps
        times = np.linspace(0, final_time, num_time_steps + 1).tolist()  # +1 to include t=0
        
        # Initialize the data log at t=0
        data_log = {"times": times,
                    "parameters": [initial_parameters],
                   # "n0n4":0,
                    #"n0n2":1,
                    "Depth":[1],
                    "CNOTS":[0],
                    "parameter_number":[0],
                    "fidelity": [1],
                    "observables values": [evaluate_observables(initial_parameters)],
                    "evolved state": []
                    }
        print('below is the data_log at t=0')
        print(data_log)
        evolved_state = self.ansatz.circuit.bind_parameters(initial_parameters)
        data_log["evolved state"].append(evolved_state)

        # Loop over all time steps (to reach the final time)
        for tt in times[1:]:
            
            print(f"Time step: {tt:<10.3f}")
            ham = hamiltonian(time=tt)
            #ham = hamiltonian
            # Perform one time step
            next_parameters, fidelity = self.one_time_step(
                ham,
                self.ansatz.circuit,
                data_log["parameters"][-1],
                time_step,
                shift_init_guess
            )
            shift_init_guess = np.zeros(len(next_parameters))
            #以下几行是我自己添加
            
            # 统计所有操作
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
            #print(f'current circuit: \n{decomposed_circ}')
            depth = decomposed_circ.depth()
            print(f'Depth    :{depth}')
            print(decomposed_circ.count_ops())
            cnots = decomposed_circ.count_ops().get('cx', 0)
            print(f'CNOTS:{cnots}')
            #以上几行是我自己添加
            print(f"Fidelity: {fidelity}")
            print(f"Elapsed time since beginning: {time_convert(time.time() - start_time)}")
            print('==================================================')

            # Get the function to evaluate the expectation value of the observables of interest given
            # the current parametrized circuit
            evaluate_observables = get_observable_evaluator(self.ansatz.circuit, observables, self.expectation, self.sampler)

            # Compute and store the relevant quantities
            data_log["parameters"].append(next_parameters)
            
            z0 = [arr[0] for arr in data_log["observables values"]]
#            z0z1=[arr[1] for arr in data_log["observables values"]]
            #data_log["n0n4"].append(z0[len(z0)-1])
            #data_log["n0n2"].append(z0z1[len(z0z1)-1])
            
            data_log["Depth"].append(depth)
            data_log["CNOTS"].append(cnots)
            data_log["parameter_number"].append(len(next_parameters))
            data_log["fidelity"].append(fidelity)
            data_log["observables values"].append(evaluate_observables(next_parameters))
            evolved_state = self.ansatz.circuit.bind_parameters(next_parameters)
            data_log["evolved state"].append(evolved_state)
            print(data_log["Depth"])
            print(data_log["CNOTS"])
            print(data_log["observables values"])
            #z0, z0z1 = np.array([evaluate_observables]).T
            #print(z0)
            
            #print('below is n0n4')
            #for i in range(len(z0)):
            #    print(z0[i])
            #print('below is n0n2')
#           # for i in range(len(z0z1)):
 #          #     print(z0z1[i])
            #print('below is cnot sequence')
            #for i in range(len(data_log["CNOTS"])):
            #    print(data_log["CNOTS"][i])
            #print('below is depth sequence')
            #for i in range(len(data_log["Depth"])):
            #    print(data_log["Depth"][i])
        #print(f'last ansatz: \n{new_circuit_ansatz}')
        return data_log
