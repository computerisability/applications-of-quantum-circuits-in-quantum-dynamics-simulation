import numpy as np
import pylab
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit.algorithms import TimeEvolutionProblem, VarQITE, VarQRTE, SciPyImaginaryEvolver, SciPyRealEvolver
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple, RealMcLachlanPrinciple
from qiskit.algorithms.gradients import ReverseEstimatorGradient, ReverseQGT, DerivativeType
from hamiltonian import *
from parametrized_circuit import * 

n_qubits = 8
tf = 2.0
dt = 0.05
#num_timesteps = int(tf/dt) + 1             0.05,0.025用这句没有问题，如果0.01用这句会报错
num_timesteps = int(tf/dt)                 #0.01用这句是对的

hamiltonian = xyz_hamiltonian(n_qubits=n_qubits, model_params=[0.25, 0.25, 0.25, 0, 0, -1])
hamiltonian = SparsePauliOp( hamiltonian.primitive.paulis, hamiltonian.primitive.coeffs )

#ansatz = AnsatzXYZFloquet(n_qubits, depth=3)   
ansatz = AnsatzXYZFloquet(n_qubits, depth = 1)   

print("电路结构:")
print(ansatz.circuit)

init_param_values = {}
for i in range(len(ansatz.circuit.parameters)):
    init_param_values[ansatz.circuit.parameters[i]] = 0 

init_state = Statevector(ansatz.circuit.assign_parameters(init_param_values))
print(init_state)

var_principle = RealMcLachlanPrinciple()

#aux_ops = one_site_op(n_qubits, 0, Z)
aux_ops = nearest_neigh_op(n_qubits, 0, [Z, Z])

aux_ops = [ SparsePauliOp.from_list([(str(aux_ops.primitive), aux_ops.coeff)]) ]

evolution_problem = TimeEvolutionProblem(hamiltonian, tf, aux_operators=aux_ops)
var_qrte = VarQRTE(ansatz.circuit, init_param_values, var_principle, Estimator(), num_timesteps=num_timesteps)
evolution_result = var_qrte.evolve(evolution_problem)

print("Complete QRTE")        

exp_val = np.array([ele[0][0] for ele in evolution_result.observables])
#np.save('exp_val.npz', exp_val)
np.savez('xyz_var_z0z1.npz', exp_val = exp_val)

print(exp_val)





