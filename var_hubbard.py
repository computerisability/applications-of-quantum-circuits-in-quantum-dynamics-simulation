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

a = 0.5 
g = 2
m = 1
ell = 0.0

hamiltonian = hubbard_ladder_hamiltonian(size_x = 2, model_params = [1, 0.8])
hamiltonian = SparsePauliOp( hamiltonian.primitive.paulis, hamiltonian.primitive.coeffs )

#ansatz = AnsatzXYZFloquet(n_qubits, depth=3)   
ansatz = AnsatzLadderHubbard(n_qubits, depth = 3)   

print("电路结构:")
print(ansatz.circuit)

init_param_values = {}
for i in range(len(ansatz.circuit.parameters)):
    init_param_values[ansatz.circuit.parameters[i]] = 0 

init_state = Statevector(ansatz.circuit.assign_parameters(init_param_values))
print(init_state)

var_principle = RealMcLachlanPrinciple()

def build_n0n4_sparse(n_qubits):
    """直接构建O2的SparsePauliOp"""
    pauli_labels = []
    coeffs = []
    
    #for n in range(n_qubits):
        # 创建Pauli字符串，例如对于4量子比特，位置0: "IIIZ", 位置1: "IIZI", 位置2: "IZII", 位置3: "ZIII"
    pauli_str = ['I'] * n_qubits                     #Z0
    pauli_str[0] = 'Z'  # 在第n个位置放置Z
    pauli_label = ''.join(pauli_str)
    coefficient = -0.25
    pauli_labels.append(pauli_label)
    coeffs.append(coefficient)
    
    pauli_str = ['I'] * n_qubits                     #Z4
    pauli_str[4] = 'Z'  # 在第n个位置放置Z
    pauli_label = ''.join(pauli_str)
    coefficient = -0.25
    pauli_labels.append(pauli_label)
    coeffs.append(coefficient)
    
    pauli_str = ['I'] * n_qubits                     #Z0Z4
    pauli_str[4] = 'Z'  # 在第n个位置放置Z
    pauli_str[0] = 'Z'  # 在第n个位置放置Z
    pauli_label = ''.join(pauli_str)
    coefficient = 0.25
    pauli_labels.append(pauli_label)
    coeffs.append(coefficient)
    
    pauli_str = ['I'] * n_qubits                     #I
    pauli_label = ''.join(pauli_str)
    coefficient = 0.25
    pauli_labels.append(pauli_label)
    coeffs.append(coefficient)
    
    
    
    return SparsePauliOp(pauli_labels, coeffs)

# 构建O2算符（使用方法1）
O2_sparse = build_n0n4_sparse(n_qubits)
print(f"O2_sparse: {O2_sparse}")

# 设置aux_ops为O2算符
aux_ops = [O2_sparse]

evolution_problem = TimeEvolutionProblem(hamiltonian, tf, aux_operators=aux_ops)
var_qrte = VarQRTE(ansatz.circuit, init_param_values, var_principle, Estimator(), num_timesteps=num_timesteps)
evolution_result = var_qrte.evolve(evolution_problem)

print("Complete QRTE")        

exp_val = np.array([ele[0][0] for ele in evolution_result.observables])
#np.save('exp_val.npz', exp_val)
np.savez('hubbard_var.npz', exp_val = exp_val)
print(exp_val)





















