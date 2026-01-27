
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
#import pyspark.serializers
#pyspark.serializers.cloudpickle = cloudpickle
from qiskit import Aer,transpile,QuantumCircuit ,qpy
from qiskit.utils import QuantumInstance
from qiskit.opflow import X, Z, PauliExpectation,ListOp
from hamiltonian import *
from parametrized_circuit import * 
from adaptive_pvqd import AdaptivePVQD
from functools import partial
from qiskit.quantum_info import Statevector, partial_trace, entropy

#times = np.arange(0.05, 0.35, 0.05).tolist()   #这句在保存数据那里起作用
#print(times)
# Define some relevant variables

n_qubits = 8                                            #此处需要与哈密顿量除以4对应
k = int(n_qubits/2)



tf = 4         
dt = 0.05  # time step
#times = np.arange(0, tf+dt, dt)

# Backend choice (statevector simulation)
backend = Aer.get_backend("statevector_simulator") 
instance = QuantumInstance(backend=backend, shots=1)

#以下为我增加的代码
# Define the initial state
psi0 = QuantumCircuit(n_qubits)
fermion_idx = [0, 2 ,4,6]                                              #此处需要与量子比特数量对应
psi0.x(fermion_idx)  # Initial state |10101010⟩
psi0.barrier()

# Convert the initial state to a Statevector
initial_state = Statevector.from_instruction(psi0)
#以上为我增加的代码

## Define the initial ansatz and parameters (if applicable)
ansatz = AnsatzLadderHubbard(n_qubits, depth=4)                    #原始代码此处为3，而adaptive这里其实是0
     
initial_parameters = np.zeros(ansatz.circuit.num_parameters)

hamiltonian = hubbard_ladder_hamiltonian(size_x=2, model_params=[1,0.8])    #SIZE-X乘以4需等于量子比特数量
print(hamiltonian)

observables = [num_op(n_qubits, 0) @ num_op(n_qubits, 4),               #若8个比特，则将此处2，修改为4
                num_op(n_qubits, 0) @ num_op(n_qubits, 2)]

# Initialize the algorithm
adaptive_pvqd = AdaptivePVQD(
    ansatz,
    maxiter=200,
    fidelity_tolerance=0.9999,
    gradient_tolerance=5e-5,
    expectation= PauliExpectation(),
    quantum_instance=instance,
)

# Evolve in time
data=adaptive_pvqd.evolve(
        hamiltonian,
        num_time_steps=int(np.ceil(tf/dt)),
        final_time=tf,
        #filename      = 'data/trial_results.dat',
        initial_parameters=initial_parameters,
        shift_init_guess=np.zeros(len(initial_parameters)),
        observables=observables
)
#print('below is the whole data')
#print(type(result))
#print(data)
print(data['evolved state'])
print(f"the type of evolved state is:{type(data['evolved state'])}")

evolved_states = data['evolved state']
# 将 psi0 转换为状态向量
tqc0 = transpile(psi0, backend)
job0 = backend.run(tqc0)
result0 = job0.result()
psi0_statevector = result0.get_statevector()

# 计算 每个 evolved_state 的纠缠熵
entanglement_entropies = []
for evolved_state in evolved_states:
    # 计算
    statevector = Statevector.from_instruction(evolved_state)
    reduced_density_matrix = partial_trace(statevector, list(range(k)))
    entanglement_entropy = entropy(reduced_density_matrix)
    entanglement_entropies.append(entanglement_entropy)

# 计算 psi0 与每个 evolved_state 的重叠度
overlaps = []
for evolved_state in evolved_states:
    tqc = transpile(evolved_state, backend)
    job = backend.run(tqc)
    result = job.result()
    evolved_statevector = result.get_statevector()

    # 计算重叠度
    sv1 = Statevector(psi0_statevector)
    sv2 = Statevector(evolved_statevector)
    overlap = abs(sv1.inner(sv2)) ** 2
    overlaps.append(overlap)


#fig4b +fig6 +fig7a
observables_values = []
for obs in data['observables values']:
    observables_values.extend(obs.tolist())
n0n4 = observables_values[0::2]                     #若为4个比特，则此处的n0n4，也是n0n2
n0n2 = observables_values[1::2]



loschmidt_echo = -1.0/n_qubits * np.log(overlaps)


depth_pvqd = data['Depth']
cnot_pvqd = data['CNOTS']
number_pvqd = data['parameter_number']


np.savez('hubbard_pvqd.npz', n0n4 = n0n4, n0n2 = n0n2,    #若为8个比特，则此处还有n0n4
        entanglement_entropies = entanglement_entropies, loschmidt_echo = loschmidt_echo, 
         depth_pvqd =depth_pvqd , cnot_pvqd = cnot_pvqd, number_pvqd = number_pvqd
    )     

a = data["evolved state"]
with open('hubbard_pvqd.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
    qpy.dump(a, f)  
