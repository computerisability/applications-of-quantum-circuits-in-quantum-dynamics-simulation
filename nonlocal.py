
import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
#import pyspark.serializers
#pyspark.serializers.cloudpickle = cloudpickle
from qiskit import Aer,transpile,QuantumCircuit, qpy
from qiskit.utils import QuantumInstance
from qiskit.opflow import X, Z, PauliExpectation,ListOp
from hamiltonian import *
from parametrized_circuit import * 
from adaptive_pvqd_nonlocal import AdaptivePVQD
from functools import partial
from qiskit.quantum_info import Statevector, partial_trace, entropy

#times = np.arange(0.05, 0.35, 0.05).tolist()   #这句在保存数据那里起作用
#print(times)
# Define some relevant variables
n_qubits = 4
k = int(n_qubits/2)



a = 0.5
g = 2
m = 1.0
#x = 1
#mu = 2
ell = 0

tf = 2         
dt = 0.05  # time step
#times = np.arange(0, tf+dt, dt)

# Backend choice (statevector simulation)
backend = Aer.get_backend("statevector_simulator") 
instance = QuantumInstance(backend=backend, shots=1)

#以下为我增加的代码
# Define the initial state
psi0 = QuantumCircuit(n_qubits)
fermion_idx = [0, 2 ]
psi0.x(fermion_idx)  # Initial state |10101010⟩
psi0.barrier()

# Convert the initial state to a Statevector
initial_state = Statevector.from_instruction(psi0)
#以上为我增加的代码

## Define a (possibly time-dependent) Hamiltonian
#hamiltonian = partial(hubbard_ladder_hamiltonian, size_x=3,model_params=[1,0.8])
#hamiltonian = partial(xyz_floquet_hamiltonian, n_qubits=n_qubits, model_params=[1,0.8,0.6,0])


## Define the initial ansatz and parameters (if applicable)
ansatz = AnsatzLadderHubbard(n_qubits, depth=0)
#ansatz= AnsatzSchwinger(n_qubits,depth=0)
#ansatz = AnsatzXYZFloquet(n_qubits, depth=0)        #原始代码此处为3，而adaptive这里其实是0
initial_parameters = np.zeros(ansatz.circuit.num_parameters)

# List of observables to measure at each time t of the simulation
#fig4b:
O1 = 1.0/n_qubits *one_site_op(n_qubits,0,Z)
O2 = 1.0/n_qubits *one_site_op(n_qubits,0,Z)
O3 = 1.0 *one_site_op(n_qubits,0,Z)
O4 = 1.0/n_qubits *nearest_neigh_op(n_qubits, 0, [X, X])
O5 = 1.0/n_qubits *nearest_neigh_op(n_qubits, 0, [Y, Y])
for n in range(1,n_qubits):
    O1 = O1 + 1.0/n_qubits *one_site_op(n_qubits,n,Z)
    O2 = O2 +(-1)**n/n_qubits *one_site_op(n_qubits,n,Z)
    O3 = O3 +(n_qubits-1-n)/(n_qubits-1) *one_site_op(n_qubits,n,Z)
for n in range(1,n_qubits-1):
    O4 = O4 +1.0/n_qubits *nearest_neigh_op(n_qubits, n, [X, X])
    O5 = O5 +1.0/n_qubits *nearest_neigh_op(n_qubits, n, [Y, Y])
    
#fig6
#L0 = one_site_op(n_qubits,0,Z)
#L1 = one_site_op(n_qubits,1,Z) + L0
#L2 = one_site_op(n_qubits,2,Z) + L1
#L3 = one_site_op(n_qubits,3,Z) + L2
#L4 = one_site_op(n_qubits,4,Z) + L3
#L5 = one_site_op(n_qubits,5,Z) + L4
#L6 = one_site_op(n_qubits,6,Z) + L5    


#fig7
HK = hk(n_qubits,a,g,m,ell)/n_qubits
HM = hm(n_qubits,a,g,m,ell)/n_qubits
HP = hp(n_qubits,a,g,m,ell)/n_qubits
H  = schwinger_hamiltonian(n_qubits,a,g,m,ell)



observables = [O1,O2,O3,O4,O5,HM,HK,HP,H]

hamiltonian = schwinger_hamiltonian(n_qubits,a,g,m,ell)
print(hamiltonian)





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
o1 = observables_values[0::9]
o2 = observables_values[1::9]
o3 = observables_values[2::9]
o4 = observables_values[3::9]
o5 = observables_values[4::9]
Hm = observables_values[5::9]
Hk = observables_values[6::9]
Hp = observables_values[7::9]
H = observables_values[8::9]


loschmidt_echo = -1.0/n_qubits * np.log(overlaps)


depth_local = data['Depth']
cnot_local = data['CNOTS']
number_local = data['parameter_number']


np.savez('schwinger_nonlocal.npz', O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,
        entanglement_entropies = entanglement_entropies, loschmidt_echo = loschmidt_echo, 
         Hm=Hm, Hk=Hk, Hp=Hp, H=H,
         depth_local=depth_local , cnot_local=cnot_local, number_local = number_local
    )     

a = data["evolved state"]
with open('schwinger_nonlocal.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
    qpy.dump(a, f)  
