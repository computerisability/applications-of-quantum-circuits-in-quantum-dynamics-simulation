import json
import numpy as np
import pickle
from qiskit import Aer,transpile,QuantumCircuit ,qpy
from qiskit.utils import QuantumInstance
from qiskit.opflow import X, Z, PauliExpectation
from hamiltonian import *
from parametrized_circuit import * 
from dualQRTE_adam import DualQRTE
from qiskit.quantum_info import Statevector, partial_trace, entropy
#from functools import partial

n_qubits = 8
k = int(n_qubits/2)

tf = 2
dt = 0.05
times = np.linspace(0, tf, int(tf/dt) + 1)

a = 0.5    #x = 1,#mu = 2
g = 2
m = 1
ell = 0

backend = Aer.get_backend("statevector_simulator") 
instance = QuantumInstance(backend=backend, shots=1)

psi0 = QuantumCircuit(n_qubits)
fermion_idx = [0, 2, 4, 6 ]
psi0.x(fermion_idx)  # Initial state |10101010⟩
psi0.barrier()

# Convert the initial state to a Statevector
initial_state = Statevector.from_instruction(psi0)

hamiltonian = schwinger_hamiltonian(n_qubits=n_qubits, model_params=[a,g,m,ell])
hamiltonian = SparsePauliOp( hamiltonian.primitive.paulis, hamiltonian.primitive.coeffs )

#ansatz = AnsatzSchwinger(n_qubits, depth=2)     
ansatz = AnsatzSchwinger8(n_qubits, depth = 1)   

initial_parameters = np.zeros(ansatz.circuit.num_parameters)

O1 = 1.0/n_qubits *one_site_op(n_qubits,0,Z)
O2 = 1.0/n_qubits *one_site_op(n_qubits,0,Z)
O3 = 1.0 *one_site_op(n_qubits,0,Z)
O4 = 1.0/n_qubits *nearest_neigh_op(n_qubits, 0, [X, X])
O5 = 1.0/n_qubits *nearest_neigh_op(n_qubits, 0, [Y, Y])
for n in range(1,n_qubits):
    #O1 = O1 + 1.0/n_qubits *one_site_op(n_qubits,n,Z)
    O2 = O2 +(-1)**n/n_qubits *one_site_op(n_qubits,n,Z)
    O3 = O3 +(n_qubits-1-n)/(n_qubits-1) *one_site_op(n_qubits,n,Z)
for n in range(1,n_qubits-1):
    O4 = O4 +1.0/n_qubits *nearest_neigh_op(n_qubits, n, [X, X])
    O5 = O5 +1.0/n_qubits *nearest_neigh_op(n_qubits, n, [Y, Y])

HK = hk(n_qubits,a,g,m,ell)/n_qubits
HM = hm(n_qubits,a,g,m,ell)/n_qubits
HP = hp(n_qubits,a,g,m,ell)/n_qubits
H  = schwinger_hamiltonian(n_qubits=n_qubits, model_params=[a,g,m,ell])



observables = [O1,O2,O3,O4,O5,HM,HK,HP,H]




# Initialize the algorithm
dual_qrte = DualQRTE(
    ansatz,
    maxiter=1000,
    gradient_tolerance=1e-5,
    expectation= PauliExpectation(),
    quantum_instance=instance
)

# Evolve in time
data=dual_qrte.evolve(
        hamiltonian,
        num_time_steps=int(np.ceil(tf/dt)),
        final_time=tf,
        initial_parameters=initial_parameters,
        shift_init_guess=np.zeros(len(initial_parameters)),
        observables=observables
)


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


np.savez('schwinger_dual_adam_.npz', O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,
        entanglement_entropies = entanglement_entropies, loschmidt_echo = loschmidt_echo, 
         Hm=Hm, Hk=Hk, Hp=Hp, H=H,
         depth_local=depth_local , cnot_local=cnot_local, number_local = number_local
    )     

a = data["evolved state"]
with open('schwinger_dual_adam_.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
    qpy.dump(a, f)  







