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
from functools import partial

n_qubits = 8
k = int(n_qubits/2)

tf = 2
dt = 0.05
times = np.linspace(0, tf, int(tf/dt) + 1)


backend = Aer.get_backend("statevector_simulator") 
instance = QuantumInstance(backend=backend, shots=1)

psi0 = QuantumCircuit(n_qubits)
fermion_idx = [0, 2, 4, 6 ]
psi0.x(fermion_idx)  # Initial state |10101010⟩
psi0.barrier()

# Convert the initial state to a Statevector
initial_state = Statevector.from_instruction(psi0)

hamiltonian = xyz_hamiltonian(n_qubits=n_qubits, model_params=[0.25, 0.25, 0.25, 0, 0, -1])
hamiltonian = SparsePauliOp( hamiltonian.primitive.paulis, hamiltonian.primitive.coeffs )

#ansatz = AnsatzSchwinger(n_qubits, depth=2)     
ansatz = AnsatzXYZFloquet(n_qubits, depth = 1)   

initial_parameters = np.zeros(ansatz.circuit.num_parameters)

observables=[one_site_op(n_qubits, 0, X),
            one_site_op(n_qubits, 0, Z),
            nearest_neigh_op(n_qubits, 0, [X, X]),
            nearest_neigh_op(n_qubits, 0, [Z, Z])]




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
z0 = observables_values[1::4]                                    #若为8个比特，则这里还有
z0z1 = observables_values[3::4]  


loschmidt_echo = -1.0/n_qubits * np.log(overlaps)


depth_local = data['Depth']
cnot_local = data['CNOTS']
number_local = data['parameter_number']


np.savez('heisenberg_dual_adam_1.npz', z0 = z0, z0z1 = z0z1,
        entanglement_entropies = entanglement_entropies, loschmidt_echo = loschmidt_echo, 
         depth_local=depth_local , cnot_local=cnot_local, number_local = number_local
    )     

a = data["evolved state"]
with open('heisenberg_dual_adam_1.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
    qpy.dump(a, f)  







