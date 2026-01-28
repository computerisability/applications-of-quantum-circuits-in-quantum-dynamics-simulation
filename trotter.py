import pickle
from hamiltonian import *
from qiskit import QuantumCircuit, Aer ,qpy
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import StateFn, CircuitSampler, ListOp, PauliExpectation
from qiskit.primitives import Estimator
import numpy as np
from parametrized_circuit import *         #感觉奇怪！难道这一句不需要吗！！！！！
from functools import partial
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import Statevector, partial_trace, entropy
from qiskit.synthesis import SuzukiTrotter, LieTrotter

def trotterized_hamiltonian(hamiltonian, trotter_steps, final_time):
    '''
    Circuit implementing Trotterization of the time evolutiom operator for a given Hamiltonian.
    This is a non-targeted (i.e. not resource efficient) way of implementing Trotterization.
        
    Args:
        hamiltonian (SumPauliOp): Hamiltonian to Trotterize
        trotter_steps (int): number of trotter steps to implement
        final_time (float): total simulation time
        
    Returns:
        QuantumCircuit implementing the Trotterization of the time evolutiom operator
    '''

    dt = final_time/trotter_steps
    qc = QuantumCircuit(n_qubits) 
    # Loop over the number of Trotter steps
    for ts in range(trotter_steps):
        qc = qc.compose(PauliEvolutionGate(operator = hamiltonian, time = dt, synthesis = LieTrotter(reps = 1)), range(n_qubits))
        qc.barrier()
    return qc
# Statevector simulations:
backend = Aer.get_backend("statevector_simulator")
sampler = CircuitSampler(backend)
expectation = PauliExpectation() 
shots = 0

# Fix some global variables
a = 0.5
g = 2
m = 1.0
#x  = 1/(ga*ga)
#mu  = 2*ma/(ga*ga)
ell     = 0

n_qubits = 8                                           #量子比特数量
k = 4                                                     #计算一半纠缠熵

tf = 0.2
times = np.arange(0.05, 0.25, 0.05).tolist()

dt = 0.05  # time step
#times = np.arange(0, tf+dt, dt)  # range of times for the simulation
#times = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3 ]

#times = np.linspace(0, tf, int(tf/dt) + 2)             #加2，确实还是很难以理解，做出的图也奇怪！！
print(times)

# times = times[0::4]
# dt = times[1]-times[0]
## Define the initial state
psi0 = QuantumCircuit(n_qubits)


fermion_idx = [0, 2, 4, 6]                                     #计算一半纠缠熵  
psi0.x(fermion_idx)
print(f'the type od psi0 is:{type(psi0)}')
#print(psi0)
psi0.barrier()
print(psi0)
# 将初始态转换为状态向量
initial_state = Statevector.from_instruction(psi0)

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



observables = ListOp([O1,O2,O3,O4,O5,HM,HK,HP,H])


#fig4c
entanglement_entropies = []

#fig5
# 存储每个时刻的重叠度
overlap_values = []


# Simulate the time evolution of psi0 by looping over multiple times:
obs_values = []
evolved_states = []
for t in times:
    print(f'Time: {t}')
    n = int(t/dt)
    #n = 5
    print(f'Number of Trotter steps: {n}')    
    hamiltonian   = schwinger_hamiltonian(n_qubits,a,g,m,ell)  
    # Prepare the Hamiltonian at the given time t
    qc_trot = trotterized_hamiltonian(                #这个函数至关重要
        hamiltonian=hamiltonian,
        trotter_steps=n,
        final_time=t
    )

    # Apply the trotterized Hamiltonian on the initial condition and perform measurements on the time evolved state
    evolved_circ_state = psi0.compose(qc_trot) 
    
    #下面这些代码，是我添加的
    # 将演化后的态转换为状态向量
    statevector = Statevector.from_instruction(evolved_circ_state)

    # 计算重叠度
    overlap = np.abs(initial_state.inner(statevector)) ** 2
    overlap_values.append(overlap)

    print(f'Overlap at t={t}: {overlap}')
    #上面几行代码，是我增加的关于保真空度的代码
    #下面几行代码，是我增加的关于纠缠熵的代码
    reduced_density_matrix = partial_trace(statevector, list(range(k)))  # 对第一个量子比特进行部分迹
    
    # 计算纠缠熵
    entanglement_entropy = entropy(reduced_density_matrix)
    entanglement_entropies.append(entanglement_entropy)
    #上面几行代码，是我增加的关于纠缠熵的代码
    
    evolved_states.append(evolved_circ_state)
    
    expectation_values = StateFn(observables, is_measurement=True) @ StateFn(evolved_circ_state)
    expectation_values = expectation.convert(expectation_values)
    sampled_op = sampler.convert(expectation_values)

    if shots > 1:
        variance = expectation.compute_variance(sampled_op)
        error = np.real(np.sqrt(np.array(variance) / shots))
        obs_values.append([np.real(np.array(sampled_op.eval())), error])

    else:
        obs_values.append(np.real(np.array(sampled_op.eval())))
       
    
    #ground_state_energy = hamiltonian.expect(evolved_circ_state)
    #print(ground_state_energy)

# Get the depth and the number of CNOTs in the Trotter circuit (to be used when plotting the results)
decomposed_circ = qc_trot.decompose(reps=3)
depth = decomposed_circ.depth()
cnots = decomposed_circ.count_ops()["cx"]
#print(f'Final circuit:\n{decomposed_circ}')              #这是分解后的电路
print(qc_trot)
print(f'Depth: {depth}')
print(f'CNOTS: {cnots}')
# 输出重叠度结果
print("Overlap values:", overlap_values)


print("lambda values:", overlap_values)
print("times:", times)
o1,o2,o3,o4,o5,Hm,Hk,Hp,H = np.array(obs_values).T
#print(hamiltonian)

#fig4b  +fig6

#np.savez('observables_trotter.npz', times=times, O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,
#         Hm=Hm, Hk=Hk, Hp=Hp, H=H
#    )

#fig4c
#np.savetxt('trotter_B_n=8.dat', overlap_values, fmt='%.10f')

#np.savez('entanglement_trotter.npz',  times=times, entanglement_entropies=entanglement_entropies)

#fig5
loschmidt_echo = -1.0/n_qubits * np.log(overlap_values)

#np.savez('loschmidt_echo_trotter.npz',  times=times, loschmidt_echo=loschmidt_echo)


#np.savez('schwinger_trotter.npz', times=times, O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,
#         Hm=Hm, Hk=Hk, Hp=Hp, H=H,
#         entanglement_entropies=entanglement_entropies, loschmidt_echo=loschmidt_echo
#         )

#with open('schwinger_trotter.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
#    qpy.dump(evolved_states, f)
