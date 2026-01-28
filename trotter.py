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
    
    # # Get the time-independent terms of the (XYZ + drive term) model and make the circuit compact
    ham = hamiltonian(time=0)
    xx_yy_zz_terms = ham[0:3*(n_qubits-1)]
    xx_yy_zz_terms_split = [xx_yy_zz_terms[i:i+3] for i in range(0, len(xx_yy_zz_terms), 3)]
    xx_yy_zz_terms_compact = xx_yy_zz_terms_split[0::2] + xx_yy_zz_terms_split[1::2]
    
    # Loop over the number of Trotter steps
    for ts in range(trotter_steps):
        
        # # Time-independent terms (same terms and same coefficients added for each Trotter step)
        for term in xx_yy_zz_terms_compact:
            qc = qc.compose(PauliEvolutionGate(operator=term, time=dt), range(n_qubits))

        # # Time-dependent terms (same terms but different coefficients added for each Trotter step)
        ham = hamiltonian(time=ts*dt)
        if len(ham) != 3*(n_qubits-1):  # i.e. if final_time > 0:
            for term in [ham[3*(n_qubits-1):]]:
                qc = qc.compose(PauliEvolutionGate(operator=term, time=dt,synthesis=SuzukiTrotter(reps=1)), range(n_qubits))
        
        
        #qc = qc.compose(PauliEvolutionGate(operator=hamiltonian, time=dt), range(n_qubits))
        qc.barrier()
    return qc
# Statevector simulations:
backend = Aer.get_backend("statevector_simulator")
sampler = CircuitSampler(backend)
expectation = PauliExpectation() 
shots = 0



n_qubits = 8                                           #量子比特数量
k = 4                                                     #计算一半纠缠熵

tf = 0.15
times = np.arange(0.05, 0.2, 0.05).tolist()

dt = 0.05  # time step
#times = np.arange(0, tf+dt, dt)  # range of times for the simulation
#times = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3 ]

#times = np.linspace(0, tf, int(tf/dt) + 2)             #加2，确实还是很难以理解，做出的图也奇怪！！
print(times)

# times = times[0::4]
# dt = times[1]-times[0]
## Define the initial state

model_params = [0.25, 0.25, 0.25, -1]

psi0 = QuantumCircuit(n_qubits)


fermion_idx = list(range(n_qubits))[1::2]                                      
psi0.x(fermion_idx)
print(f'the type od psi0 is:{type(psi0)}')
#print(psi0)
psi0.barrier()
print(psi0)
# 将初始态转换为状态向量
initial_state = Statevector.from_instruction(psi0)




observables = ListOp([one_site_op(n_qubits, 0, X),
            one_site_op(n_qubits, 0, Z),
            nearest_neigh_op(n_qubits, 0, [X, X]),
            nearest_neigh_op(n_qubits, 0, [Z, Z])])



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
    #n = int(t/dt)      #square
    n = int(t/dt) * int(t/dt)
    #n = 10
    print(f'Number of Trotter steps: {n}')    
    hamiltonian = partial(xyz_floquet_hamiltonian, n_qubits=n_qubits, model_params=model_params)
    # Prepare the Hamiltonian at the given time t
    qc_trot = trotterized_hamiltonian(                #这个函数至关重要
        hamiltonian = hamiltonian,
        trotter_steps = n,
        final_time = t
    )
    print(qc_trot)
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
#print(f'Final circuit:\n{decomposed_circ}')
#print(qc_trot)
print(f'Depth: {depth}')
print(f'CNOTS: {cnots}')
# 输出重叠度结果
print("Overlap values:", overlap_values)


print("lambda values:", overlap_values)
print("times:", times)
x0, z0, x0x1, z0z1 = np.array(obs_values).T                           
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

'''
np.savez('driven_square.npz', times=times, x0 = x0, z0 = z0,    x0x1 = x0x1, z0z1 = z0z1,
         entanglement_entropies=entanglement_entropies, loschmidt_echo=loschmidt_echo
         )

with open('driven_square.qpy', 'wb') as f:                              #量子电路算出来，需要保存为qpy文件
#    pickle.dump(evolved_states, f)
    qpy.dump(evolved_states, f)
'''