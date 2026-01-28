import pickle
import numpy as np
#from pauli_function import *
from hamiltonian import *
from qutip import tensor, basis, mesolve, Qobj, entropy_vn, fidelity
import json



spins = 8                                                  #量子比特数量

dt = 0.05
n_steps = 40                                               #总的演化时间

tf = dt * n_steps
times = np.arange(0, tf+dt, dt)

h = hubbard_ladder_hamiltonian(size_x = 2, model_params=[1,0.8])
h = [Qobj(h.to_matrix())]


## Define the initial state

# |  spins-1 spins-2   ...    1    0 > 
# |     up    down     ...    up  down > 
#psi0_tensor = tensor([up, down]*(int(spins/2)))
#psi0 = Qobj(tensor(psi0_tensor).full())

#observables = [num_op(spins, 0) @ num_op(spins, 2),               #若8个比特，则将此处2，修改为4
#                num_op(spins, 0) @ num_op(spins, 1)]

up = basis(2,0)
down = basis(2,1)
psi0_tensor = tensor([down, up, down, up, up, down, up, down])         #修改初态
psi0 = Qobj(tensor(psi0_tensor).full())



observables = [Qobj(   (num_op(spins, 0) @ num_op(spins, 4) )  .to_matrix()),#若8个比特，则将此处2，修改为4
               Qobj(   (num_op(spins, 0) @ num_op(spins, 2)  ) .to_matrix())
               ]

#第一步：明确要测量那些  第二步：运行程序
### Master equation evolution
#
##output = mesolve(h, psi0, times, e_ops=observables, args={'w':1})

options = {"store_states": True}  # 启用保存态
output = mesolve(h, psi0, times, e_ops=observables, options=options)
output1 = mesolve(h, psi0, times)

n0n4, n0n2 = output.expect[0:2]                                                  #若为8个比特，则第一个n0n2为n0n4



psi_t = output.states
states_data = [state.full() for state in psi_t]  # 转为 NumPy 数组
#with open('hubbard_exact_states.pkl', 'wb') as f:                         #exact用pkl类文件是可以的
#    pickle.dump(psi_t, f)


#fig 4c:entanglement entropy
# half-cut entanglement entropy
entanglement = []
for psi in psi_t:
    psi0_tensor._data = psi._data 
    reduced_psi = psi0_tensor.ptrace(list(range(spins//2)))
    entanglement.append(entropy_vn(reduced_psi,2)) 
#
#print(f'the length of entanglement is:{len(entanglement)}')
#print(entanglement)

#np.savez('entanglement_exact.npz',  times=times, entanglement=entanglement)      #之前是这一句

#question:what is the purpose for format(spins)?

#fig 5:
loschmidt_echo = [-np.log(abs(psi0.overlap(psi))**2)/spins for psi in psi_t]
#kappa = [-np.log(abs(psi0.overlap(psi))**2)/(ma*spins) for psi in psi_t]

#np.savez('loschmidt_echo_exact.npz'.format(spins),  times=times, loschmidt_echo=loschmidt_echo)  之前这

#np.savez('mzero.npz'.format(spins), adia_times=adia_times, kappa=kappa)



#np.savez('hubbard_exact.npz', times=times, n0n4= n0n4, n0n2 = n0n2,           #若为8个比特，则还有n0n4
#         entanglement=entanglement,loschmidt_echo=loschmidt_echo)


########### Should not use the following code
#from hamiltonian import floquet_term 
#h_drive = 10* floquet_term(spins)
#coeff = lambda time, args: np.sin(args['w'] * time)
#h = [Qobj(h.to_matrix()),[Qobj(h_drive.to_matrix()), coeff]] 
#psi0 = np.load('gs.npy')
#print(psi0)
#psi0 = np.ones((2**spins, 1))/2**(spins/2)
#psi0 = Qobj(psi0.transpose())