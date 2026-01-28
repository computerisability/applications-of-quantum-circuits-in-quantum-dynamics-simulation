import pickle
import numpy as np
#from pauli_function import *
from hamiltonian import *
from qutip import tensor, basis, mesolve, Qobj, entropy_vn, fidelity
import json


ell = 0.0
spins = 8                                                  #量子比特数量
a = 0.5
g = 2  # ga*ga = J/w, w = 1/(2a), J = g*g*a/2
m = 1  # 2*ma = m/w
#x  = 1/(ga*ga)
#mu = 2*ma/(ga*ga)
dt = 0.05
n_steps = 40                                               #总的演化时间

tf = dt * n_steps
times = np.arange(0, tf+dt, dt)

h = schwinger_hamiltonian(spins,a,g,m,ell)

hm = hm(spins,a,g,m,ell)
hk = hk(spins,a,g,m,ell)
hp = hp(spins,a,g,m,ell)







h = [Qobj(h.to_matrix())]
hm = Qobj(hm.to_matrix())
hk = Qobj(hk.to_matrix())
hp = Qobj(hp.to_matrix())

## Define the initial state
up = basis(2,0)
down = basis(2,1)
# |  spins-1 spins-2   ...    1    0 > 
# |     up    down     ...    up  down > 
psi0_tensor = tensor([up, down]*(int(spins/2)))
psi0 = Qobj(tensor(psi0_tensor).full())

adia_end = 10
adia_times = np.arange(0, adia_end, 0.01)
def hk_coeff(t,args):
    return t/adia_end

adia_h=[hm, [hk,hk_coeff]]  # hm + hk_coeff(t) * hk
#output = mesolve(adia_h, psi0, adia_times)
#psi0 = output.states[-1]


## The difference between two neighbouring states
#fidelity_results = []
#for i in range(len(output.states) - 1):
#    fid = fidelity(output.states[i], output.states[i + 1])
#    fidelity_results.append(fid)
#print(fidelity_results)

##
##### Prepare the observables to measure

#fig4b 
#0.5*02+0.5 is the :particle number density
O1 = 1.0 * one_site_op(spins, 0, Z)
O2 = 1.0/spins * one_site_op(spins, 0, Z)
O3 = 1.0/spins * one_site_op(spins, 0, Z)
O4 = 1.0/spins * nearest_neigh_op(spins, 0, [X, X])
O5 = 1.0/spins * nearest_neigh_op(spins, 0, [Y, Y])
for n in range(1,spins):
    O1 = O1 + 1.0/spins * one_site_op(spins, n, Z)
    O2 = O2 + (-1.0)**n /spins * one_site_op(spins, n, Z)
    O3 = O3 + (spins-1-n)/(spins-1) * one_site_op(spins, n, Z)
for n in range(1,spins-1):
    O4 = O4 + 1.0/spins * nearest_neigh_op(spins, n, [X, X])
    O5 = O5 + 1.0/spins * nearest_neigh_op(spins, n, [Y, Y])
##
O1 = Qobj(O1.to_matrix())
O2 = Qobj(O2.to_matrix())
O3 = Qobj(O3.to_matrix())
O4 = Qobj(O4.to_matrix())
O5 = Qobj(O5.to_matrix())



#fig6:local electric fields
#for below, should multiply 0.5



##以下代码，运行出来的效果，似乎不太好
##from qiskit.opflow import I
##identity = I^spins
##so = 0*identity
##O = []
##for n in range(0,spins-1):
##    a = ell + 0.5 * sum((-1)**k for k in range(0, n+1))
##    so = so + PauliOp(generate_pauli([],[n],spins), 1.0)
##    O.append(Qobj((a*identity + 0.5*so).to_matrix()))
##
##observables.extend(O)
#fig7: part of the hamiltonian
observables=[O1,O2,O3,O4,O5,hm/spins,hk/spins,hp/spins,h[0]/spins]

#第一步：明确要测量那些  第二步：运行程序









### Master equation evolution
#
##output = mesolve(h, psi0, times, e_ops=observables, args={'w':1})

options = {"store_states": True}  # 启用保存态
output = mesolve(h, psi0, times, e_ops=observables, options=options)
output1 = mesolve(h, psi0, times)
#
### Save the data in a file
#
o1,o2,o3,o4,o5,Hm, Hk, Hp, H= output.expect[0:9]
#np.savez('observables_exact.npz', times=times, O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,
#         Hm=Hm, Hk=Hk, Hp=Hp, H=H)

#缠绕熵、保真度，得先运行出态    第一步：明确要测量那些  第二步：运行程序 第三步：对结果的态进行处理
#目前已经运行出output，其中里面包含了态
#psi_t = output.states
#print(psi_t)
#np.savez('exact_states.npz'.format(spins),  times=times, psi_t=psi_t)



#psi_t = output1.states               #之前的是这一句


psi_t = output.states
states_data = [state.full() for state in psi_t]  # 转为 NumPy 数组
#with open('schwinger_exact_states.pkl', 'wb') as f:                         #exact用pkl类文件是可以的
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



#np.savez('schwinger_exact.npz', times=times, O1=o1,O2=o2,O3=o3,O4=o4,O5=o5,Hm=Hm, Hk=Hk, Hp=Hp, H=H,
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