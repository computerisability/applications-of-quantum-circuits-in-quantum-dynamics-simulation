import pickle
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj
from qiskit import qpy
from qiskit.quantum_info import Statevector

tf = 2                                                                            #这里手工修改时间
dt = 0.05                                                                           #手工修改时间步长
times = np.linspace(0, tf, int(tf/dt) + 1)
times = times[1:]                              #不要第一个元素

with open('schwinger_exact_states.pkl', 'rb') as f:
    exact_states = pickle.load(f)             #是一个列表，每个元素均为向量
exact_states = exact_states[1:]               #不要第一个元素

#平方
with open('schwinger_trotter.qpy', 'rb') as f:           #TROTTER做出来，缺少第一个元素
    trotter_states = qpy.load(f)
#approximate_states = approximate_states[1:]           #只有TROTTER,需要注释掉这句
#print(approximate_states[2].draw())           #draw可将量子电路画出，且显示参数

with open('schwinger_pvqd.qpy', 'rb') as f:           #TROTTER做出来，缺少第一个元素
    pvqd_states = qpy.load(f)
pvqd_states = pvqd_states[1:]

#线性1阶
with open('schwinger_linear.qpy', 'rb') as f:           #TROTTER做出来，缺少第一个元素
    local_states = qpy.load(f)
#local_states = local_states[1:]

#线性2阶
with open('schwinger_nonlocal.qpy', 'rb') as f:           #TROTTER做出来，缺少第一个元素
    nonlocal_states = qpy.load(f)
nonlocal_states = nonlocal_states[1:]








# 将每个 QuantumCircuit 转换为状态向量
#approx_statevectors = [Statevector(circuit) for circuit in approximate_states]
# 将 Statevector 转换为 Qobj（需手动指定维度）
# 1. 将 QuantumCircuit 转换为 Qobj 态
trotter_statevectors = [Statevector(circuit) for circuit in trotter_states]
trotter_states_qobj = [
    Qobj(statevector.data.reshape(-1, 1), dims=[[256], [1]])                       #这里手工修改dim
    for statevector in trotter_statevectors
]
         
trotter_L2 = []
trotter_BURES = []
l2_trotter = 0.0          # 累加L²距离
bures_trotter = 0.0       # 累加Bures距离
# 3. 计算每个时间步的距离并累加
for exact, approx in zip(exact_states, trotter_states_qobj):
    overlap = exact.overlap(approx)
    l2 = 2 * (1 - np.real(overlap))                # 当前时间步的距离       
    bures = 2 * (1 - np.abs(overlap))
    l2_trotter += l2                                 # 累加到总和中
    bures_trotter += bures
    trotter_L2.append(np.sqrt(l2_trotter))            #开根号并添加到GLOBAL_L2中
    trotter_BURES.append(np.sqrt(bures_trotter))      #开根号并添加到GLOBAL_BURES中


pvqd_statevectors = [Statevector(circuit) for circuit in pvqd_states]
pvqd_states_qobj = [
    Qobj(statevector.data.reshape(-1, 1), dims=[[256], [1]])                       #这里手工修改dim
    for statevector in pvqd_statevectors
]
         
pvqd_L2 = []
pvqd_BURES = []
l2_pvqd = 0.0          # 累加L²距离
bures_pvqd = 0.0       # 累加Bures距离
# 3. 计算每个时间步的距离并累加
for exact, approx in zip(exact_states, pvqd_states_qobj):
    overlap = exact.overlap(approx)
    l2 = 2 * (1 - np.real(overlap))                # 当前时间步的距离       
    bures = 2 * (1 - np.abs(overlap))
    l2_pvqd += l2                                 # 累加到总和中
    bures_pvqd += bures
    pvqd_L2.append(np.sqrt(l2_pvqd))            #开根号并添加到GLOBAL_L2中
    pvqd_BURES.append(np.sqrt(bures_pvqd))      #开根号并添加到GLOBAL_BURES中


local_statevectors = [Statevector(circuit) for circuit in local_states]
local_states_qobj = [
    Qobj(statevector.data.reshape(-1, 1), dims=[[256], [1]])                       #这里手工修改dim
    for statevector in local_statevectors
]
         
local_L2 = []
local_BURES = []
l2_local = 0.0          # 累加L²距离
bures_local = 0.0       # 累加Bures距离
# 3. 计算每个时间步的距离并累加
for exact, approx in zip(exact_states, local_states_qobj):
    overlap = exact.overlap(approx)
    l2 = 2 * (1 - np.real(overlap))                # 当前时间步的距离       
    bures = 2 * (1 - np.abs(overlap))
    l2_local += l2                                 # 累加到总和中
    bures_local += bures
    local_L2.append(np.sqrt(l2_local))            #开根号并添加到GLOBAL_L2中
    local_BURES.append(np.sqrt(bures_local))      #开根号并添加到GLOBAL_BURES中









nonlocal_statevectors = [Statevector(circuit) for circuit in nonlocal_states]
nonlocal_states_qobj = [
    Qobj(statevector.data.reshape(-1, 1), dims=[[256], [1]])                       #这里手工修改dim
    for statevector in nonlocal_statevectors
]
         
nonlocal_L2 = []
nonlocal_BURES = []
l2_nonlocal = 0.0          # 累加L²距离
bures_nonlocal = 0.0       # 累加Bures距离
# 3. 计算每个时间步的距离并累加
for exact, approx in zip(exact_states, nonlocal_states_qobj):
    overlap = exact.overlap(approx)
    l2 = 2 * (1 - np.real(overlap))                # 当前时间步的距离       
    bures = 2 * (1 - np.abs(overlap))
    l2_nonlocal += l2                                 # 累加到总和中
    bures_nonlocal += bures
    nonlocal_L2.append(np.sqrt(l2_nonlocal))            #开根号并添加到GLOBAL_L2中
    nonlocal_BURES.append(np.sqrt(bures_nonlocal))      #开根号并添加到GLOBAL_BURES中
# 6. 可视化
fig, ax = plt.subplots(1, 1)

ax.plot(times, pvqd_BURES, label="p-VQD n=3", linestyle='dashed', linewidth=1, color='black')
ax.errorbar(times, local_BURES, label="Trotter linear", marker='o', linestyle='', elinewidth=1, color='red', capsize=2, markersize=3.5)
ax.errorbar(times, trotter_BURES, label="Trotter square", marker='^', linestyle='', elinewidth=1, color='blue', capsize=2, markersize=3.5)
#ax.plot(times, nonlocal_BURES, label="Bures nonlocal", linestyle='dashed', linewidth=1, color='green')

#ax.errorbar(times, trotter_L2, label=r"$l^2$ trotter", marker='d', linestyle='', elinewidth=1, color='red', capsize=2, markersize=3.5)
#ax.errorbar(times, pvqd_L2, label=r"$l^2$ pvqd", marker='d', linestyle='', elinewidth=1, color='yellow', capsize=2, markersize=3.5)
#ax.errorbar(times, local_L2, label=r"$l^2$ local", marker='d', linestyle='', elinewidth=1, color='blue', capsize=2, markersize=3.5)
#ax.errorbar(times, nonlocal_L2, label=r"$l^2$ nonlocal", marker='d', linestyle='', elinewidth=1, color='green', capsize=2, markersize=3.5)
ax.set(ylabel = "Bures distances",xlabel = r"t")
ax.set_ylim(ymax = 1, ymin = 0)
ax.set_xlim(xmin=0)

ax.set_title("Schwinger model,N=8", fontsize=14)
# 添加图例
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=2, fancybox=True, shadow=False)
plt.savefig('施温格平方black.pdf') 
plt.show()
