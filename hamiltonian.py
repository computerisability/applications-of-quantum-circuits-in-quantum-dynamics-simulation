# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:11:16 2024

@author: 83464
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:29:50 2024

@author: 83464
"""

# Import the relevant libraries
from qiskit.opflow import I, X, Y, Z      #这个是老版本
#from qiskit.quantum_info import Pauli, SparsePauliOp
from functools import reduce
import numpy as np

def tens_prod(op_list):
	return reduce(lambda x, y: x^y, op_list)
	
def one_site_op(n_qubits, site, pauli_op):
	""" s_i = I^{i-1} otimes s otimes I^{N-i}, where s is a Pauli operator
	Returns:
		PauliOp object acting on all qubits
	"""
	# WARNING: PauliOp strings are read from right to left
	return tens_prod([I]*(n_qubits-site-1) + [pauli_op] + [I]*site)   #这句是老版本
    #return SparsePauliOp.from_sparse_list([("pauli_op",[site],1),num_qubits = n_qubits])

def nearest_neigh_op(n_qubits, site, pauli_ops):
	""" s_i * s_{i+1}= I^{i-1} otimes s otimes s otimes I^{N-i-1}

	Args:
		pauli_ops (List[PauliOp,PauliOp]): either X, Y, Z
	Returns:
		PauliOp object acting on all qubits
	"""
	# WARNING: PauliOp strings are read from right to left
	if site < n_qubits - 1:
		return tens_prod([I] * (n_qubits - site - 2) + pauli_ops[::-1] + [I] * site)
	elif site == n_qubits - 1:
		return tens_prod([pauli_ops[0]] + [I] * (n_qubits - 2) + [pauli_ops[1]])
	else:
		print(f'Index {site} out of bounds.')

def non(n_qubits,k,j,pauli_ops):         #k必然是大于j的
    return tens_prod([I]*(n_qubits-k-1) +[pauli_ops[0]]+ [I]*(k-j-1) +[pauli_ops[1]]+[I]*j)    
    
def xyz_hamiltonian(n_qubits, model_params, bound_cond='open'):
	""" Heisenberg XYZ model with either periodic or open boundary conditions.

	Args:
		n_qubits (int): number of qubits
		model_params (List[float,...]): 3 coupling parameters and the field strength
		bound_cond (str): boundary condition; either 'periodic' or 'open'

	Returns:
		h (PauliSumOp): Hamiltonian of the Heisenberg XYZ model
	"""

	jx, jy, jz, field = model_params  # unpack the model parameters
	h = 0  # initialize the Hamiltonian

	# Z terms
	for k in range(n_qubits):
		h += field * one_site_op(n_qubits, k, Z)

	# If the boundary conditions are periodic, set n_max = n_qubits
	n_max = n_qubits - 1 if bound_cond == 'open' else n_qubits
	for k in range(n_max):
		h += jx * nearest_neigh_op(n_qubits, k, [X, X])  # XX terms
		h += jy * nearest_neigh_op(n_qubits, k, [Y, Y])  # YY terms
		h += jz * nearest_neigh_op(n_qubits, k, [Z, Z])  # ZZ terms

	return h


def floquet_term(n_qubits):
	""" Driving term (to combine with the XYZ model)

	Args:
		n_qubits (int): number of qubits

	Returns:
		h (PauliSumOp): external driving field term
	"""
	h = 0
	for k in range(n_qubits):
		h += (-1)**k * one_site_op(n_qubits, k, Z)
	return h


def xyz_floquet_hamiltonian(n_qubits, model_params, time, bound_cond='open'):
	""" Combine the XYZ Hamiltonian with a Floquet (term to make the system non-integrable)

	Args:
		n_qubits (int): number of qubits
		model_params (List[float,...]): 3 coupling parameters and the field strength
		time (float): time of the evolution
		bound_cond (str): boundary condition; either 'periodic' or 'open'

	Returns:
		h (PauliSumOp): XYZ-Floquet Hamiltonian
	"""
	w = 0  # frequency of oscillation of the Floquet term
	h = xyz_hamiltonian(n_qubits, model_params, bound_cond=bound_cond)
	h += floquet_term(n_qubits) * np.sin(w * time)
	return h.reduce()


def c_op(n_qubits, site):
	""" Fermionic annihilation operator defined through the Jordan-Wigner mapping.

	Args:
		n_qubits (int): number of qubits
		site (int): location where the operator acts non-trivially; site=0,...,n_qubits-1

	Returns:
		Annihilation operator (PauliSumOp)
	"""
	s_plus = (X + 1j*Y)/2
	return tens_prod([I]*(n_qubits-site-1) + [s_plus] + [Z]*site)


def c_dag_op(n_qubits, site):
	""" Fermionic creation operator.

	Args:
		n_qubits (int): number of qubits
		site (int): location where the operator acts non-trivially; site=0,...,n_qubits-1

	Returns:
		Creation operator (PauliSumOp)
	"""
	return c_op(n_qubits, site).adjoint()


def num_op(n_qubits, site):
	""" Fermionic number operator.

	Args:
		n_qubits (int): number of qubits
		site (int): index where the operator pauli_op acts non-trivially; site=0,...,n_qubits-1

	Returns:
		Number operator (PauliSumOp)
	"""
	# Note: would be faster probably to do: n_i = (I-Z_i)/2
	return c_dag_op(n_qubits, site) @ c_op(n_qubits, site)


def hubbard_ladder_hamiltonian(size_x, model_params ):                         #后面那个括号前去掉time=None
	""" Hubbard model on a Lx x 2 lattice with open boundary conditions.

	Args:
		size_x (int): dimensions of the square lattice
		model_params (List[float,...]): hopping and Coulomb strength parameters
		time (float): time of the evolution (useless parameter here since the Hamiltonian is 
					time-independent; only here to avoid changing the structure of the code)

	Returns:
		h (PauliSumOp): Hamiltonian of the 2D Hubbard model
	"""

	hop, coul = model_params  # unpack the model parameters
	h = 0  # initialize the Hamiltonian
	n_qubits = 2*size_x*2  

	# Implement the hopping terms
	site_indices = range(int(n_qubits/2))
	hopping_idx_pairs = [(i, i+1) for i in site_indices[:-1]]  # spin up modes nearest neighbour index pairs
	hopping_idx_pairs += list(zip(site_indices[0::2][:-1], site_indices[1::2][1:]))  # spin up modes long-range index pairs 
	hopping_idx_pairs += list(np.array(hopping_idx_pairs) + int(n_qubits/2)) # spin down modes index pairs 
	hopping_idx_pairs = [list(x) for x in hopping_idx_pairs]
	print(f'Hopping index pairs: {hopping_idx_pairs}')

	for i, j in hopping_idx_pairs:
		h += -hop * c_dag_op(n_qubits, i) @ c_op(n_qubits, j) 
		h += -hop * c_dag_op(n_qubits, j) @ c_op(n_qubits, i)

	# Implement the on-site interaction terms
	qubit_indices = range(n_qubits)
	int_idx_pairs = list(zip(qubit_indices[:int(n_qubits/2)], qubit_indices[int(n_qubits/2):]))
	print(f'On-site interaction index pairs: {int_idx_pairs}')

	for i, j in int_idx_pairs:
		h += coul * num_op(n_qubits, i) @ num_op(n_qubits, j)	

	return h.reduce()


def hubbard_2D_hamiltonian(size_x, size_y, model_params, fermionic_indexing='allupalldown', time=None, bound_cond='open'):
	""" 2D Hubbard model on a rectangular lattice with either periodic or open boundary conditions.

	Args:
		size_x, size_y (int): dimensions of the square lattice
		model_params (List[float,...]): hopping and Coulomb strength parameters
		fermionic_indexing (str): specify the ordering of the indexing of the fermionic modes
								(either 'alternating' or 'allupalldown')
		time (float): time of the evolution (useless parameter here since the Hamiltonian is 
					time-independent; only here to avoid changing the structure of the code)
		bound_cond (str): boundary condition; either 'periodic' or 'open'

	Returns:
		h (PauliSumOp): Hamiltonian of the 2D Hubbard model
	"""

	hop, coul = model_params  # unpack the model parameters
	h = 0  # initialize the Hamiltonian
	n_qubits = 2*size_x*size_y  

	grid_2d_indices = list(np.ndindex(size_x, size_y))  # get all the 2D indices of the square grid

	# Reordoring of the 2D grid indices in a snake path
	grid_2d_indices_snake = []
	for i in range(size_y):
		if i % 2 == 0:
			grid_2d_indices_snake += grid_2d_indices[i::size_y]
		else:
			grid_2d_indices_snake += list(reversed(grid_2d_indices[i::size_y]))
	
	# Mapping from 2D grid indices to the up and down fermionic modes at that site
	if fermionic_indexing == 'alternating':
		grid_mode_idx = list(map(lambda i: [2*i, 2*i+1], range(len(grid_2d_indices_snake))))
	elif fermionic_indexing == 'allupalldown':
		grid_mode_idx = list(map(lambda i: [i, int(i+n_qubits/2)], range(len(grid_2d_indices_snake))))
	else:
		print(f'Indexing not yet implemented')
	index_map = dict(zip(grid_2d_indices_snake, grid_mode_idx))

	if bound_cond == 'open':

		# Loop over all 2D coordinates in the square grid except for the one in the top right to avoid overcounting
		for grid_2d_idx in grid_2d_indices[:-1]:
		
			# Get the x and y coordinates of the 2D grid index "grid_2d_idx" and its associated up and down fermionic mode indices
			i, j = grid_2d_idx
			up_idx, down_idx = index_map[grid_2d_idx]

			# Get the up-up and down-down hopping interaction indices between two nearest neighbours (avoiding overcounting)
			if i == size_x-1:
				nn_hop_idx_pairs = list(zip([up_idx, down_idx], index_map[(i, j+1)]))
			elif j == size_y-1:
				nn_hop_idx_pairs = list(zip([up_idx, down_idx], index_map[(i+1, j)]))
			else:
				nn_hop_idx_pairs = list(zip([up_idx, down_idx], index_map[(i, j+1)]))
				nn_hop_idx_pairs += list(zip([up_idx, down_idx], index_map[(i+1, j)]))

			# Add the nearest neighbour hopping terms (that were not already added) to the Hamiltonian
			for nn_hop_idx_pair in nn_hop_idx_pairs:
				print(f'Hopping index pair: {nn_hop_idx_pair}')
				i, j = nn_hop_idx_pair
				h += -hop * c_dag_op(n_qubits, i) @ c_op(n_qubits, j) 
				h += -hop * c_dag_op(n_qubits, j) @ c_op(n_qubits, i)

		# Loop over ALL 2D coordinates in the square grid (not so efficient to redo a loop over the grid indices
		# but it's better for clarity)
		for grid_2d_idx in grid_2d_indices_snake:
			up_idx, down_idx = index_map[grid_2d_idx]
			print(f'On-site interaction index pair: {up_idx, down_idx}')
			# Add on-site Coulomb interaction terms to the Hamiltonian
			h += coul * num_op(n_qubits, up_idx) @ num_op(n_qubits, down_idx)

	else:
		print('Periodic boundary conditions have not yet been implemented.')

	return h.reduce()

def identity_operator(n_qubits):
    # 对每个比特使用恒等算子I，并通过tens_prod将它们组合起来
    identity_ops = [one_site_op(n_qubits, site, I) for site in range(n_qubits)]
    return tens_prod(identity_ops)  # 将所有的I算子进行张量积

def schwinger_hamiltonian(n_qubits,a,g,m,ell):
    
    #x, mu,ell = model_params
    
    h=0
    for k in range(n_qubits):
        h +=0.5*m * one_site_op(n_qubits,k,I)
    for k in range(n_qubits):
        h +=0.5*m * ((-1)**k) *one_site_op(n_qubits,k,Z)
    for k in range(n_qubits-1):
        h +=1/(4*a) *nearest_neigh_op(n_qubits,k,[X,X])
        h +=1/(4*a) *nearest_neigh_op(n_qubits,k,[Y,Y])
    for j in range(n_qubits-1):
        h +=0.5*g*g*a*  0.5*(n_qubits-1-j)*(one_site_op(n_qubits,j,I)+(-1)**j *one_site_op(n_qubits,j,Z))
    for j in range(n_qubits-2):
        for k in range(j+1,n_qubits-1):
            h+= 0.5*g*g*a* 0.5*(n_qubits-1-k)*(  (-1)**(j+k)*tens_prod([I]*n_qubits)
                                    + (-1)**j *one_site_op(n_qubits, k, Z)
                                    + (-1)**k *one_site_op(n_qubits, j, Z)
                                    +non(n_qubits, k,j, [Z,Z]))
            #可以暂时先不考虑ell，因为设定为0
    h+=0.5*g*g*a* (n_qubits-1) * ell *ell *one_site_op(n_qubits,n_qubits-1,I)
    for j in range(n_qubits-1):
        for k in range(0,j+1):
            h+=0.5*g*g*a* ell * ( (-1)**k *one_site_op(n_qubits,n_qubits-1,I) + one_site_op(n_qubits,k,Z))
    #print(h)
    return h.reduce()

def hk(n_qubits,a,g,m,ell):
    
    #x, mu,ell = model_params
    
    h=0
    
    for k in range(n_qubits-1):
        h +=1/(4*a) *nearest_neigh_op(n_qubits,k,[X,X])
        h +=1/(4*a) *nearest_neigh_op(n_qubits,k,[Y,Y])
    
    return h.reduce()

def hm(n_qubits,a,g,m,ell):
    
    #x, mu,ell = model_params
    
    h=0
    for k in range(n_qubits):
        h +=0.5*m * one_site_op(n_qubits,k,I)
    for k in range(n_qubits):
        h +=0.5*m * ((-1)**k) *one_site_op(n_qubits,k,Z)
    
    return h.reduce()

def hp(n_qubits,a,g,m,ell):
    
    #x, mu,ell = model_params
    
    h=0
    
    
    for j in range(n_qubits-1):
        h +=0.5*g*g*a*  0.5*(n_qubits-1-j)*(one_site_op(n_qubits,j,I)+(-1)**j *one_site_op(n_qubits,j,Z))
    for j in range(n_qubits-2):
        for k in range(j+1,n_qubits-1):
            h+= 0.5*g*g*a* 0.5*(n_qubits-1-k)*(  (-1)**(j+k)*tens_prod([I]*n_qubits)
                                    + (-1)**j *one_site_op(n_qubits, k, Z)
                                    + (-1)**k *one_site_op(n_qubits, j, Z)
                                    +non(n_qubits, k,j, [Z,Z]))
            #可以暂时先不考虑ell，因为设定为0
    h+=0.5*g*g*a* (n_qubits-1) * ell *ell *one_site_op(n_qubits,n_qubits-1,I)
    for j in range(n_qubits-1):
        for k in range(0,j+1):
            h+=0.5*g*g*a* ell * ( (-1)**k *one_site_op(n_qubits,n_qubits-1,I) + one_site_op(n_qubits,k,Z))
    #print(h)
    return h.reduce()
    
