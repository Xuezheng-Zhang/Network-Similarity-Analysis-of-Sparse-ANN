#DeltaCon: proposed in A Principled Massive-Graph Similarity Function
from __future__ import division
import pandas as pd
import numpy as np
import random
import time
import sys
import os


from scipy.sparse import dok_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse import diags
from scipy.sparse import load_npz
from numpy.linalg import inv
from numpy import concatenate
from numpy import square
from numpy import array
from numpy import trace
from numpy import amax
from math import sqrt

def load_adjacency_from_npz(filepath, make_undirected=False, remove_self_loops=True):
	'''
	Load adjacency matrix from npz file
	
	Parameters:
		filepath: Path to the .npz file
		make_undirected: If True, convert directed graph to undirected (A + A.T)
		remove_self_loops: If True, remove diagonal elements (set to 0)
	
	Returns:
		scipy.sparse matrix: Adjacency matrix
	'''
	adj_matrix = load_npz(filepath)
	
	# Convert to undirected if needed
	if make_undirected:
		adj_matrix = adj_matrix + adj_matrix.T
	
	# Remove self-loops if needed
	if remove_self_loops:
		adj_matrix.setdiag(0)
		adj_matrix.eliminate_zeros()
	
	return adj_matrix

def Partition(num, size):
	'''
	randomly divide size nodes into num groups
	'''
	partitions={}
	nodes=[x for x in range(1, size+1)]
	group_size=int(size/num)
	for i in range(num-1):
		partitions[i]=[]
		for j in range(group_size):
			node=random.choice(nodes)
			nodes.remove(node)
			partitions[i].append(node)

	#the last partition get the rest nodes
	partitions[num-1]=nodes[:]

	return partitions

def Partition2e(partitions, size):
	'''
	change partition into e vector
	size is the dimension n
	'''
	e={}
	for p in partitions:
		e[p]=[]
		for i in range(1, size+1):
			if i in partitions[p]:
				e[p].append(1.0)
			else:
				e[p].append(0.0)
	return e

def InverseMatrix(A, partitions):
	'''
	use Fast Belief Propagatioin
	CITATION: Danai Koutra, Tai-You Ke, U. Kang, Duen Horng Chau, Hsing-Kuo
	Kenneth Pao, Christos Faloutsos
	Unifying Guilt-by-Association Approaches
	return [I+a*D-c*A]^-1
	'''
	num=len(partitions)		#the number of partition

	I=identity(A.shape[0])          #identity matirx
	D=diags(sum(A).toarray(), [0])  #diagonal degree matrix

	c1=trace(D.toarray())+2
	c2=trace(square(D).toarray())-1
	h_h=sqrt((-c1+sqrt(c1*c1+4*c2))/(8*c2))

	a=4*h_h*h_h/(1-4*h_h*h_h)
	c=2*h_h/(1-4*h_h*h_h)

	#M=I-c*A+a*D
	#S=inv(M.toarray())

	M=c*A+a*D
	for i in range(num):
		inv=array([partitions[i][:]]).T
		mat=array([partitions[i][:]]).T
		power=1
		while amax(M.toarray())>10**(-9) and power<10:
			mat=M.dot(mat)
			inv+=mat
			power+=1
		if i==0:
			MatrixR=inv
		else:
			MatrixR=concatenate((MatrixR, array(inv)), axis=1)

	S=csc_matrix(MatrixR)
	return S

# def InverseMatrix(A, e):
#     num = len(e) 

#     if A.nnz == 0:
#         MatrixR = np.array([e[i] for i in range(num)]).T
#         return csc_matrix(MatrixR)

#     degrees = np.array(A.sum(axis=1)).flatten()
#     max_deg = np.max(degrees)

#     epsilon = 1.0 / (1.0 + max_deg)

#     # S = I + epsilon*A + epsilon^2*A^2 + ...
#     M = epsilon * A

#     MatrixR = None
#     for i in range(num):
#         target_vec = array([e[i][:]]).T
#         inv_vec = target_vec.copy()
#         mat = target_vec.copy()
        
#         power = 1
#         while power < 10:  
#             mat = M.dot(mat)
#             inv_vec += mat
#             power += 1
            
#         if MatrixR is None:
#             MatrixR = inv_vec
#         else:
#             MatrixR = concatenate((MatrixR, inv_vec), axis=1)

#     return csc_matrix(MatrixR)

def Similarity(A1, A2, g):
	'''
	use deltacon to compute similarity
	DELTACON: A Principled Massive-Graph Similarity Function
	g is the number of partition
	'''
	size=A1.shape[0]

	partitions=Partition(g, size)
	e=Partition2e(partitions, size)
	
	S1=InverseMatrix(A1, e)
	S2=InverseMatrix(A2, e)

	d=0
	for i in range(size):
		for j in range(g):
			d+=(sqrt(S1[i,j])-sqrt(S2[i,j]))**2
	d=sqrt(d)
	sim=1/(1+d)
	return sim

def DeltaCon(A1, A2, g):
	#compute average sim
	Iteration=1
	average=0.0
	for i in range(Iteration):
		average+=Similarity(A1, A2, g)
	average/=Iteration
	return average

if __name__ == '__main__':	
	start=time.time()
	
	if len(sys.argv) >= 3:
		file1 = sys.argv[1]
		file2 = sys.argv[2]
		g = int(sys.argv[3]) if len(sys.argv) > 3 else 5
		
		A1 = load_adjacency_from_npz(file1, make_undirected=False, remove_self_loops=True)
		A2 = load_adjacency_from_npz(file2, make_undirected=False, remove_self_loops=True)
	
	# Check if matrices have the same size
	if A1.shape[0] != A2.shape[0]:
		print(f"Error: Matrices have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
		sys.exit(1)
	
	sim=DeltaCon(A1, A2, g)
	print('sim:', sim)
	end=time.time()
	print('time:',(end-start))