from scipy.sparse import load_npz


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


def Jaccard(A1, A2):
    # Ensure matrices are in the same format and have same shape
    if A1.shape != A2.shape:
        raise ValueError(f"Matrices must have the same shape: {A1.shape} vs {A2.shape}")
    
    # Convert to COO format for efficient edge extraction
    A1_coo = A1.tocoo()
    A2_coo = A2.tocoo()
    
    # Get edge sets as tuples (row, col) - excluding self-loops
    edges1 = set(zip(A1_coo.row, A1_coo.col))
    edges2 = set(zip(A2_coo.row, A2_coo.col))
    
    # Remove self-loops if any
    edges1 = {(r, c) for r, c in edges1 if r != c}
    edges2 = {(r, c) for r, c in edges2 if r != c}
    
    # Calculate intersection and union
    intersection = edges1 & edges2
    union = edges1 | edges2
    
    # Jaccard similarity
    if len(union) == 0:
        # Both graphs have no edges
        return 1.0 if len(edges1) == 0 and len(edges2) == 0 else 0.0
    
    jaccard = len(intersection) / len(union)
    return jaccard


if __name__ == '__main__':
    import sys
    import time
    
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        
        A1 = load_adjacency_from_npz(file1, make_undirected=False, remove_self_loops=True)
        A2 = load_adjacency_from_npz(file2, make_undirected=False, remove_self_loops=True)
        
        # Check if matrices have the same size
        if A1.shape[0] != A2.shape[0]:
            print(f"Error: Matrices have different sizes: {A1.shape[0]} vs {A2.shape[0]}")
            sys.exit(1)
        
        start = time.time()
        jaccard_sim = Jaccard(A1, A2)
        end = time.time()
        
        print(f'Jaccard similarity: {jaccard_sim:.6f}')
        print(f'Time: {end - start:.4f} seconds')
