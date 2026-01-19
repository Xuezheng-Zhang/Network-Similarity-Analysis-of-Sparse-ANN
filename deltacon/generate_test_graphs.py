import numpy as np
import os
from scipy.sparse import csr_matrix, save_npz

def create_erdos_renyi_graph(n_nodes, p, seed=None):
    """
    Create an Erdos-Renyi random graph (sparse)
    
    Parameters:
        n_nodes: Number of nodes
        p: Probability of edge creation
        seed: Random seed
    
    Returns:
        Adjacency matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random edges
    rows = []
    cols = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < p:
                rows.append(i)
                cols.append(j)
                rows.append(j)  # Undirected graph
                cols.append(i)
    
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def create_path_graph(n_nodes):
    """
    Create a path graph (linear chain: 0-1-2-3-...)
    
    Parameters:
        n_nodes: Number of nodes
    
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    rows = []
    cols = []
    
    for i in range(n_nodes - 1):
        rows.append(i)
        cols.append(i + 1)
        rows.append(i + 1)  # Undirected
        cols.append(i)
    
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def create_ring_graph(n_nodes):
    """
    Create a ring graph (cycle: 0-1-2-...-n-0)
    
    Parameters:
        n_nodes: Number of nodes
    
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    rows = []
    cols = []
    
    for i in range(n_nodes):
        next_node = (i + 1) % n_nodes
        rows.append(i)
        cols.append(next_node)
        rows.append(next_node)  # Undirected
        cols.append(i)
    
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def create_star_graph(n_nodes):
    """
    Create a star graph (node 0 connected to all others)
    
    Parameters:
        n_nodes: Number of nodes
    
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    rows = []
    cols = []
    
    center = 0
    for i in range(1, n_nodes):
        rows.append(center)
        cols.append(i)
        rows.append(i)  # Undirected
        cols.append(center)
    
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def create_complete_graph(n_nodes):
    """
    Create a complete graph (all nodes connected to all others)
    
    Parameters:
        n_nodes: Number of nodes
    
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    rows = []
    cols = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            rows.append(i)
            cols.append(j)
            rows.append(j)  # Undirected
            cols.append(i)
    
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def create_null_graph(n_nodes):
    """
    Create a null graph (all nodes do not connected to all others)
    
    Parameters:
        n_nodes: Number of nodes
    
    Returns:
        scipy.sparse.csr_matrix: all zero
    """
    rows = []
    cols = []
    data = []
        
    adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    return adj

def main():
    # Create test_graphs directory
    output_dir = "deltacon/test_graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating test graphs...")
    
    # Test 1: Small Erdos-Renyi graph (10 nodes, p=0.3)
    adj1 = create_erdos_renyi_graph(10, 0.3, seed=42)
    save_npz(os.path.join(output_dir, "test_erdos_10_0.3.npz"), adj1)
    print(f"  Created: test_erdos_10_0.3.npz (shape: {adj1.shape}, edges: {adj1.nnz})")
    
    # Test 2: Medium Erdos-Renyi graph (20 nodes, p=0.2)
    adj2 = create_erdos_renyi_graph(20, 0.2, seed=123)
    save_npz(os.path.join(output_dir, "test_erdos_20_0.2.npz"), adj2)
    print(f"  Created: test_erdos_20_0.2.npz (shape: {adj2.shape}, edges: {adj2.nnz})")
    
    # Test 3: Path graph (15 nodes)
    adj3 = create_path_graph(15)
    save_npz(os.path.join(output_dir, "test_path_15.npz"), adj3)
    print(f"  Created: test_path_15.npz (shape: {adj3.shape}, edges: {adj3.nnz})")
    
    # Test 4: Ring graph (12 nodes)
    adj4 = create_ring_graph(12)
    save_npz(os.path.join(output_dir, "test_ring_12.npz"), adj4)
    print(f"  Created: test_ring_12.npz (shape: {adj4.shape}, edges: {adj4.nnz})")
    
    # Test 5: Star graph (10 nodes)
    adj5 = create_star_graph(1000)
    save_npz(os.path.join(output_dir, "test_star_10.npz"), adj5)
    print(f"  Created: test_star_10.npz (shape: {adj5.shape}, edges: {adj5.nnz})")
    
    # Test 6: Complete graph (10 nodes)
    adj6 = create_complete_graph(1000)
    save_npz(os.path.join(output_dir, "test_complete_10.npz"), adj6)
    print(f"  Created: test_complete_10.npz (shape: {adj6.shape}, edges: {adj6.nnz})")
    
    # Test 7: Another Erdos-Renyi graph (15 nodes, p=0.4)
    adj7 = create_erdos_renyi_graph(15, 0.4, seed=456)
    save_npz(os.path.join(output_dir, "test_erdos_15_0.4.npz"), adj7)
    print(f"  Created: test_erdos_15_0.4.npz (shape: {adj7.shape}, edges: {adj7.nnz})")
    
    # Test 8: Small Erdos-Renyi graph (8 nodes, p=0.5)
    adj8 = create_erdos_renyi_graph(8, 0.5, seed=789)
    save_npz(os.path.join(output_dir, "test_erdos_8_0.5.npz"), adj8)
    print(f"  Created: test_erdos_8_0.5.npz (shape: {adj8.shape}, edges: {adj8.nnz})")

    # Test 9: Null graph (10 nodes)
    adj9 = create_null_graph(10)
    save_npz(os.path.join(output_dir, "test_null_10.npz"), adj9)
    print(f"  Created: test_null_10.npz (shape: {adj9.shape}, edges: {adj9.nnz})")
    print(f"\nAll test graphs saved to: {output_dir}/")
    print(f"Total graphs generated: 8")

if __name__ == '__main__':
    main()
