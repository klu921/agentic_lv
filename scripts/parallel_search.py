#!/usr/bin/env python3
"""
Parallel cosine similarity search implementations
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
from functools import partial
import os

# Set number of threads for NumPy operations
def set_numpy_threads(n_threads=None):
    """Set number of threads for NumPy operations"""
    if n_threads is None:
        n_threads = mp.cpu_count()
    
    # Set environment variables before importing numpy
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)

# Method 1: Chunk-based parallel processing
def cosine_similarity_chunk(query_vec, data_chunk):
    """Compute cosine similarity for a chunk of data"""
    # Assuming vectors are already normalized
    similarities = np.dot(data_chunk, query_vec)
    return similarities

def parallel_cosine_search_chunks(query_vec, data_matrix, top_k=30, n_workers=None):
    """
    Parallel cosine search by splitting data into chunks
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    n_samples = data_matrix.shape[0]
    chunk_size = n_samples // n_workers
    
    # Split data into chunks
    chunks = []
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_workers - 1 else n_samples
        chunks.append((start_idx, end_idx, data_matrix[start_idx:end_idx]))
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for start_idx, end_idx, chunk in chunks:
            future = executor.submit(cosine_similarity_chunk, query_vec, chunk)
            futures.append((start_idx, future))
        
        # Collect results
        all_scores = np.zeros(n_samples)
        for start_idx, future in futures:
            chunk_scores = future.result()
            all_scores[start_idx:start_idx + len(chunk_scores)] = chunk_scores
    
    # Get top-k indices
    top_indices = np.argpartition(-all_scores, top_k)[:top_k]
    top_indices = top_indices[np.argsort(-all_scores[top_indices])]
    
    return top_indices, all_scores[top_indices]

# Method 2: Batch query processing
def batch_cosine_search(queries, data_matrix, top_k=30):
    """
    Process multiple queries at once using matrix multiplication
    This is often faster than processing one at a time
    """
    # queries shape: (n_queries, embedding_dim)
    # data_matrix shape: (n_samples, embedding_dim)
    
    # Compute all similarities at once
    similarities = np.dot(queries, data_matrix.T)  # Shape: (n_queries, n_samples)
    
    # Get top-k for each query
    results = []
    for i in range(len(queries)):
        scores = similarities[i]
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        results.append((top_indices, scores[top_indices]))
    
    return results

# Method 3: Using Faiss for extremely fast search
try:
    import faiss
    
    class FaissSearch:
        """Fast similarity search using Faiss library"""
        
        def __init__(self, data_matrix, use_gpu=False):
            self.dimension = data_matrix.shape[1]
            self.n_samples = data_matrix.shape[0]
            
            # Create index
            if use_gpu and faiss.get_num_gpus() > 0:
                # Use GPU if available
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatIP(res, self.dimension)
            else:
                # Use CPU with multiple threads
                self.index = faiss.IndexFlatIP(self.dimension)
                
            # Add vectors to index (assuming normalized)
            self.index.add(data_matrix.astype(np.float32))
        
        def search(self, query_vec, top_k=30):
            """Search for top-k similar vectors"""
            query = query_vec.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query, top_k)
            return indices[0], scores[0]
        
        def batch_search(self, query_vecs, top_k=30):
            """Search for multiple queries at once"""
            queries = query_vecs.astype(np.float32)
            scores, indices = self.index.search(queries, top_k)
            return indices, scores

except ImportError:
    print("Faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")
    FaissSearch = None

# Method 4: Optimized NumPy implementation
def optimized_cosine_topk(query_vec, data_matrix, k=30):
    """
    Optimized version using partial sorting
    """
    # Use einsum for potentially faster dot product
    similarities = np.einsum('d,nd->n', query_vec, data_matrix)
    
    # Use argpartition for O(n) top-k selection instead of O(n log n) full sort
    if k < len(similarities):
        # Get indices of top-k elements (unsorted)
        top_k_indices = np.argpartition(-similarities, k)[:k]
        # Sort only the top-k elements
        top_k_sorted = top_k_indices[np.argsort(-similarities[top_k_indices])]
        return top_k_sorted, similarities[top_k_sorted]
    else:
        # If k >= n, just sort everything
        sorted_indices = np.argsort(-similarities)
        return sorted_indices, similarities[sorted_indices]

# Benchmark function
def benchmark_search_methods(data_matrix, query_vec, top_k=30):
    """Benchmark different search methods"""
    
    print(f"Data shape: {data_matrix.shape}")
    print(f"Number of CPUs: {mp.cpu_count()}\n")
    
    # Standard NumPy
    start = time.time()
    similarities = np.dot(data_matrix, query_vec)
    top_indices = np.argsort(-similarities)[:top_k]
    numpy_time = time.time() - start
    print(f"Standard NumPy: {numpy_time:.4f}s")
    
    # Optimized NumPy
    start = time.time()
    optimized_cosine_topk(query_vec, data_matrix, top_k)
    opt_time = time.time() - start
    print(f"Optimized NumPy: {opt_time:.4f}s")
    
    # Parallel chunks
    start = time.time()
    parallel_cosine_search_chunks(query_vec, data_matrix, top_k)
    parallel_time = time.time() - start
    print(f"Parallel chunks: {parallel_time:.4f}s")
    
    # Faiss (if available)
    if FaissSearch:
        faiss_index = FaissSearch(data_matrix)
        start = time.time()
        faiss_index.search(query_vec, top_k)
        faiss_time = time.time() - start
        print(f"Faiss search: {faiss_time:.4f}s")
    
    print(f"\nSpeedup: {numpy_time/opt_time:.2f}x with optimization")
    print(f"Speedup: {numpy_time/parallel_time:.2f}x with parallelization")
    if FaissSearch:
        print(f"Speedup: {numpy_time/faiss_time:.2f}x with Faiss")

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create normalized random vectors
    n_samples = 100000
    embedding_dim = 1536  # Common embedding size
    
    data = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    query = np.random.randn(embedding_dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    # Run benchmark
    benchmark_search_methods(data, query, top_k=30)