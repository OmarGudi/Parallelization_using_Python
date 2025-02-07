import time
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

def multiply_matrices(A, B):
    return np.dot(A, B)

def parallel_matrix_multiplication(A, B, num_cores):
    chunk_size = A.shape[0] // num_cores
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    
    with Pool(processes=num_cores) as pool:
        result = pool.starmap(multiply_chunk, [(A, B, start, end) for start, end in ranges])
    
    return np.vstack(result)

def multiply_chunk(A, B, start_row, end_row):
    return np.dot(A[start_row:end_row, :], B)

A = np.random.rand(10000, 10000)
B = np.random.rand(10000, 10000)
execution_times = []
num_cores = os.cpu_count()

for num_processes in range(1, num_cores + 1):
    start_time = time.time()
    result = parallel_matrix_multiplication(A, B, num_processes)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    print(f"Tiempo de ejecución con {num_processes} núcleos: {execution_time:.4f} segundos")

plt.plot(range(1, num_cores + 1), execution_times, marker='o')
plt.xlabel('Número de núcleos')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Tiempo de ejecución para multiplicación de matrices con diferentes núcleos')
plt.grid(True)
plt.show()
