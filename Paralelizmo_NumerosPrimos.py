import time
import matplotlib.pyplot as plt
import os
import math
from multiprocessing import Pool

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def calculate_primes(start, end):
    primes = [n for n in range(start, end) if is_prime(n)]
    return primes

def prime_calculation_parallel(limit, num_cores):
    chunk_size = limit // num_cores
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_cores)]
    
    with Pool(processes=num_cores) as pool:
        result = pool.starmap(calculate_primes, ranges)
    
    primes = [prime for sublist in result for prime in sublist]
    return primes

limit = 10**7
execution_times = []
num_cores = os.cpu_count()

for num_processes in range(1, num_cores + 1):
    start_time = time.time()
    primes = prime_calculation_parallel(limit, num_processes)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    print(f"Tiempo de ejecución con {num_processes} núcleos: {execution_time:.4f} segundos")

for i in range(1, num_cores + 1):
    print(f"\nResultados con {i} núcleos: {len(primes)} números primos encontrados")

plt.plot(range(1, num_cores + 1), execution_times, marker='o')
plt.xlabel('Número de núcleos')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title(f'Tiempo de ejecución con diferentes núcleos para cálculo de primos hasta {limit}')
plt.grid(True)
plt.show()
