import time
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = np.random.random(), np.random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return inside_circle

def parallel_monte_carlo_pi(total_samples, num_cores):
    samples_per_core = total_samples // num_cores
    with Pool(processes=num_cores) as pool:
        results = pool.map(monte_carlo_pi, [samples_per_core] * num_cores)
    
    total_inside_circle = sum(results)
    return (4 * total_inside_circle) / total_samples

total_samples = 10**8
execution_times = []
num_cores = os.cpu_count()

for num_processes in range(1, num_cores + 1):
    start_time = time.time()
    pi_approximation = parallel_monte_carlo_pi(total_samples, num_processes)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)

    print(f"Tiempo de ejecución con {num_processes} núcleos: {execution_time:.4f} segundos")
    print(f"Aproximación de Pi con {num_processes} núcleos: {pi_approximation:.6f}")

plt.plot(range(1, num_cores + 1), execution_times, marker='o')
plt.xlabel('Número de núcleos')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Simulación de Monte Carlo para aproximar Pi')
plt.grid(True)
plt.show()