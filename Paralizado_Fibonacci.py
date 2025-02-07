import time
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

def fibonacci(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

def fibonacci_paralelizado(n, num_cores):
    with Pool(processes=num_cores) as pool:
        resultados = pool.map(fibonacci, range(n-7, n+7))
    return resultados

n = 950
execution_times = []
fib_results = []
num_cores = os.cpu_count()
print('El numero de cores en tu computadora es: ' + str(num_cores))

for num_processes in range(1, num_cores + 1):
    start_time = time.time()
    resultados = fibonacci_paralelizado(n, num_processes)
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    fib_results.append(resultados)

for i in range(1, num_cores + 1):
    print(f"\nResultados con {i} núcleos:")
    print(fib_results[i-1])

for i in range(1, num_cores + 1):
    print(f"\nTiempo de ejecución con {i} núcleos: {execution_times[i-1]:.4f} segundos")

plt.plot(range(1, num_cores + 1), execution_times, marker='o')
plt.xlabel('Número de núcleos')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title(f'Tiempo de ejecución de Fibonacci({n}) con diferentes números de núcleos')
plt.grid(True)
plt.show()
