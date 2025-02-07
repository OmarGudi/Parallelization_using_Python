import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Medir el tiempo de ejecución del código
start_time = time.time()  # Registrar el tiempo de inicio

# Calculando el Fibonacci para un número grande (esto toma mucho tiempo)
resultado = fibonacci(35)

end_time = time.time()  # Registrar el tiempo de finalización
execution_time = end_time - start_time  # Calcular el tiempo transcurrido

print(f"Resultado: {resultado}")
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
