import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from multiprocessing import Pool
import matplotlib.pyplot as plt
import os


#Crear una comparativa entre dos librerias que hagan paralelización usando el mismo algoritmo.
#Realizar una prueba usando un algoritmo parecido al que planeo usar.
#Revisar si puedo usar CUDA.

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_model(start_idx, end_idx, x_train, y_train):
    model = create_model()
    model.fit(x_train[start_idx:end_idx], y_train[start_idx:end_idx], epochs=5, verbose=0)
    accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy[1]

def parallel_training(num_processes, x_train, y_train):
    pool = Pool(processes=num_processes)
    chunk_size = len(x_train) // num_processes
    result = pool.starmap(train_model, [(i * chunk_size, (i + 1) * chunk_size, x_train, y_train) for i in range(num_processes)])
    pool.close()
    pool.join()
    return result

def run_parallel_experiments():
    num_cores = os.cpu_count()
    times = []
    accuracies = []

    for num_processes in range(1, num_cores + 1):
        print(f"Ejecutando con {num_processes} núcleos...")
        
        import time
        start_time = time.time()
        accuracies_for_run = parallel_training(num_processes, x_train, y_train)
        end_time = time.time()

        execution_time = end_time - start_time
        avg_accuracy = np.mean(accuracies_for_run)
        
        print(f"Tiempo de ejecución con {num_processes} núcleos: {execution_time:.4f} segundos")
        print(f"Precisión promedio con {num_processes} núcleos: {avg_accuracy:.4f}")
        
        times.append(execution_time)
        accuracies.append(avg_accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_cores + 1), times, marker='o', linestyle='-', color='b', label='Tiempo de ejecución')
    plt.xlabel('Número de núcleos')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Tiempo de Ejecución vs. Número de Núcleos')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_parallel_experiments()
