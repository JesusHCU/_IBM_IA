import numpy as np

def generar_numeros_enteros_aleatorios(N, minimo, maximo):
    return list(np.random.randint(minimo, maximo + 1, size=N))