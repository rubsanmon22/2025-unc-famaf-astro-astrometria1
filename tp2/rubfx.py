#%%
import numpy as np 
#%%

# ACÁ ESTAN TODAS LAS FUNCIONES, 
# CADA UNA SERÁ IMPLEMENTADA DE FORMA PRÁCTICA EN tp1.ipynb
#%%

#Generador lineal de congruencia  (enteros)

def gclz (x0, a, c, m, n):

    x = x0

    numeros = []
    for i in range(n):
        x = (a * x + c) % m
        numeros.append(x)

    return numeros
# %%

#Generador lineal de congruencia (0,1)

def gclr (x0, a, c, m, n):

    return np.array(gclz(x0, a, c, m, n)) / m

# %%

#Momentos teóricos de la distribución uniforme continua en [0,1]

def momt (k):
    return 1/(k+1)
# %%

#Calcula los momentos empíricos de la distribución obtenida

def mome (data, k):
    n = len(data) # Número de datos
    return sum(x**k for x in data) / n

#%%

#Generador de fibonacci para valores enteros

def fiboz(semilla, n, j=24, k=55, m=2**32, glcz=None):
    """
    Generador de Fibonacci con Retardo inicializado con TU generador congruencial
    
    Parameters:
    semilla: valor inicial para tu generador
    n: número de valores a generar
    j, k: retardos (default 24, 55)
    m: módulo
    tu_generador_congruencial: función de tu generador existente
    """
    
    # Primero generamos k valores iniciales con TU generador
    valores_iniciales = gclz(semilla, 1664525, 1013904223, m, k)

    # Array circular para eficiencia
    buffer = valores_iniciales[:]  # Copiamos los valores iniciales
    idx_actual = 0
    resultados = []
    
    # Generamos n números con LFG
    for _ in range(n):
        # Índices de los elementos a combinar
        idx_j = (idx_actual - j) % k
        idx_k = (idx_actual - k) % k  # Siempre idx_actual porque k = tamaño del buffer
        
        # Aplicamos la fórmula de Fibonacci
        nuevo_valor = (buffer[idx_j] + buffer[idx_k]) % m
        buffer[idx_actual] = nuevo_valor
        resultados.append(nuevo_valor)
        
        # Movemos el índice circular
        idx_actual = (idx_actual + 1) % k
    
    return resultados

#%%

#Generador de fibonacci para valores entre 0 y 1

def fibor(semilla, n, j=24, k=55, m=2**32, glcz=None):
    """
    Versión que devuelve números en [0, 1) usando glc para inicializar
    """
    enteros = fiboz(semilla, n, j, k, m, glcz)
    return [x / m for x in enteros]
# %%

# Calculo de coeficiente de correlación de pearson

def pearson(x, y):
    """
    Calcula el coeficiente de correlacion de Pearson entre dos arrays

    Parameters:
        x, y: arrays de igual longitud

    Returns:
        r: coeficiente de correlacion de Pearson
    """
    # Verificar que tienen la misma longitud
    if len(x) != len(y):
        raise ValueError("Los arrays deben tener la misma longitud")

    n = len(x)

    # Calcular medias
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calcular numerador y denominador
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    # Evitar division por cero
    if denominator == 0:
        return 0

    return numerator / denominator

#%%
#BOOTSTRAP

def bootstrap (x, func, m=1000):
    y = np.zeros(m)
    for i in range(m):
        _x = np.random.choice(x, size=len(x), replace=True)
        y[i] = func(_x)
    return y
