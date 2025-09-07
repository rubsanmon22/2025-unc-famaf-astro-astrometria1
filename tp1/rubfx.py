#%%
import numpy as np 
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