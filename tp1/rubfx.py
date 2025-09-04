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
#Generador lineal de congruencia
def gclr (x0, a, c, m, n):

    return np.array(gclz(x0, a, c, m, n)) / m
# %%
