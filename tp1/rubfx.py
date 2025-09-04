#%%
#Generador lineal de congruencia
def gcl (x0, a, c, m, n):

    x = x0

    numeros = []
    for i in range(n):
        x = (a * x + c) % m
        numeros.append(x / m)
        
    return numeros
# %%
