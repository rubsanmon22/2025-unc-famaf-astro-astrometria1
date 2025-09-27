#%%
import numpy as np 
#%%

# ACÁ ESTAN TODAS LAS FUNCIONES, 
# CADA UNA SERÁ IMPLEMENTADA DE FORMA PRÁCTICA EN tp2.ipynb


#%%

#FUNCIÓN POR PARTES QUE DEFINE f(x)
def t(x, epsilon, sigma, mu):
    """
    Defino la función por partes
    """

    if epsilon == 0:
        return np.exp(-(x-mu)/sigma)
    else:
        return (1+epsilon*((x-mu)/sigma))**(-1/epsilon)

#%%
#DISTRIBUCIÓN FISHER-TIPPETT

def f(x, epsilon, sigma, mu):

    """
    Defino la función distribución de FISHER-TIPPET
    Uso t(x)
    Parametros:
        x: input
        sigma:
        epsilon:

    Returns:
        f: Valor de la función F-T
    """
    t_ep = t(x, epsilon, sigma, mu)
    return (1/sigma) * (t_ep**(epsilon + 1)) * (np.exp(-t_ep))
#%%
#FUNCIÓN ACUMULADA DE f

def F(x, epsilon, sigma, mu):

    """

    Defino función acumulada de f(x)

    """

    return np.exp(-t(x, epsilon, sigma, mu))

#%%
#Función inversa

def invF(y, epsilon, sigma, mu):
    """
    Defino función inversa de F(x)
    """
    if epsilon == 0:
        return -sigma * np.log(-np.log(y)) + mu
    else:
        return sigma/epsilon * ((-np.log(y))**(-epsilon) - 1) + mu
#%%
#BOOTSTRAP

def bootstrap (x, func, m=1000):
    y = np.zeros(m)
    for i in range(m):
        _x = np.random.choice(x, size=len(x), replace=True)
        y[i] = func(_x)
    return y

