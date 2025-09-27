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
    Uso t(x)
    Parametros:
        x: input
        sigma: valor de sigma
        mu: valor de mu
        epsilon: valor de epsilon 
    
    Returns:
        t: Valor de la función por partes
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
        sigma: valor de sigma
        mu: valor de mu
        epsilon: valor de epsilon

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
    Uso t(x)
    Parametros:
        x: input
        sigma: valor de sigma
        mu: valor de mu
        epsilon: valor de epsilon

    Returns:
        F: Valor de la función acumulada
    """
    t_ep = t(x, epsilon, sigma, mu)

    return np.exp(-t_ep)

#%%
#Función inversa

def invF(y, epsilon, sigma, mu):
    """
    Defino función inversa de F(x)
    Uso F(x)
    Parametros:
        y: input
        sigma: valor de sigma
        mu: valor de mu
        epsilon: valor de epsilon
    
    Returns:
        invF: Valor de la función inversa
    """
    if epsilon == 0:
        return -sigma * np.log(-np.log(y)) + mu
    else:
        return sigma/epsilon * ((-np.log(y))**(-epsilon) - 1) + mu

#%%
# SIMULACIÓN DE UN PROCESO DE POISSON

def t_exp(lamb):

    """
    Genera tiempo entre eventos usando transformada inversa
    
    Parametros:
        lamb: tasa de eventos por unidad de tiempo
    Returns:
        t_exp: tiempo entre eventos
    """
    u = np.random.random()  # Número aleatorio entre 0 y 1
    return -np.log(1 - u) / (lamb)  # Transformada inversa

#%%
# SIMULAR PROCESO DE POISSON
def sim_poisson(lam, T):
    """
    Simula proceso de Poisson hasta tiempo T
    Parametros:
        lam: tasa de eventos por unidad de tiempo
        T: tiempo total de simulación
    Returns:
        eventos: lista de tiempos de eventos
    """
    tiempo_actual = 0
    eventos = []
    
    while tiempo_actual < T:
        dt = t_exp(lam)  # Tiempo hasta próximo evento
        tiempo_actual += dt
        if tiempo_actual < T:
            eventos.append(tiempo_actual)
    
    return eventos
# %%
