#%%
# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Configuración para reproducibilidad
np.random.seed(42)
plt.style.use('default')

#%%
################################################
# Módulo de funciones para inferencia bayesiana
# Caso lineal simple: y = a + b*x + ruido_gaussiano
################################################

#%%
# Funciones para inferencia bayesiana
def log_likelihood(params, x, y, sigma):
    """Calcula el log-likelihood gaussiano"""
    a, b = params
    y_model = a + b * x
    chi2 = np.sum((y - y_model)**2) / (2 * sigma**2)
    return -chi2

#%%
def log_prior_flat(params):
    """Prior plano (uniforme) con límites amplios"""
    a, b = params
    if -10 <= a <= 15 and -5 <= b <= 10:
        return 0.0  # log(constante) = 0
    else:
        return -np.inf

#%%
def log_prior_gaussian(params, mu_a=0, sigma_a=10, mu_b=0, sigma_b=10):
    """Prior gaussiano para a y b"""
    a, b = params
    log_p_a = -0.5 * ((a - mu_a) / sigma_a)**2
    log_p_b = -0.5 * ((b - mu_b) / sigma_b)**2
    return log_p_a + log_p_b

#%%
def log_posterior(params, x, y, sigma, prior_type='flat'):
    """Calcula el log-posterior = log-likelihood + log-prior"""
    log_like = log_likelihood(params, x, y, sigma)
    
    if prior_type == 'flat':
        log_prior = log_prior_flat(params)
    elif prior_type == 'gaussian':
        log_prior = log_prior_gaussian(params)
    else:
        raise ValueError("prior_type debe ser 'flat' o 'gaussian'")
    
    if not np.isfinite(log_prior):
        return -np.inf
    
    return log_like + log_prior

#%%
# Algoritmo Metropolis-Hastings simple
def metropolis_hastings_simple(x, y, sigma, n_steps=10000, step_size=0.1, 
                              initial_params=None, prior_type='flat'):
    """
    Implementación simple de Metropolis-Hastings
    """
    if initial_params is None:
        initial_params = [1.0, 1.0]  # valores iniciales para [a, b]
    
    # Inicializar
    samples = np.zeros((n_steps, 2))
    current_params = np.array(initial_params)
    current_log_post = log_posterior(current_params, x, y, sigma, prior_type)
    
    n_accepted = 0
    
    for i in range(n_steps):
        # Proponer nuevo estado (caminata aleatoria)
        proposal = current_params + np.random.normal(0, step_size, 2)
        proposal_log_post = log_posterior(proposal, x, y, sigma, prior_type)
        
        # Criterio de aceptación de Metropolis
        if np.isfinite(proposal_log_post):
            log_alpha = proposal_log_post - current_log_post
            alpha = min(1, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                current_params = proposal
                current_log_post = proposal_log_post
                n_accepted += 1
        
        samples[i] = current_params
    
    acceptance_rate = n_accepted / n_steps
    return samples, acceptance_rate

################################################
# Módulo de funciones para inferencia bayesiana
# Función de Schechter para la función de luminosidad
################################################
#%%
def schechter_function(M, phi_star, M_star, alpha):
    """
    Implementación de la función de Schechter Φ(M).
    """
    # Usamos np.log(10) para ln(10)
    term1 = 0.4 * np.log(10) * phi_star
    
    # Exponente 10
    exponent_10 = -0.4 * (M - M_star) * (alpha + 1)
    
    # Exponente exp
    exponent_exp = -10**(-0.4 * (M - M_star))
    
    Phi = term1 * (10**(exponent_10)) * np.exp(exponent_exp)
    
    # La función debe devolver un valor positivo para ser una función de densidad 
    return np.maximum(Phi, 1e-30)

#%%
def log_likelihood_schechter(params, x_data, y_data, sigma_data):
    """
    Calcula el logaritmo de la función de Likelihood.
    """
    phi_star, M_star, alpha = params
    
    # Predecir los valores usando el modelo Schechter
    y_model = schechter_function(x_data, phi_star, M_star, alpha)
    
    # Cálculo del log-likelihood para errores Gaussianos
    residuals = y_data - y_model
    chi_square = np.sum(residuals**2 / sigma_data**2)
    
    return -0.5 * chi_square

#%%
# Límites para los parámetros de Schechter
PHI_BOUNDS = (1e-6, 1.0)      # Rango para Phi*
M_BOUNDS = (-25.0, -15.0)     # Rango para M*
ALPHA_BOUNDS = (-3.0, 1.0)    # Rango para alpha

#%%
def log_prior_schechter(params):
    """
    Calcula el logaritmo de la función Prior.
    """
    phi_star, M_star, alpha = params
    
    # Chequeo de límites (Flat/Uniform Prior)
    if (PHI_BOUNDS[0] < phi_star < PHI_BOUNDS[1] and
        M_BOUNDS[0] < M_star < M_BOUNDS[1] and
        ALPHA_BOUNDS[0] < alpha < ALPHA_BOUNDS[1]):
        return 0.0
    else:
        return -np.inf

#%%
def log_posterior_schechter(params, x_data, y_data, sigma_data):
    """
    Calcula el logaritmo de la probabilidad posterior.
    """
    # 1. Obtener Log-Prior
    lp = log_prior_schechter(params)
    
    # Si el prior es -inf, el posterior es -inf
    if not np.isfinite(lp):
        return -np.inf
    
    # 2. Obtener Log-Likelihood
    ll = log_likelihood_schechter(params, x_data, y_data, sigma_data)
    
    # 3. Sumar para obtener el Log-Posterior
    return lp + ll
#%%

def mhschechter(M_data, phi_data, phi_err, n_steps=10000, step_size=None, initial_params=None):
    """
    Implementación de Metropolis-Hastings para la función de Schechter.
    
    Parameters:
    -----------
    M_data : array
        Magnitudes absolutas observadas
    phi_data : array
        Función de luminosidad observada
    phi_err : array
        Errores en la función de luminosidad
    n_steps : int
        Número de pasos MCMC
    step_size : list
        Tamaños de paso para [phi_star, M_star, alpha]
    initial_params : list
        Parámetros iniciales [phi_star, M_star, alpha]
    
    Returns:
    --------
    chain : array
        Cadena MCMC con forma (n_steps, 3)
    acceptance_rate : float
        Tasa de aceptación
    """
    
    # Valores por defecto
    if step_size is None:
        step_size = [0.01, 0.2, 0.05]  # [phi_star, M_star, alpha]
    
    if initial_params is None:
        initial_params = [0.1, -20.0, -1.0]  # [phi_star, M_star, alpha]
    
    # Convertir a arrays numpy
    step_size = np.array(step_size)
    current_params = np.array(initial_params)
    
    # Inicializar cadena
    chain = np.zeros((n_steps, 3))
    n_accepted = 0
    
    # Evaluar log-posterior inicial
    current_log_post = log_posterior_schechter(current_params, M_data, phi_data, phi_err)
    
    for i in range(n_steps):
        # Proponer nuevos parámetros
        proposal = current_params + np.random.normal(0, step_size, 3)
        
        # Evaluar log-posterior de la propuesta
        proposal_log_post = log_posterior_schechter(proposal, M_data, phi_data, phi_err)
        
        # Criterio de aceptación de Metropolis
        if np.isfinite(proposal_log_post):
            log_alpha = proposal_log_post - current_log_post
            alpha = min(1.0, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                current_params = proposal
                current_log_post = proposal_log_post
                n_accepted += 1
        
        # Guardar estado actual
        chain[i] = current_params
    
    acceptance_rate = n_accepted / n_steps
    return chain, acceptance_rate