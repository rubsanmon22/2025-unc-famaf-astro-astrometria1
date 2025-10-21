#%%
# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

# Configuración para reproducibilidad
np.random.seed(42)
plt.style.use('default')
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