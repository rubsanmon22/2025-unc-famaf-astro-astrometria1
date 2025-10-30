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
def compute_stats(samples):
    mean_vals = np.mean(samples, axis=0)
    std_vals = np.std(samples, axis=0)
    percentiles = np.percentile(samples, [2.5, 50, 97.5], axis=0)
    return mean_vals, std_vals, percentiles
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

#%%

burn_in = 3000

def analyze_mixing(chains, condition_name):
    """Analiza métricas de mezclado para un conjunto de cadenas"""
    print(f"\n--- {condition_name} ---")
    
    # Remover burn-in
    chains_clean = [chain[burn_in:] for chain in chains]
    
    # Calcular estadísticas por cadena
    means_per_chain = np.array([np.mean(chain, axis=0) for chain in chains_clean])
    stds_per_chain = np.array([np.std(chain, axis=0) for chain in chains_clean])
    
    # Variación entre cadenas vs dentro de cadenas
    between_chain_var = np.std(means_per_chain, axis=0)  # Variación entre medias
    within_chain_var = np.mean(stds_per_chain, axis=0)   # Variación promedio dentro de cadenas
    
    print("Variación entre cadenas vs dentro de cadenas:")
    for i, param_name in enumerate(['phi_star', 'M_star', 'alpha']):
        ratio = between_chain_var[i] / (within_chain_var[i] / np.sqrt(len(chains_clean[0])))
        print(f"  {param_name}: Entre/Dentro = {ratio:.3f} (ideal < 1.1)")
    
    return means_per_chain, stds_per_chain, between_chain_var, within_chain_var

#%%
def compute_ks_distance(chains, param_idx):
    """Calcula distancias Kolmogorov-Smirnov entre pares de cadenas"""
    from scipy.stats import ks_2samp
    
    chains_clean = [chain[burn_in:, param_idx] for chain in chains]
    distances = []
    
    for i in range(len(chains_clean)):
        for j in range(i+1, len(chains_clean)):
            stat, pval = ks_2samp(chains_clean[i], chains_clean[j])
            distances.append(stat)
    
    return np.mean(distances)
#%%

def compute_chain_separation(chains, window_size=500):
    """Calcula la separación entre cadenas como función del número de pasos"""
    n_steps = len(chains[0])
    n_params = chains[0].shape[1]
    
    # Puntos donde evaluar la separación
    eval_points = range(window_size, n_steps, window_size)
    separations = np.zeros((len(eval_points), n_params))
    
    for i, step in enumerate(eval_points):
        # Tomar ventana de datos hasta este paso
        window_data = [chain[:step] for chain in chains]
        
        # Calcular medias de cada cadena en esta ventana
        means = np.array([np.mean(chain, axis=0) for chain in window_data])
        
        # Separación = desviación estándar de las medias
        separations[i] = np.std(means, axis=0)
    
    return eval_points, separations

#%%
# Función de log-likelihood negativa (para minimización)
def neg_log_likelihood(params, M_r, phi_obs, phi_err):
    """
    Calcula el log-likelihood negativo para la función de Schechter
    
    Parameters:
    -----------
    params : array-like
        [phi_star, M_star, alpha] - parámetros de la función de Schechter
    M_r : array
        Magnitudes absolutas observadas
    phi_obs : array
        Valores observados de la función de luminosidad
    phi_err : array
        Errores en phi_obs
    
    Returns:
    --------
    float : log-likelihood negativo
    """
    phi_star, M_star, alpha = params
    
    # Verificar que los parámetros estén en rangos físicos
    if phi_star <= 0 or alpha >= 0:  # alpha debe ser negativo
        return np.inf
    
    try:
        # Calcular modelo teórico
        phi_model = schechter_function(M_r, phi_star, M_star, alpha)
        
        # Verificar que el modelo sea válido
        if np.any(phi_model <= 0) or np.any(~np.isfinite(phi_model)):
            return np.inf
        
        # Calcular chi-cuadrado (equivalente a -2*log-likelihood)
        chi2 = np.sum(((phi_obs - phi_model) / phi_err)**2)
        
        return 0.5 * chi2  # Retornar -log-likelihood
        
    except:
        return np.inf

#%%
# Gradiente numérico
def numerical_gradient(params, M_r, phi_obs, phi_err, epsilon=1e-6):
    """
    Calcula el gradiente numérico del log-likelihood negativo
    
    Parameters:
    -----------
    params : array-like
        [phi_star, M_star, alpha]
    epsilon : float
        Paso para la diferenciación numérica
    
    Returns:
    --------
    array : gradiente [d/d_phi_star, d/d_M_star, d/d_alpha]
    """
    grad = np.zeros_like(params)
    f0 = neg_log_likelihood(params, M_r, phi_obs, phi_err)
    
    for i in range(len(params)):
        # Perturbación hacia adelante
        params_plus = params.copy()
        params_plus[i] += epsilon
        f_plus = neg_log_likelihood(params_plus, M_r, phi_obs, phi_err)
        
        # Perturbación hacia atrás
        params_minus = params.copy()
        params_minus[i] -= epsilon
        f_minus = neg_log_likelihood(params_minus, M_r, phi_obs, phi_err)
        
        # Derivada centrada
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    return grad

#%%
# IMPLEMENTACIÓN DEL GRADIENTE DESCENDENTE

M_r = None  # Datos de magnitudes absolutas (definir externamente)
phi = None  # Datos de función de luminosidad observada (definir externamente)
phi_err = None  # Errores en phi (definir externamente)

def gradient_descent(initial_params, M_r, phi_obs, phi_err, 
                    learning_rate=0.001, max_iterations=10000, 
                    tolerance=1e-8, adaptive_lr=True):
    """
    Implementación del algoritmo de gradiente descendente
    
    Parameters:
    -----------
    initial_params : array
        Parámetros iniciales [phi_star, M_star, alpha]
    learning_rate : float
        Tasa de aprendizaje inicial
    max_iterations : int
        Número máximo de iteraciones
    tolerance : float
        Criterio de convergencia
    adaptive_lr : bool
        Si usar tasa de aprendizaje adaptiva
    
    Returns:
    --------
    dict : Resultados del algoritmo
    """
    
    # Inicialización
    params = np.array(initial_params, dtype=float)
    history = {'params': [params.copy()], 
               'likelihood': [], 
               'gradient_norm': [],
               'learning_rates': []}
    
    lr = learning_rate
    best_params = params.copy()
    best_likelihood = neg_log_likelihood(params, M_r, phi_obs, phi_err)
    
    print(f"Iniciando gradiente descendente...")
    print(f"Parámetros iniciales: φ*={params[0]:.3e}, M*={params[1]:.2f}, α={params[2]:.3f}")
    print(f"Likelihood inicial: {-best_likelihood:.3f}")
    
    for iteration in range(max_iterations):
        # Calcular likelihood actual
        current_likelihood = neg_log_likelihood(params, M_r, phi_obs, phi_err)
        
        # Calcular gradiente
        grad = numerical_gradient(params, M_r, phi_obs, phi_err)
        grad_norm = np.linalg.norm(grad)
        
        # Guardar en historial
        history['likelihood'].append(-current_likelihood)
        history['gradient_norm'].append(grad_norm)
        history['learning_rates'].append(lr)
        
        # Verificar convergencia
        if grad_norm < tolerance:
            print(f"Convergencia alcanzada en iteración {iteration}")
            print(f"Norma del gradiente: {grad_norm:.2e}")
            break
        
        # Actualizar parámetros
        new_params = params - lr * grad
        
        # Verificar que los parámetros estén en rangos válidos
        new_params[0] = max(new_params[0], 1e-6)  # phi_star > 0
        new_params[2] = min(new_params[2], -0.01)  # alpha < 0
        
        # Calcular nueva likelihood
        new_likelihood = neg_log_likelihood(new_params, M_r, phi_obs, phi_err)
        
        # Tasa de aprendizaje adaptiva
        if adaptive_lr:
            if new_likelihood < current_likelihood:  # Mejora
                lr *= 1.05  # Aumentar ligeramente
                params = new_params
                if new_likelihood < best_likelihood:
                    best_params = new_params.copy()
                    best_likelihood = new_likelihood
            else:  # No mejora
                lr *= 0.5  # Reducir tasa de aprendizaje
                if lr < 1e-10:
                    print(f"Tasa de aprendizaje muy pequeña en iteración {iteration}")
                    break
        else:
            params = new_params
            if new_likelihood < best_likelihood:
                best_params = new_params.copy()
                best_likelihood = new_likelihood
        
        history['params'].append(params.copy())
        
        # Progreso cada 1000 iteraciones
        if iteration % 1000 == 0:
            print(f"Iter {iteration:5d}: φ*={params[0]:.3e}, M*={params[1]:.2f}, α={params[2]:.3f}, "
                  f"LL={-current_likelihood:.3f}, |grad|={grad_norm:.2e}, lr={lr:.2e}")
    
    else:
        print(f"Máximo de iteraciones alcanzado ({max_iterations})")
    
    return {
        'best_params': best_params,
        'best_likelihood': -best_likelihood,
        'final_params': params,
        'final_likelihood': -current_likelihood,
        'history': history,
        'iterations': min(iteration + 1, max_iterations),
        'converged': grad_norm < tolerance
    }
#%%
def likelihood_surface_1d(param_idx, param_range, best_params):
    """Analiza la superficie de likelihood en 1D"""
    likelihoods = []
    params_test = best_params.copy()
    
    for val in param_range:
        params_test[param_idx] = val
        ll = -neg_log_likelihood(params_test, M_r, phi, phi_err)
        likelihoods.append(ll)
    
    return np.array(likelihoods)



#%%
def numerical_hessian(params, M_r, phi_obs, phi_err, epsilon=1e-4):
    """Calcula la matriz Hessiana numéricamente"""
    n = len(params)
    hessian = np.zeros((n, n))
    f0 = neg_log_likelihood(params, M_r, phi_obs, phi_err)
    
    for i in range(n):
        for j in range(n):
            # Términos de la segunda derivada
            params_pp = params.copy()
            params_pp[i] += epsilon
            params_pp[j] += epsilon
            f_pp = neg_log_likelihood(params_pp, M_r, phi_obs, phi_err)
            
            params_pm = params.copy()
            params_pm[i] += epsilon
            params_pm[j] -= epsilon
            f_pm = neg_log_likelihood(params_pm, M_r, phi_obs, phi_err)
            
            params_mp = params.copy()
            params_mp[i] -= epsilon
            params_mp[j] += epsilon
            f_mp = neg_log_likelihood(params_mp, M_r, phi_obs, phi_err)
            
            params_mm = params.copy()
            params_mm[i] -= epsilon
            params_mm[j] -= epsilon
            f_mm = neg_log_likelihood(params_mm, M_r, phi_obs, phi_err)
            
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
    
    return hessian