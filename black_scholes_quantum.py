import numpy as np
from scipy.stats import lognorm, norm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

# ==========================================
# 1. FONCTIONS UTILITAIRES (HELPER FUNCTIONS)
# ==========================================

def create_distribution(num_qubits, mu, sigma, low, high):
    """
    Crée la distribution de probabilité dans le circuit quantique.
    """
    # 1. Discrétisation de l'espace
    num_points = 2**num_qubits
    x_values = np.linspace(low, high, num_points)
    step = x_values[1] - x_values[0] if num_points > 1 else 1.0
    
    # 2. Calcul des probabilités classiques (PDF Log-Normale)
    probs = lognorm.pdf(x_values, s=np.sqrt(sigma), scale=np.exp(mu))
    probs *= step 
    probs = probs / np.sum(probs) # Normalisation
    
    # 3. Chargement dans le circuit
    state_prep = StatePreparation(np.sqrt(probs))
    
    return state_prep, x_values

def _apply_controlled_rotation(qc, uncertainty_qubits, ancilla_qubit, state_index, theta, num_qubits):
    """
    Applique une rotation Ry(theta) sur l'ancilla contrôlée par un état spécifique |state_index>.
    """
    binary_string = format(state_index, f'0{num_qubits}b')
    
    # Identifier les qubits à inverser (X) pour que le contrôle s'active sur cet état spécifique
    # Qiskit est Little-Endian, on inverse la chaîne pour mapper aux qubits [0, 1, ...]
    ctrl_qubits_to_flip = [k for k, bit in enumerate(reversed(binary_string)) if bit == '0']
    
    # 1. Activer le contrôle
    if ctrl_qubits_to_flip:
        qc.x([uncertainty_qubits[k] for k in ctrl_qubits_to_flip])
        
    # 2. Rotation Multi-Contrôlée
    qc.mcry(theta, uncertainty_qubits, ancilla_qubit)
    
    # 3. Désactiver le contrôle (Uncompute)
    if ctrl_qubits_to_flip:
        qc.x([uncertainty_qubits[k] for k in ctrl_qubits_to_flip])

def add_call_payoff(qc, uncertainty_qubits, ancilla_qubit, x_values, K, max_value):
    """
    Oracle Call: Payoff = max(S - K, 0)
    """
    num_qubits = len(uncertainty_qubits)
    for i in range(2**num_qubits):
        S_i = x_values[i]
        payoff = max(S_i - K, 0)
        
        if payoff <= 1e-6: continue
            
        # Mapping [0, max_value] -> Angle [0, pi]
        # sin^2(theta/2) = payoff / max_value
        payoff_norm = min(payoff / max_value, 1.0)
        theta = 2 * np.arcsin(np.sqrt(payoff_norm))
        
        _apply_controlled_rotation(qc, uncertainty_qubits, ancilla_qubit, i, theta, num_qubits)

def add_put_payoff(qc, uncertainty_qubits, ancilla_qubit, x_values, K, max_value):
    """
    Oracle Put: Payoff = max(K - S, 0)
    """
    num_qubits = len(uncertainty_qubits)
    for i in range(2**num_qubits):
        S_i = x_values[i]
        payoff = max(K - S_i, 0)
        
        if payoff <= 1e-6: continue
            
        payoff_norm = min(payoff / max_value, 1.0)
        theta = 2 * np.arcsin(np.sqrt(payoff_norm))
        
        _apply_controlled_rotation(qc, uncertainty_qubits, ancilla_qubit, i, theta, num_qubits)

def _run_amplitude_estimation(qc, objective_qubit, epsilon, alpha, max_payoff, r, T):
    """
    Exécute l'algorithme AE et traite les résultats.
    """
    backend = AerSimulator()
    qc_transpiled = transpile(qc, backend=backend, optimization_level=2)
    
    problem = EstimationProblem(
        state_preparation=qc_transpiled,
        objective_qubits=[objective_qubit]
    )
    
    sampler = Sampler()
    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=alpha, 
        sampler=sampler
    )
    
    result = ae.estimate(problem)
    
    # Post-processing
    raw_amplitude = result.estimation_processed
    expected_payoff = raw_amplitude * max_payoff
    price = expected_payoff * np.exp(-r * T)
    
    conf_int_scaled = [
        val * max_payoff * np.exp(-r * T) for val in result.confidence_interval_processed
    ]
    
    return {
        "price": price,
        "confidence_interval": conf_int_scaled,
        "raw_amplitude": raw_amplitude,
        "circuit_depth": qc_transpiled.depth(),
        "num_oracle_queries": result.num_oracle_queries
    }

# ==========================================
# 2. FONCTIONS PRINCIPALES (CALL & PUT)
# ==========================================

def black_scholes_call_quantum(S0, K, r, sigma, T, num_uncertainty_qubits=2, epsilon=0.05, alpha=0.1):
    """
    Pricing d'Option CALL.
    """
    # Paramètres de distribution
    mu_log = ((r - 0.5 * sigma**2) * T + np.log(S0))
    sigma_log = sigma * np.sqrt(T)
    mean = np.exp(mu_log + sigma_log**2/2)
    std = np.sqrt((np.exp(sigma_log**2) - 1) * np.exp(2*mu_log + sigma_log**2))
    
    low, high = np.maximum(0, mean - 3*std), mean + 3*std
    
    # Circuit
    qr_uncertainty = list(range(num_uncertainty_qubits))
    qr_objective = num_uncertainty_qubits
    qc = QuantumCircuit(num_uncertainty_qubits + 1)
    
    # 1. Distribution
    dist_gate, x_values = create_distribution(num_uncertainty_qubits, mu_log, sigma_log**2, low, high)
    qc.append(dist_gate, qr_uncertainty)
    
    # 2. Oracle Call
    max_payoff = max(max(x_values) - K, 1e-6) # Eviter division par zero
    add_call_payoff(qc, qr_uncertainty, qr_objective, x_values, K, max_payoff)
    
    # 3. Exécution
    try:
        res = _run_amplitude_estimation(qc, qr_objective, epsilon, alpha, max_payoff, r, T)
        res["type"] = "CALL"
        return res
    except Exception as e:
        return {"error": str(e)}

def black_scholes_put_quantum(S0, K, r, sigma, T, num_uncertainty_qubits=2, epsilon=0.05, alpha=0.1):
    """
    Pricing d'Option PUT.
    """
    # Paramètres de distribution (Identiques au Call)
    mu_log = ((r - 0.5 * sigma**2) * T + np.log(S0))
    sigma_log = sigma * np.sqrt(T)
    mean = np.exp(mu_log + sigma_log**2/2)
    std = np.sqrt((np.exp(sigma_log**2) - 1) * np.exp(2*mu_log + sigma_log**2))
    
    low, high = np.maximum(0, mean - 3*std), mean + 3*std
    
    # Circuit
    qr_uncertainty = list(range(num_uncertainty_qubits))
    qr_objective = num_uncertainty_qubits
    qc = QuantumCircuit(num_uncertainty_qubits + 1)
    
    # 1. Distribution
    dist_gate, x_values = create_distribution(num_uncertainty_qubits, mu_log, sigma_log**2, low, high)
    qc.append(dist_gate, qr_uncertainty)
    
    # 2. Oracle Put
    # Pour un put, le payoff max est K (quand S=0) ou le max observé sur la grille
    max_payoff = max(K - min(x_values), 1e-6)
    add_put_payoff(qc, qr_uncertainty, qr_objective, x_values, K, max_payoff)
    
    # 3. Exécution
    try:
        res = _run_amplitude_estimation(qc, qr_objective, epsilon, alpha, max_payoff, r, T)
        res["type"] = "PUT"
        return res
    except Exception as e:
        return {"error": str(e)}