# Quantum Binomial Option Pricing  
### Hackathon – Quantum Finance Challenge

This project implements and compares several option pricing methods:

1. **Black-Scholes Analytic Formula** (reference solution)  
2. **Classical Binomial Tree Model**  
3. **Quantum Binomial Model using Quantum Amplitude Estimation (QAE)**  
   - Implemented with **5 qubits**
   - And **6 qubits**

The objective is to evaluate whether a quantum version of the binomial approach can achieve **higher precision** or **better scaling** than the classical method.

---

## 🧠 Problem Overview

We aim to price **European Call and Put options** under different market scenarios:

- Risk-free rate `r`
- Strike `K`
- Initial asset price `S0`
- Volatility `σ`
- Time to maturity `T`

For each scenario, we compute:

| Method | Call Price | Put Price |
|--------|------------|-----------|
| Black-Scholes Analytic | ✔ Reference | ✔ Reference |
| Classical Binomial Tree | ✔ Approximation | ✔ Approximation |
| Quantum Binomial (QAE) | ✔ Approximation | ✔ Approximation |

The goal is to compare how each method deviates from the reference (Black-Scholes analytic).

---

## 📊 Visual Results

We generated 3 types of graphs for both **CALL** and **PUT**:

### 1. **Pricing Comparison**
Plot of prices for all 20 scenarios:

- Blue → Black-Scholes analytic  
- Orange → Classical binomial (large deviations in several scenarios)  
- Green → Quantum binomial (5 qubits)  
- Red → Quantum binomial (6 qubits)

Result:  
➡️ The **quantum curves follow Black-Scholes much more closely**  
➡️ The classical binomial diverges strongly for several (r, K, S0, σ, T) combinations.

---

### 2. **Relative Error (%) Comparison**
For each scenario:

\[
\text{Relative Error} = 100 \cdot \frac{|P_{\text{method}} - P_{\text{BS}}|}{P_{\text{BS}}}
\]

Graph:  
- Classical binomial often shows **large percentage errors**  
- Quantum binomial (5 qubits) is significantly more accurate  
- Quantum binomial (6 qubits) nearly overlaps the analytic reference

Result:  
➡️ **Quantum Binomial = Lower error and more stable**  
➡️ Increasing qubits improves the approximation

---

### 3. **Absolute Error (optional)**
Same analysis but using absolute difference.  
Shows the same trend: quantum is consistently closer to Black-Scholes.

---

## ⚛️ Quantum Approach

The quantum pricing uses **Quantum Amplitude Estimation (QAE)**:

- A payoff distribution is encoded using a log-normal quantum circuit.
- The option payoff is mapped to a specific qubit.
- QAE extracts the probability amplitude → equivalent to expected discounted payoff.
- QAE provides **quadratic speedup** compared to classical Monte Carlo.

Advantages:

- Requires **fewer samples** for similar precision  
- Scales better for large binomial trees  
- Naturally handles superposition → simulating many price paths at once

Even on a simulator, our tests show:

### 🔥 The quantum binomial model converges faster than the classical binomial.

---

## 📂 Repository Structure

