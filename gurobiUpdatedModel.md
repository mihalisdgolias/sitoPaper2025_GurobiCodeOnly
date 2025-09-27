Here is the detailed documentation for the **Unified Terminal Cooperation Optimization Model**, including the mathematical formulation, nomenclature, and specifics of both the Gurobi and Pyomo implementations.

# Unified Container Terminal Cooperation Optimization Model

This model is a **Mixed-Integer Non-Linear Program (MINLP)** designed to find the optimal assignment of vessels to terminals to maximize port-wide profit (MAXPROF) or maximize minimum terminal profit (MAXMIN), incorporating a piecewise cost function and a flexible transfer pricing scheme.

***

## 1. Nomenclature

### Sets and Indices

| Symbol | Description |
| :---: | :--- |
| $\mathcal{I}$ | Set of container terminals, indexed by $i$. |
| $\mathcal{J}$ | Set of vessels (cargo assignments), indexed by $j$. |

### Parameters (Input Data & Calculated Baseline)

| Symbol | Description |
| :---: | :--- |
| $V_j$ | Volume of cargo for vessel $j$ (TEU). |
| $C_i$ | Maximum handling capacity of terminal $i$ (TEU/week). |
| $u_{opt,i}$ | Optimal utilization point where marginal cost is minimized. |
| $MC_{start,i}$ | Initial Marginal Cost (MC) at $u=0$ for terminal $i$. |
| $\alpha_{cost,i}$ | Slope of MC decrease in Cost Phase 1 ($\text{slope1}$). |
| $\beta_{cost,i}$ | Slope of MC increase in Cost Phase 2 ($\text{slope2}$). |
| $S$ | CI capability subsidy ($\$/\text{TEU}$). |
| $\Gamma_i$ | Binary: 1 if terminal $i$ is CI-capable. |
| $I_{j,i}^{CI}$ | Binary: 1 if vessel $j$ is CI-capable and initially assigned to $i$. |
| $x_{j,i}^{before}$ | Binary: Initial vessel assignment. |
| $R_{before,i}$ | Terminal $i$'s fixed **Total Revenue** in the initial state. |
| $TC_{before,i}$ | Terminal $i$'s **Total Cost** in the initial state. |
| $P_{before,i}$ | Terminal $i$'s **Profit** in the initial state. |

### Decision Variables

| Symbol | Description | Domain |
| :---: | :--- | :--- |
| $x_{ji}$ | Vessel assignment: 1 if vessel $j$ is assigned to $i$. | Binary ($\{0, 1\}$) |
| $Q_i$ | Total cargo volume handled by terminal $i$ (TEU). | $\mathbb{R}^+$ |
| $u_i$ | Utilization rate of terminal $i$ ($Q_i / C_i$). | $\mathbb{R} \in [0, 1]$ |
| $TC_i$ | Total Production Cost of terminal $i$ (after optimization). | $\mathbb{R}^+$ |
| $P_i$ | Total Profit of terminal $i$ (after cooperation and transfers). | $\mathbb{R}$ |
| $F_i$ | Transfer Fee ($\$/\text{TEU}$) set by terminal $i$. | $\mathbb{R} \in [0, 1000]$ |
| $P^{min}$ | Minimum profit achieved by any terminal (MAXMIN only). | $\mathbb{R}$ |

***

## 2. Mathematical Formulation

The model is formulated as a Mixed-Integer Non-Linear Program (MINLP).

### 2.1. Objectives

The model is solved for two alternative objectives:

#### A. Total Profit Maximization (MAXPROF)
$$\text{Maximize: } Z_{\text{MAXPROF}} = \sum_{i \in \mathcal{I}} P_i$$

#### B. Max-Min Profit (MAXMIN)
$$\text{Maximize: } Z_{\text{MAXMIN}} = P^{min}$$
Subject to: $P^{min} \le P_i \quad \forall i \in \mathcal{I}$

### 2.2. Core Function Definitions (Expressions)

#### i. Piecewise Total Cost $TC_i(u_i)$
The total production cost is defined by the integral of the piecewise marginal cost function, $MC_i(u)$.

**Marginal Cost (MC) Definition:**
$$MC_i(u) = \begin{cases} MC_{start,i} - \alpha_{cost,i}u & \text{if } u \le u_{opt,i} \\ MC_{min,i} + \beta_{cost,i}(u - u_{opt,i}) & \text{if } u > u_{opt,i} \end{cases}$$
where $MC_{min,i} = MC_{start,i} - \alpha_{cost,i}u_{opt,i}$.

**Total Cost (TC) Formulation:**
$$TC_i = \text{PWL}_{\text{Gurobi}}(u_i) \quad \text{or} \quad \text{Sigmoid}_{\text{Pyomo}}(u_i)$$

#### ii. Net Profit from Transfers $P_{transfer,i}$
This term accounts for all incoming and outgoing transfer fees, plus CI subsidies.

$$P_{transfer,i} = \sum_{j \in \mathcal{J}} \sum_{k \in \mathcal{I}, k \ne i} \left[ x_{ji} x_{jk}^{before} \cdot V_j \cdot (F_k + \Gamma_i S I_{j,k}^{CI}) \right] - \sum_{j \in \mathcal{J}} \sum_{k \in \mathcal{I}, k \ne i} \left[ x_{kj} x_{ji}^{before} \cdot V_j \cdot (F_i + \Gamma_i S I_{j,i}^{CI}) \right]$$

#### iii. Final Profit $P_i$
The profit is the initial revenue plus the net income from transfers, minus the change in production cost.

$$P_i = R_{before,i} + P_{transfer,i} - (TC_i - TC_{before,i})$$

### 2.3. Constraints

| ID | Description | Mathematical Formulation |
| :---: | :--- | :--- |
| **C1** | **Vessel Assignment** | $\sum_{i \in \mathcal{I}} x_{ji} = 1 \quad \forall j \in \mathcal{J}$ |
| **C2** | **Volume Calculation** | $Q_i = \sum_{j \in \mathcal{J}} V_j \cdot x_{ji} \quad \forall i \in \mathcal{I}$ |
| **C3** | **Utilization Definition** | $u_i \cdot C_i = Q_i \quad \forall i \in \mathcal{I}$ |
| **C4** | **Volume Conservation** | $\sum_{i \in \mathcal{I}} Q_i = \sum_{j \in \mathcal{J}} V_j$ |
| **C5** | **Minimum Volume** | $Q_i \ge 0.1 \cdot C_i \quad \forall i \in \mathcal{I}$ |
| **C6** | **Profit Stability** | $P_i \ge 0.99 \cdot P_{before,i} \quad \forall i \in \mathcal{I}$ |

***

## 3. Solver Implementation Specifics

The model is solved using two specialized approaches to handle the non-linearity and binary variables.

### 3.1. Gurobi Implementation (Exact MINLP)

* **Cost Link:** The relationship between $u_i$ and $TC_i$ is modeled precisely using Gurobi's **General Constraint for Piecewise Linear Functions (`model.addGenConstrPWL`)**. This is the most accurate method for the intended cost structure.
* **Pricing:**
    * *Marginal Cost (MC)* pricing is also modeled using `GenConstrPWL` to link the fee $F_i$ to the current $u_i$.
    * *Optimized* pricing treats $F_i$ as a continuous optimization variable bounded by $\left[0, 1000\right]$.

### 3.2. Pyomo Implementation (Sigmoid Approximation)

* **Solver:** Requires an MINLP-capable solver like **BONMIN**, or, less robustly, **IPOPT**.
* **Cost Link:** The exact piecewise function is replaced by a continuous, differentiable **Sigmoid approximation** to allow standard NLP solvers to operate:

    $$TC_i(u_i) \approx TC_{i}^{\text{Phase 1}} \cdot (1 - \sigma(u_i)) + TC_{i}^{\text{Phase 2}} \cdot \sigma(u_i)$$

    where the Sigmoid function $\sigma(u)$ has a steepness parameter $k=10000$ to ensure a sharp transition near $u_{opt}$. This compromise sacrifices absolute accuracy for solver stability.
* **MAXMIN:** The MAXMIN objective is typically only implemented for MAXPROF in the Pyomo version due to the added complexity of the MINLP formulation.