# Game-Theoretic Model for Container Terminal Cooperation

## Abstract

This document presents the mathematical formulation and solution algorithms for analyzing container terminal cooperation using cooperative game theory. The model extends traditional terminal cooperation optimization by incorporating Core analysis and Shapley value calculations to assess cooperation stability and fair profit allocation.

---

## 1. Problem Definition

### 1.1 Basic Setup

Consider a set of container terminals $I = \{1, 2, \ldots, n\}$ and a set of vessels $J = \{1, 2, \ldots, m\}$. Each terminal $i \in I$ has:
- Capacity $C_i$
- Initial vessel assignments from the baseline (no-cooperation) scenario
- Cost and revenue functions that depend on utilization

### 1.2 Game-Theoretic Framework

The terminal cooperation problem is modeled as a cooperative game $(I, v)$ where:
- $I$ is the set of players (terminals)
- $v: 2^I \rightarrow \mathbb{R}$ is the characteristic function that assigns a value to each coalition $S \subseteq I$

---

## 2. Mathematical Formulation

### 2.1 Coalition Optimization Problem

For any coalition $S \subseteq I$, we solve the following Mixed-Integer Nonlinear Program (MINLP):

**Decision Variables:**
- $x_{ji} \in \{0,1\}$: Binary variable indicating if vessel $j$ is assigned to terminal $i$
- $Q_i \geq 0$: Total volume handled by terminal $i$
- $u_i \in [0,1]$: Utilization rate of terminal $i$
- $TC_i \geq 0$: Total production cost of terminal $i$
- $P_i$: Total profit of terminal $i$
- $F_i \geq 0$: Transfer fee per TEU for terminal $i$ (if optimized pricing)

**Objective Function:**
$$\max \sum_{i \in S} P_i$$

**Constraints:**

*Vessel Assignment:*
$$\sum_{i \in I} x_{ji} = 1 \quad \forall j \in J$$

*Volume Calculation:*
$$Q_i = \sum_{j \in J} V_j \cdot x_{ji} \quad \forall i \in I$$

*Utilization Definition:*
$$u_i \cdot C_i = Q_i \quad \forall i \in I$$

*Minimum Volume Constraint:*
$$Q_i \geq 0.1 \cdot C_i \quad \forall i \in I$$

*Production Cost (Piecewise Linear):*
$$TC_i = \text{PWL}(u_i; \text{cost parameters}) \quad \forall i \in I$$

*Coalition Constraint (Key Difference):*
$$x_{ji} = x_{ji}^{baseline} \quad \forall j \in J, \forall i \notin S$$

This constraint ensures that terminals not in coalition $S$ maintain their original vessel assignments.

*Transfer Restrictions:*
- Vessel transfers are only allowed between terminals within coalition $S$
- Terminals outside $S$ cannot participate in transfer fee mechanisms

*Profit Calculation:*
$$P_i = R_i^{baseline} + P_i^{transfer} - (TC_i - TC_i^{baseline}) \quad \forall i \in I$$

where $P_i^{transfer}$ represents net income from vessel transfers (non-zero only for $i \in S$).

*Participation Constraint:*
$$P_i \geq 0.99 \cdot P_i^{baseline} \quad \forall i \in I$$

### 2.2 Characteristic Function

The characteristic function $v(S)$ for coalition $S \subseteq I$ is defined as:

$$v(S) = \sum_{i \in S} P_i^*(S)$$

where $P_i^*(S)$ is the optimal profit of terminal $i$ when coalition $S$ cooperates, obtained by solving the coalition optimization problem.

**Special Cases:**
- $v(\emptyset) = 0$ (empty coalition has no value)
- $v(\{i\}) = P_i^{baseline}$ (singleton coalitions receive baseline profit)
- $v(I)$ is the grand coalition value (full cooperation)

---

## 3. Core Analysis

### 3.1 Core Definition

The Core of the game is the set of profit allocations $\pi = (\pi_1, \pi_2, \ldots, \pi_n)$ such that:

1. **Efficiency:** $\sum_{i \in I} \pi_i = v(I)$
2. **Individual Rationality:** $\pi_i \geq v(\{i\}) = P_i^{baseline} \quad \forall i \in I$
3. **Coalition Rationality:** $\sum_{i \in S} \pi_i \geq v(S) \quad \forall S \subseteq I, S \neq \emptyset$

### 3.2 Core Membership Algorithm

**Input:** Characteristic function $v$, proposed allocation $\pi$

**Algorithm:**
```
1. Initialize violations = []
2. For each S ⊆ I, S ≠ ∅, S ≠ I:
   a. Calculate coalition_sum = Σ(i∈S) π_i
   b. If v(S) > coalition_sum + ε:
      i. Record violation: (S, v(S), coalition_sum, improvement)
3. Return:
   - in_core = (violations is empty)
   - violation_details = violations
   - stability_score = 1 - |violations|/(2^n - 2)
```

**Computational Complexity:** $O(2^n)$ coalition checks

---

## 4. Shapley Value Analysis

### 4.1 Shapley Value Definition

The Shapley value for player $i$ is:

$$\phi_i(v) = \sum_{S \subseteq I \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

where the sum is over all coalitions $S$ that do not contain player $i$.

### 4.2 Shapley Value Algorithm

**Input:** Characteristic function $v$, number of terminals $n$

**Algorithm:**
```
For each terminal i ∈ I:
1. Initialize shapley_value_i = 0
2. For each coalition_size s = 0 to n-1:
   a. For each coalition S of size s not containing i:
      i. Calculate marginal_contribution = v(S ∪ {i}) - v(S)
      ii. Calculate weight = s! × (n-s-1)! / n!
      iii. Add weight × marginal_contribution to shapley_value_i
3. Return shapley_value_i
```

**Properties:**
- **Efficiency:** $\sum_{i \in I} \phi_i(v) = v(I)$
- **Symmetry:** Equal treatment of symmetric players
- **Dummy:** Dummy players receive zero value
- **Additivity:** $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$

**Computational Complexity:** $O(n \cdot 2^n)$

---

## 5. Solution Algorithm

### 5.1 Overall Algorithm Structure

**Input:** Terminal data, vessel data, cooperation parameters

**Main Algorithm:**
```
1. DATA_VALIDATION()
2. baseline_state = CALCULATE_BASELINE_STATE()
3. 
4. // Generate all possible coalitions
5. coalitions = GENERATE_ALL_COALITIONS(n)
6. 
7. // Calculate characteristic function
8. characteristic_function = {}
9. For each coalition S in coalitions:
10.    result = SOLVE_COALITION_OPTIMIZATION(S)
11.    characteristic_function[S] = EXTRACT_COALITION_VALUE(result, S)
12. 
13. // Solve grand coalition
14. grand_coalition_result = SOLVE_COALITION_OPTIMIZATION(I)
15. π = EXTRACT_PROFIT_ALLOCATION(grand_coalition_result)
16. 
17. // Core analysis
18. core_analysis = CHECK_CORE_MEMBERSHIP(characteristic_function, π)
19. 
20. // Shapley value calculation
21. shapley_values = CALCULATE_SHAPLEY_VALUES(characteristic_function, n)
22. 
23. Return {characteristic_function, core_analysis, shapley_values, π}
```

### 5.2 Computational Complexity Analysis

**Total Optimization Problems:** $2^n - 1$ (all non-empty coalitions)

**Per-Problem Complexity:** 
- Variables: $O(mn + n)$ 
- Constraints: $O(mn + n)$
- Solution time: Depends on solver and problem size

**Overall Complexity:** $O(2^n \cdot T_{MINLP})$ where $T_{MINLP}$ is the time to solve one MINLP

**Memory Requirements:** $O(2^n)$ to store characteristic function

### 5.3 Scalability Considerations

| Terminals | Coalitions | Practical Feasibility |
|-----------|------------|----------------------|
| 3         | 7          | Trivial              |
| 4         | 15         | Easy                 |
| 5         | 31         | Moderate             |
| 6         | 63         | Challenging          |
| 7         | 127        | Difficult            |
| 8+        | 255+       | Impractical          |

---

## 6. Implementation Considerations

### 6.1 Numerical Stability

- **Tolerance Levels:** Use $\epsilon = 10^{-6}$ for Core violation detection
- **Solver Settings:** Set appropriate optimality gaps and time limits
- **Fallback Mechanisms:** Use baseline profits when optimization fails

### 6.2 Parallelization Opportunities

- **Coalition Independence:** Coalition optimization problems can be solved in parallel
- **Load Balancing:** Distribute coalitions across processors by size
- **Memory Management:** Stream results to disk for large problems

### 6.3 Approximation Methods

For larger problems, consider:
- **Sampling:** Randomly sample coalitions instead of exhaustive enumeration
- **Monte Carlo Shapley:** Estimate Shapley values using Monte Carlo simulation
- **Core Approximation:** Check only "critical" coalitions for Core violations

---

## 7. Interpretation of Results

### 7.1 Core Analysis Interpretation

- **In Core:** Cooperation arrangement is stable; no coalition has incentive to defect
- **Core Violations:** Identifies specific coalitions that can improve by leaving
- **Stability Score:** Quantifies overall stability (0 = completely unstable, 1 = perfectly stable)

### 7.2 Shapley Value Interpretation

- **Fair Allocation:** Represents each terminal's "fair share" based on marginal contributions
- **Comparison with Cooperation:** Shows whether current allocation over/under-compensates terminals
- **Implementation Guide:** Shapley values suggest transfer payment adjustments

### 7.3 Policy Implications

- **Subsidy Design:** Use Shapley values to design targeted subsidies
- **Mechanism Selection:** Choose pricing mechanisms that approximate Shapley allocation
- **Coalition Formation:** Identify most valuable terminal combinations

---

## 8. Limitations and Extensions

### 8.1 Current Limitations

- **Computational Scalability:** Exponential complexity limits practical application
- **Static Analysis:** No consideration of dynamic or repeated interactions
- **Perfect Information:** Assumes complete knowledge of all parameters
- **Transferable Utility:** Assumes side payments are possible

### 8.2 Potential Extensions

- **Dynamic Games:** Incorporate multi-period cooperation
- **Incomplete Information:** Robust analysis under uncertainty
- **Non-Transferable Utility:** Alternative solution concepts
- **Bargaining Models:** Nash bargaining and other negotiation frameworks

---

## References

1. Shapley, L. S. (1953). A value for n-person games. *Contributions to the Theory of Games*, 2(28), 307-317.
2. Gillies, D. B. (1953). Some theorems on n-person games. *Princeton University Press*.
3. Owen, G. (1995). *Game Theory* (3rd ed.). Academic Press.
4. Maschler, M., Solan, E., & Zamir, S. (2013). *Game Theory*. Cambridge University Press.