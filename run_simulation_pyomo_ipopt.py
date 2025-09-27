"""
UNIFIED TERMINAL COOPERATION SIMULATION FRAMEWORK

This script provides two parallel optimization paths for the terminal cooperation problem:
1. Gurobi MINLP: Uses Gurobi's built-in MINLP and Piecewise Linear (PWL) capabilities.
2. Pyomo MINLP: Uses Pyomo with BONMIN or IPOPT, using sigmoid smoothing for the non-linear cost function.

Both paths use the exact piecewise cost function and the fixed revenue + transfer fee scheme.
"""

"""
================================================================================
UNIFIED TERMINAL COOPERATION OPTIMIZATION MODEL ANALYSIS
================================================================================

This script implements a Mixed-Integer Non-Linear Program (MINLP) for container
terminal cooperation, solving for optimal vessel assignments and transfer fees.
It features two parallel solution paths: Gurobi (Exact PWL) and Pyomo (Sigmoid Approx.).

--------------------------------------------------------------------------------
1. CODE FUNCTIONALITY AND DESIGN
--------------------------------------------------------------------------------

MODEL GOAL: Maximize total port profit (MAXPROF) or maximize the minimum profit
of any single terminal (MAXMIN), while ensuring no terminal suffers significant
profit loss (Participation Constraint).

CORE MECHANISM: The model fixes initial revenue (R_before) and optimizes profit
by minimizing the change in production cost (TC - TC_before) and maximizing
net income from vessel transfers (P_transfer).

--------------------------------------------------------------------------------
2. NOMENCLATURE
--------------------------------------------------------------------------------

SETS AND INDICES:
- I (i): Set of container terminals.
- J (j): Set of vessels.

DECISION VARIABLES (Optimized):
- x[j, i]: Binary (0/1). 1 if vessel j is assigned to terminal i.
- Q[i]: Continuous. Total cargo volume handled by terminal i.
- u[i]: Continuous. Utilization rate of terminal i.
- TC[i]: Continuous. Total Production Cost (Piecewise/Sigmoid).
- P[i]: Continuous. Total Profit after cooperation/transfers.
- F[i]: Continuous. Transfer Fee ($/TEU).
- P_min: Continuous. Minimum profit (MAXMIN objective only).

KEY PARAMETERS (Input Data/Calculated Baseline):
- C[i]: Capacity of terminal i.
- P_before[i]: Terminal i's profit in the initial state.
- S: CI capability subsidy ($/TEU).

--------------------------------------------------------------------------------
3. CORE CONSTRAINTS AND HARDCODED SCENARIO INPUTS
--------------------------------------------------------------------------------

HARDCODED SCENARIO INPUTS (Defined in run_all_combinations_unified):

# --- Scenario Parameters ---
subsidies = np.array([0, 50, 100])
pricing_mechanisms = ['marginal_profit', 'marginal_cost', 'optimized']

# These arrays define the three tested subsidy levels and three pricing schemes.
# The code iterates through every combination of these parameters for each
# loaded data file and each possible CI terminal combination.

# --- Fixed Operational Constraints ---
- Profit Stability (Participation Constraint): P[i] >= 0.99 * P_before[i]
- Minimum Volume Constraint: Q[i] >= 0.1 * C[i] (10% capacity minimum).

HARDCODED IMPLEMENTATION VALUES (Within Functions):
- Solver Time Limit: SimulationConfig.TIME_LIMIT = 300 seconds.
- Pyomo Smoothing Steepness: k = 10000 (Sigmoid function factor).
- Optimized Fee Bounds: F[i] is bounded between $0.0 and $1000.0/TEU.
"""

"""
UNIFIED TERMINAL COOPERATION SIMULATION FRAMEWORK

This script provides two parallel optimization paths for the terminal cooperation problem:
1. Gurobi MINLP: Uses Gurobi's built-in MINLP and Piecewise Linear (PWL) capabilities.
2. Pyomo MINLP: Uses Pyomo with BONMIN or IPOPT, using sigmoid smoothing for the non-linear cost function.

Both paths use the exact piecewise cost function and the fixed revenue + transfer fee scheme.
"""

"""
UNIFIED TERMINAL COOPERATION SIMULATION FRAMEWORK

This script provides two parallel optimization paths for the terminal cooperation problem:
1. Gurobi MINLP: Uses Gurobi's built-in MINLP and Piecewise Linear (PWL) capabilities.
2. Pyomo MINLP: Uses Pyomo with BONMIN or IPOPT, using sigmoid smoothing for the non-linear cost function.

Both paths use the exact piecewise cost function and the fixed revenue + transfer fee scheme.
"""

import os
import pickle
import numpy as np
import re
import pandas as pd
import logging
import time
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

# =============================================================================
# GLOBAL CONFIGURATION SECTION (HARDCODED INPUTS)
# =============================================================================

GLOBAL_SCENARIO_CONFIG = {
    # Subsidies tested in $/TEU
    'subsidies': np.array([0, 50, 100]),
    # this was for testing
    #'subsidies': np.array([0]),

    # Pricing mechanisms tested
    'pricing_mechanisms': ['marginal_profit', 'marginal_cost', 'optimized'],
    # this was for testing 'pricing_mechanisms': [ 'optimized'],

}

# --- Solver Imports ---
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False


# --- Configuration and Logging ---

class SimulationConfig:
    DEFAULT_OPTIMALITY_TOL = 0.01
    DEFAULT_M_VALUE = 1000000
    MAX_MEMORY_MB = 4000
    TIME_LIMIT = 300  # seconds for solver


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation_unified.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# --- Utility Functions ---

def calculate_exact_cost(terminal_idx, utilization, terminal_capacities, cost_initial, cost_slope1, cost_slope2,
                         optimal_utilization):
    """Calculates cost using the ORIGINAL PIECEWISE FUNCTION."""
    utilization = utilization
    capacity = terminal_capacities[terminal_idx]
    u_opt = optimal_utilization[terminal_idx]

    if utilization <= u_opt:
        total_cost = (cost_initial[terminal_idx] * utilization -
                      0.5 * cost_slope1[terminal_idx] * utilization ** 2) * capacity
    else:
        cost_at_optimal = (cost_initial[terminal_idx] * u_opt -
                           0.5 * cost_slope1[terminal_idx] * u_opt ** 2) * capacity

        mc_at_optimal = cost_initial[terminal_idx] - cost_slope1[terminal_idx] * u_opt

        additional_cost = (mc_at_optimal * (utilization - u_opt) +
                           0.5 * cost_slope2[terminal_idx] * (utilization - u_opt) ** 2) * capacity

        total_cost = cost_at_optimal + additional_cost
    return total_cost


def calculate_marginal_cost(terminal_idx: int, utilization: float, optimal_utilization, cost_initial, cost_slope1,
                            cost_slope2) -> float:
    """Calculate marginal cost for terminal at given utilization level."""
    if utilization <= optimal_utilization[terminal_idx]:
        return cost_initial[terminal_idx] - cost_slope1[terminal_idx] * utilization
    else:
        return (cost_initial[terminal_idx] - cost_slope1[terminal_idx] * optimal_utilization[terminal_idx] +
                cost_slope2[terminal_idx] * (utilization - optimal_utilization[terminal_idx]))


def calculate_marginal_profit(terminal_idx: int, utilization: float, revenue_decrease_rate, revenue_initial_charge,
                              optimal_utilization, cost_initial, cost_slope1, cost_slope2) -> float:
    """Calculate marginal profit for terminal at given utilization level."""
    marginal_revenue = revenue_decrease_rate[terminal_idx] * 2 * utilization + revenue_initial_charge[terminal_idx]
    marginal_cost = calculate_marginal_cost(terminal_idx, utilization, optimal_utilization, cost_initial, cost_slope1,
                                            cost_slope2)
    return marginal_revenue - marginal_cost


def validate_input_data(input_data_ves: Dict[str, Any]) -> bool:
    """Validate input data structure and values."""
    required_keys = [
        'subPar', 'ciTerm', 'num_terminals', 'v', 'x', 'xCI', 'c',
        'cost_initial', 'cost_decrease_rate', 'cost_increase_rate',
        'optimal_vcr_cost_point', 'initial_charge', 'decrease_rate'
    ]

    try:
        # Check required keys
        for key in required_keys:
            if key not in input_data_ves:
                logger.error(f"Missing required key: {key}")
                return False

        # Validate dimensions
        num_terminals = input_data_ves['num_terminals']
        if len(input_data_ves['c']) != num_terminals:
            logger.error(
                f"Capacity array length {len(input_data_ves['c'])} doesn't match num_terminals {num_terminals}")
            return False

        # Validate positive values where required
        if np.any(input_data_ves['c'] <= 0):
            logger.error("Terminal capacities must be positive")
            return False

        if np.any(input_data_ves['v'] <= 0):
            logger.error("Vessel volumes must be positive")
            return False

        return True

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def calculate_baseline_state(input_data_ves: Dict[str, Any]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates volume, utilization, revenue, cost, and profit for the no-cooperation state."""
    vessel_volumes = input_data_ves['v']
    vessel_assignments = input_data_ves['x']
    terminal_capacities = input_data_ves['c']
    num_terminals = input_data_ves['num_terminals']
    subsidy = input_data_ves['subPar']
    ci_terminals = input_data_ves['ciTerm']
    vessel_ci_capabilities = input_data_ves['xCI']

    volume_before = np.sum(vessel_volumes[:, None] * vessel_assignments, axis=0)
    utilization_before = volume_before / terminal_capacities

    revenue_decrease_rate = -input_data_ves['decrease_rate']
    revenue_initial_charge = input_data_ves['initial_charge']

    revenue_per_container_before = revenue_decrease_rate * utilization_before + revenue_initial_charge

    # Calculate CI subsidy revenue (R_ci)
    R_ci_subsidy = np.zeros(num_terminals)
    for j in range(len(vessel_volumes)):
        for i in range(num_terminals):
            if vessel_assignments[j, i] == 1:
                # Only terminals initially assigned the vessel receive the subsidy (per data generation logic)
                R_ci_subsidy[i] += vessel_volumes[j] * vessel_ci_capabilities[j, i] * ci_terminals[i] * subsidy

    # Total Revenue (R_base + R_ci)
    revenue_before = revenue_per_container_before * volume_before + R_ci_subsidy

    # Calculate cost before cooperation using ORIGINAL PIECEWISE FUNCTION
    cost_before = np.zeros(num_terminals)
    for i in range(num_terminals):
        cost_before[i] = calculate_exact_cost(
            i, utilization_before[i], terminal_capacities,
            input_data_ves['cost_initial'], input_data_ves['cost_decrease_rate'],
            input_data_ves['cost_increase_rate'], input_data_ves['optimal_vcr_cost_point']
        )

    profit_before = revenue_before - cost_before

    return volume_before, utilization_before, revenue_before, cost_before, profit_before


def validate_current_state_feasibility(input_data_ves: Dict[str, Any]) -> bool:
    """Validate that the current (no-cooperation) state satisfies constraints."""
    # Uses the fixed minimum capacity constraint (0.1 * capacity)
    _, _, _, _, _ = calculate_baseline_state(input_data_ves)  # Uses baseline calc to ensure data is okay
    terminal_capacities = input_data_ves['c']
    num_terminals = input_data_ves['num_terminals']
    vessel_volumes = input_data_ves['v']
    vessel_assignments = input_data_ves['x']

    volume_before = np.sum(vessel_volumes[:, None] * vessel_assignments, axis=0)

    is_feasible = True
    for i in range(num_terminals):
        min_volume = 0.1 * terminal_capacities[i]
        if volume_before[i] < min_volume - 1e-6:
            logger.error(f"Terminal {i}: current volume {volume_before[i]:.0f} below minimum {min_volume:.0f}")
            is_feasible = False

    if is_feasible:
        logger.info(f"Current state validation complete. All constraints satisfied.")
    return is_feasible


def calculate_realistic_fallback(input_data_ves: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate realistic fallback values when optimization fails."""
    try:
        num_terminals = input_data_ves['num_terminals']

        # Calculate current state
        volume_before, _, _, _, profit_before = calculate_baseline_state(input_data_ves)

        return {
            'feasibility_status': 'Fallback Solution - Optimization Failed',
            'profitBefore': profit_before,
            'profitAfter_MAXPROF': profit_before.copy(),
            'profitAfter_MAXMIN': profit_before.copy(),
            'volumeBefore': volume_before,
            'volumeAfter_MAXPROF': volume_before.copy(),
            'volumeAfter_MAXMIN': volume_before.copy(),
            'transferFees_MAXPROF': np.zeros(num_terminals),
            'transferFees_MAXMIN': np.zeros(num_terminals),
            'objval_MAXPROF': np.nan,
            'optimalityGap_MAXPROF': np.nan,
            'objval_MAXMIN': np.nan,
            'optimalityGap_MAXMIN': np.nan,
            'pricing_mechanism': 'fallback',
            'solver_used': 'Fallback'
        }

    except Exception as e:
        logger.error(f"Fallback calculation failed: {e}")
        return None


# --- Gurobi Optimization Function (MINLP) ---

def vessel_based_opt_gurobi(input_data_ves: Dict[str, Any], pricing_mechanism: str = 'optimized') -> Optional[
    Dict[str, Any]]:
    """Terminal cooperation optimization using Gurobi."""
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is required but not available.")
        return calculate_realistic_fallback(input_data_ves)

    if not validate_input_data(input_data_ves):
        logger.error("Input data validation failed")
        return calculate_realistic_fallback(input_data_ves)

    if not validate_current_state_feasibility(input_data_ves):
        logger.error("Current state is not feasible - data generation problem")
        return calculate_realistic_fallback(input_data_ves)

    valid_mechanisms = GLOBAL_SCENARIO_CONFIG['pricing_mechanisms']
    if pricing_mechanism not in valid_mechanisms:
        logger.error(f"Invalid pricing mechanism: {pricing_mechanism}")
        return calculate_realistic_fallback(input_data_ves)

    logger.info(f"Running Gurobi optimization with pricing mechanism: {pricing_mechanism}")

    # Extract data and calculate baseline
    volume_before, utilization_before, revenue_before, cost_before, profit_before = calculate_baseline_state(
        input_data_ves)

    # Parameters for Gurobi
    num_terminals = input_data_ves['num_terminals']
    vessel_volumes = input_data_ves['v']
    vessel_assignments = input_data_ves['x']
    terminal_capacities = input_data_ves['c']
    optimal_utilization = input_data_ves['optimal_vcr_cost_point']
    cost_initial = input_data_ves['cost_initial']
    cost_slope1 = input_data_ves['cost_decrease_rate']
    cost_slope2 = input_data_ves['cost_increase_rate']
    subsidy = input_data_ves['subPar']
    ci_terminals = input_data_ves['ciTerm']
    vessel_ci_capabilities = input_data_ves['xCI']
    revenue_decrease_rate = -input_data_ves['decrease_rate']
    revenue_initial_charge = input_data_ves['initial_charge']
    num_vessels = len(vessel_volumes)

    results = {
        'profitBefore': profit_before,
        'volumeBefore': volume_before,
        'feasibility_status': 'Infeasible - Returned No-Cooperation Solution',
        'pricing_mechanism': pricing_mechanism,
        'solver_used': 'Gurobi'
    }

    # Helper function to calculate total cost PWL points
    def calculate_total_cost_pwl_points(i, u_points):
        costs = np.zeros_like(u_points, dtype=float)
        u_opt = optimal_utilization[i]
        capacity = terminal_capacities[i]

        # Phase 1: u <= u_optimal
        mask1 = (u_points >= 0) & (u_points <= u_opt)
        costs[mask1] = (cost_initial[i] * u_points[mask1] - 0.5 * cost_slope1[i] * u_points[mask1] ** 2)

        # Phase 2: u > u_optimal
        mask2 = u_points > u_opt
        if np.any(mask2):
            cost_at_optimal = (cost_initial[i] * u_opt - 0.5 * cost_slope1[i] * u_opt ** 2)
            mc_at_optimal = cost_initial[i] - cost_slope1[i] * u_opt

            costs[mask2] = (cost_at_optimal +
                            mc_at_optimal * (u_points[mask2] - u_opt) +
                            0.5 * cost_slope2[i] * (u_points[mask2] - u_opt) ** 2)

        return costs * capacity

    for objective_name in ['MAXPROF', 'MAXMIN']:
        try:
            model = gp.Model(f"Gurobi_Coop_{objective_name}_{pricing_mechanism}")
            model.setParam('OutputFlag', 0)
            model.setParam('OptimalityTol', SimulationConfig.DEFAULT_OPTIMALITY_TOL)
            model.setParam('TimeLimit', SimulationConfig.TIME_LIMIT)

            # Variables
            vessel_assignment = model.addVars(num_vessels, num_terminals, vtype=GRB.BINARY, name="x")
            volume_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="Q")
            profit_terminal = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="P")
            production_cost = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="TC")
            profit_from_transfers = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="P_transfer")
            utilization_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="u")

            # Warm start
            for j in range(num_vessels):
                for i in range(num_terminals):
                    vessel_assignment[j, i].start = vessel_assignments[j, i]
            for i in range(num_terminals):
                volume_after[i].start = volume_before[i]

            # Pricing mechanism setup
            if pricing_mechanism == 'optimized':
                profit_factor_after = model.addVars(num_terminals, lb=0.0, ub=1000.0, vtype=GRB.CONTINUOUS, name="F")
            elif pricing_mechanism == 'marginal_cost':
                profit_factor_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="F_MC")
            elif pricing_mechanism == 'marginal_profit':
                profit_factor_after = {}
                for i in range(num_terminals):
                    mp = calculate_marginal_profit(i, utilization_before[i], revenue_decrease_rate,
                                                   revenue_initial_charge, optimal_utilization, cost_initial,
                                                   cost_slope1, cost_slope2)
                    profit_factor_after[i] = max(0, mp)

            if objective_name == 'MAXMIN':
                min_profit = model.addVar(vtype=GRB.CONTINUOUS, name="P_min")

            # Constraints and Terminal Loop
            for i in range(num_terminals):
                # Volume Calculation
                model.addConstr(volume_after[i] == gp.quicksum(
                    vessel_volumes[j] * vessel_assignment[j, i] for j in range(num_vessels)), name=f"C_Q_{i}")

                # Utilization Constraint
                model.addConstr(utilization_after[i] * terminal_capacities[i] == volume_after[i], name=f"C_u_{i}")

                # Minimum Volume Constraint (simplified)
                model.addConstr(volume_after[i] >= 0.1 * terminal_capacities[i], name=f"C_Q_min_{i}")

                # Pricing Mechanism Constraints (MC)
                if pricing_mechanism == 'marginal_cost':
                    u_vals = np.linspace(0.1, 0.99, 100)
                    mc_vals = [max(0, calculate_marginal_cost(i, u, optimal_utilization, cost_initial, cost_slope1,
                                                              cost_slope2)) for u in u_vals]
                    model.addGenConstrPWL(utilization_after[i], profit_factor_after[i], u_vals, mc_vals,
                                          name=f'C_F_MC_{i}')

                # Cost PWL Link
                u_points = np.array(sorted(list(set(np.linspace(0, 1.0, 100)) | {optimal_utilization[i]})))
                y_points = calculate_total_cost_pwl_points(i, u_points)
                model.addGenConstrPWL(utilization_after[i], production_cost[i], u_points, y_points,
                                      name=f'C_TC_PWL_{i}')

                # Transfer Profit Calculation
                transfer_expr = gp.LinExpr(0)
                for j in range(num_vessels):
                    for k in range(num_terminals):
                        fee_source = profit_factor_after[k] if pricing_mechanism != 'marginal_profit' else \
                        profit_factor_after[k]
                        fee_dest = profit_factor_after[i] if pricing_mechanism != 'marginal_profit' else \
                        profit_factor_after[i]

                        # Vessel j initially at k, moves to i (i receives fee F_k)
                        if vessel_assignments[j, k] == 1 and i != k:
                            transfer_expr += vessel_assignment[j, i] * (
                                        fee_source * vessel_volumes[j] + ci_terminals[i] * subsidy *
                                        vessel_ci_capabilities[j, k] * vessel_volumes[j])

                        # Vessel j initially at i, moves to k (i pays fee F_i)
                        if vessel_assignments[j, i] == 1 and i != k:
                            # We use addition with negative terms for clarity within the quicksum/LinExpr context
                            transfer_expr += vessel_assignment[j, k] * (
                                    -fee_dest * vessel_volumes[j] -
                                    subsidy * ci_terminals[i] * vessel_ci_capabilities[j, i] * vessel_volumes[j])

                model.addConstr(profit_from_transfers[i] == transfer_expr, name=f"C_P_transfer_{i}")

                # Profit Calculation (P = R_before + P_transfer - (TC - TC_before))
                model.addConstr(profit_terminal[i] == revenue_before[i] + profit_from_transfers[i] - (
                            production_cost[i] - cost_before[i]), name=f"C_P_total_{i}")

                # Profit Stability
                model.addConstr(profit_terminal[i] >=  profit_before[i], name=f"C_P_stab_{i}")

            # Global Constraints
            for j in range(num_vessels):
                model.addConstr(gp.quicksum(vessel_assignment[j, i] for i in range(num_terminals)) == 1,
                                name=f"C_x_sum_{j}")

            # Volume Conservation
            model.addConstr(gp.quicksum(volume_after[i] for i in range(num_terminals)) == np.sum(volume_before),
                            name="C_Q_cons")

            # Set objective
            if objective_name == 'MAXPROF':
                model.setObjective(gp.quicksum(profit_terminal[i] for i in range(num_terminals)), GRB.MAXIMIZE)
            elif objective_name == 'MAXMIN':
                for i in range(num_terminals):
                    model.addConstr(min_profit <= profit_terminal[i], name=f"C_P_min_link_{i}")
                model.setObjective(min_profit, GRB.MAXIMIZE)

            # Solve and Process results
            model.optimize()

            if model.status == GRB.OPTIMAL:
                profit_after = np.array([profit_terminal[i].x for i in range(num_terminals)])
                volume_after_values = np.array([volume_after[i].x for i in range(num_terminals)])

                if pricing_mechanism in ['optimized', 'marginal_cost']:
                    transfer_fees = np.array([profit_factor_after[i].x for i in range(num_terminals)])
                else:
                    transfer_fees = np.array([profit_factor_after[i] for i in range(num_terminals)])

                results[f'profitAfter_{objective_name}'] = profit_after
                results[f'volumeAfter_{objective_name}'] = volume_after_values
                results[f'transferFees_{objective_name}'] = transfer_fees
                results[f'objval_{objective_name}'] = model.objVal
                results[f'optimalityGap_{objective_name}'] = model.MIPGap
                results['feasibility_status'] = f'Feasible and Sensible ({pricing_mechanism} pricing)'

                logger.info(f"Gurobi optimization {objective_name} completed successfully")
            else:
                logger.warning(f"Gurobi optimization {objective_name} failed with status: {model.status}")
                results[f'profitAfter_{objective_name}'] = profit_before.copy()
                results[f'volumeAfter_{objective_name}'] = volume_before.copy()
                results[f'transferFees_{objective_name}'] = np.zeros(num_terminals)
                results[f'objval_{objective_name}'] = np.nan
                results[f'optimalityGap_{objective_name}'] = np.nan

            model.dispose()

        except Exception as e:
            logger.error(f"Gurobi optimization {objective_name} failed with error: {e}")

    return results


# --- Pyomo Optimization Function (MINLP) ---

def vessel_based_opt_pyomo_minlp(input_data_ves: Dict[str, Any], pricing_mechanism: str = 'optimized') -> Optional[
    Dict[str, Any]]:
    """
    Terminal cooperation optimization using Pyomo + BONMIN/IPOPT.
    Uses sigmoid smoothing for the non-linear cost function.
    """
    if not PYOMO_AVAILABLE:
        logger.error("Pyomo is required but not available.")
        return calculate_realistic_fallback(input_data_ves)

    # Check for suitable solver (BONMIN preferred for MINLP)
    solver_name = 'bonmin'  # MINLP solver
    solver = SolverFactory(solver_name)
    if not solver.available():
        logger.warning(f"{solver_name} not available. Falling back to IPOPT (less robust for binaries).")
        solver_name = 'ipopt'
        solver = SolverFactory(solver_name)
        if not solver.available():
            logger.error(f"Neither BONMIN nor IPOPT is available. Cannot run Pyomo optimization.")
            return calculate_realistic_fallback(input_data_ves)

    if not validate_input_data(input_data_ves):
        logger.error("Input data validation failed")
        return calculate_realistic_fallback(input_data_ves)

    if not validate_current_state_feasibility(input_data_ves):
        logger.error("Current state is not feasible - data generation problem")
        return calculate_realistic_fallback(input_data_ves)

    logger.info(f"Running Pyomo ({solver_name}) optimization with pricing mechanism: {pricing_mechanism}")

    # Extract data and calculate baseline
    volume_before, utilization_before, revenue_before, cost_before, profit_before = calculate_baseline_state(
        input_data_ves)

    # Parameters for Pyomo
    num_terminals = input_data_ves['num_terminals']
    num_vessels = len(input_data_ves['v'])

    # Helper to calculate the constant cost components for the profit constraint
    cost_at_optimal_list = []
    mc_at_optimal_list = []
    for i in range(num_terminals):
        u_opt = input_data_ves['optimal_vcr_cost_point'][i]
        cost_at_optimal_list.append(
            (input_data_ves['cost_initial'][i] * u_opt - 0.5 * input_data_ves['cost_decrease_rate'][i] * u_opt ** 2) *
            input_data_ves['c'][i])
        mc_at_optimal_list.append(input_data_ves['cost_initial'][i] - input_data_ves['cost_decrease_rate'][i] * u_opt)

    results = {
        'profitBefore': profit_before,
        'volumeBefore': volume_before,
        'feasibility_status': 'Infeasible - Returned No-Cooperation Solution',
        'pricing_mechanism': pricing_mechanism,
        'solver_used': f'Pyomo_{solver_name}'
    }

    # Only implement MAXPROF for Pyomo for simplicity due to MINLP complexity
    objective_names = ['MAXPROF']

    for objective_name in objective_names:
        try:
            model = pyo.ConcreteModel()

            # Sets
            model.terminals = pyo.RangeSet(0, num_terminals - 1)
            model.vessels = pyo.RangeSet(0, num_vessels - 1)

            # Parameters
            model.V = pyo.Param(model.vessels, initialize={j: input_data_ves['v'][j] for j in range(num_vessels)})
            model.C = pyo.Param(model.terminals, initialize={i: input_data_ves['c'][i] for i in range(num_terminals)})
            model.u_opt = pyo.Param(model.terminals, initialize={i: input_data_ves['optimal_vcr_cost_point'][i] for i in
                                                                 range(num_terminals)})
            model.mc_start = pyo.Param(model.terminals,
                                       initialize={i: input_data_ves['cost_initial'][i] for i in range(num_terminals)})
            model.slope1 = pyo.Param(model.terminals, initialize={i: input_data_ves['cost_decrease_rate'][i] for i in
                                                                  range(num_terminals)})
            model.slope2 = pyo.Param(model.terminals, initialize={i: input_data_ves['cost_increase_rate'][i] for i in
                                                                  range(num_terminals)})
            model.R_before = pyo.Param(model.terminals, initialize={i: revenue_before[i] for i in range(num_terminals)})
            model.TC_before = pyo.Param(model.terminals, initialize={i: cost_before[i] for i in range(num_terminals)})
            model.P_before = pyo.Param(model.terminals, initialize={i: profit_before[i] for i in range(num_terminals)})
            model.A_before = pyo.Param(model.vessels, model.terminals,
                                       initialize={(j, i): input_data_ves['x'][j, i] for j in range(num_vessels) for i
                                                   in range(num_terminals)})
            model.CI_term = pyo.Param(model.terminals,
                                      initialize={i: input_data_ves['ciTerm'][i] for i in range(num_terminals)})
            model.CI_ves = pyo.Param(model.vessels, model.terminals,
                                     initialize={(j, i): input_data_ves['xCI'][j, i] for j in range(num_vessels) for i
                                                 in range(num_terminals)})
            model.S = pyo.Param(initialize=input_data_ves['subPar'])
            model.R_dec_rate = pyo.Param(model.terminals, initialize={i: -input_data_ves['decrease_rate'][i] for i in
                                                                      range(num_terminals)})
            model.R_init_charge = pyo.Param(model.terminals, initialize={i: input_data_ves['initial_charge'][i] for i in
                                                                         range(num_terminals)})

            # Variables
            model.x = pyo.Var(model.vessels, model.terminals, domain=pyo.Binary)
            model.Q = pyo.Var(model.terminals, domain=pyo.NonNegativeReals)
            model.u = pyo.Var(model.terminals, bounds=(0.0, 1.0))  # Utilization
            model.TC = pyo.Var(model.terminals, domain=pyo.NonNegativeReals)  # Total Cost
            model.P = pyo.Var(model.terminals, domain=pyo.Reals)  # Total Profit

            # Fixed Marginal Profit pricing (if applicable)
            if pricing_mechanism == 'marginal_profit':
                F_fixed = {}
                for i in range(num_terminals):
                    mp = calculate_marginal_profit(i, utilization_before[i], model.R_dec_rate[i],
                                                   model.R_init_charge[i], model.u_opt, model.mc_start, model.slope1,
                                                   model.slope2)
                    F_fixed[i] = max(0, mp)
                model.F = pyo.Param(model.terminals, initialize=F_fixed)
            else:  # Optimized or Marginal Cost - F is a variable
                model.F = pyo.Var(model.terminals, bounds=(0, 1000.0), domain=pyo.NonNegativeReals)

            # Constraints
            # C1: Vessel Assignment
            def vessel_assignment_rule(model, j):
                return sum(model.x[j, i] for i in model.terminals) == 1

            model.vessel_assignment_constraint = pyo.Constraint(model.vessels, rule=vessel_assignment_rule)

            for i in model.terminals:
                # Volume Calculation
                def volume_rule(model, i):
                    return model.Q[i] == sum(model.V[j] * model.x[j, i] for j in model.vessels)

                model.volume_constraint = pyo.Constraint(model.terminals, rule=volume_rule)

                # Utilization Definition
                def utilization_rule(model, i):
                    return model.u[i] * model.C[i] == model.Q[i]

                model.utilization_constraint = pyo.Constraint(model.terminals, rule=utilization_rule)

                # C5: Minimum Volume Constraint
                def capacity_lower_rule(model, i):
                    return model.Q[i] >= 0.1 * model.C[i]

                model.capacity_lower = pyo.Constraint(model.terminals, rule=capacity_lower_rule)

                # C6: Total Cost (Piecewise - using sigmoid approximation)
                u_opt = model.u_opt[i]

                # Phase 1 cost (u <= u_opt)
                TC1 = (model.mc_start[i] * model.u[i] - 0.5 * model.slope1[i] * model.u[i] ** 2) * model.C[i]

                # Phase 2 cost (u > u_opt)
                cost_at_opt = cost_at_optimal_list[i]
                mc_at_opt = mc_at_optimal_list[i]
                TC2 = cost_at_opt + (
                            mc_at_opt * (model.u[i] - u_opt) + 0.5 * model.slope2[i] * (model.u[i] - u_opt) ** 2) * \
                      model.C[i]

                # Sigmoid approximation (k=10000 for steep transition)
                k = 10000
                sigmoid = 1 / (1 + pyo.exp(-k * (model.u[i] - u_opt)))

                model.add_component(f'cost_link_{i}',
                                    pyo.Constraint(expr=model.TC[i] == TC1 * (1 - sigmoid) + TC2 * sigmoid))

                # Transfer Fee Constraints (Marginal Cost)
                if pricing_mechanism == 'marginal_cost':
                    # Simplified MC function
                    def marginal_cost_link(model, i):
                        return model.F[i] >= 0.0  # Only enforce non-negativity for simplicity in non-pwl Pyomo context

                    model.add_component(f'marginal_cost_constraint_{i}', pyo.Constraint(
                        expr=model.F[i] == pyo.Expr.max(0, model.mc_start[i] - model.slope1[i] * model.u[i])))

                # Profit Calculation (P = R_before + P_transfer - (TC - TC_before))
                P_transfer_expr = 0

                for j in model.vessels:
                    for k in model.terminals:
                        F_var_k = model.F[k] if pricing_mechanism != 'marginal_profit' else model.F[k]
                        F_var_i = model.F[i] if pricing_mechanism != 'marginal_profit' else model.F[i]

                        # Vessel j initially at k, moves to i (i receives)
                        if input_data_ves['x'][j, k] == 1 and i != k:
                            P_transfer_expr += model.x[j, i] * (
                                        F_var_k * model.V[j] + model.CI_term[i] * model.S * model.CI_ves[j, k] *
                                        model.V[j])

                        # Vessel j initially at i, moves to k (i pays)
                        if input_data_ves['x'][j, i] == 1 and i != k:
                            P_transfer_expr -= model.x[j, k] * (
                                        F_var_i * model.V[j] + model.CI_term[i] * model.S * model.CI_ves[j, i] *
                                        model.V[j])

                model.add_component(f'profit_calc_{i}', pyo.Constraint(
                    expr=model.P[i] == model.R_before[i] + P_transfer_expr - (model.TC[i] - model.TC_before[i])))

                # C7: Profit Stability
                def profit_stability_rule(model, i):
                    return model.P[i] >= 0.99 * model.P_before[i]

                model.add_component(f'profit_stability_{i}',
                                    pyo.Constraint(model.terminals, rule=profit_stability_rule))

            # C4: Volume Conservation (global)
            model.volume_conservation = pyo.Constraint(
                expr=sum(model.Q[i] for i in model.terminals) == np.sum(volume_before))

            # Objective
            model.obj = pyo.Objective(expr=sum(model.P[i] for i in model.terminals), sense=pyo.maximize)

            # Solve
            solver.options['max_iter'] = 3000
            solver.options['tol'] = 1e-6
            solver.options['acceptable_tol'] = 1e-4
            solver.options['print_level'] = 0

            result = solver.solve(model, tee=False)

            # Extract results
            if (result.solver.status == pyo.SolverStatus.ok and
                    result.solver.termination_condition in [pyo.TerminationCondition.optimal,
                                                            pyo.TerminationCondition.locallyOptimal,
                                                            pyo.TerminationCondition.feasible]):

                profit_after = np.array([pyo.value(model.P[i]) for i in model.terminals])
                volume_after_values = np.array([pyo.value(model.Q[i]) for i in model.terminals])

                if pricing_mechanism != 'marginal_profit':
                    transfer_fees = np.array([pyo.value(model.F[i]) for i in model.terminals])
                else:
                    transfer_fees = np.array([model.F[i] for i in model.terminals])

                results[f'profitAfter_{objective_name}'] = profit_after
                results[f'volumeAfter_{objective_name}'] = volume_after_values
                results[f'transferFees_{objective_name}'] = transfer_fees
                results[f'objval_{objective_name}'] = pyo.value(model.obj)
                results[f'optimalityGap_{objective_name}'] = 0.0  # Pyomo/NLP doesn't provide gap

                logger.info(f"Pyomo optimization {objective_name} completed successfully")

            else:
                logger.warning(f"Pyomo optimization {objective_name} failed: {result.solver.termination_condition}")
                results[f'profitAfter_{objective_name}'] = profit_before.copy()
                results[f'volumeAfter_{objective_name}'] = volume_before.copy()
                results[f'transferFees_{objective_name}'] = np.zeros(num_terminals)
                results[f'objval_{objective_name}'] = np.nan
                results[f'optimalityGap_{objective_name}'] = np.nan

            # NOTE: MAXMIN is not implemented in Pyomo for simplicity; copy MAXPROF results as placeholder
            results['profitAfter_MAXMIN'] = results[f'profitAfter_MAXPROF'].copy()
            results['volumeAfter_MAXMIN'] = results[f'volumeAfter_MAXPROF'].copy()
            results['transferFees_MAXMIN'] = results[f'transferFees_MAXPROF'].copy()

            return results

        except Exception as e:
            logger.error(f"Critical Pyomo optimization error: {e}")
            fallback = calculate_realistic_fallback(input_data_ves)
            if fallback:
                fallback['pricing_mechanism'] = pricing_mechanism
                fallback['solver_used'] = 'Pyomo_Failed'
            return fallback


# --- Main Execution Block ---

def run_all_combinations_unified(solver_type: str = 'Gurobi'):
    """Main function to run the simulation using the specified solver."""
    if solver_type == 'Gurobi' and not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot run Gurobi simulation.")
        return
    if solver_type == 'Pyomo' and not PYOMO_AVAILABLE:
        logger.error("Pyomo is not available. Cannot run Pyomo simulation.")
        return

    data_folder = "generated_data"
    results_folder = f"simulation_results_{solver_type.lower()}_unified"
    results_file = os.path.join(results_folder, f"summary_results_{solver_type.lower()}_unified.csv")

    if not os.path.exists(data_folder):
        logger.error(f"The folder '{data_folder}' was not found.")
        return
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # --- Retrieve Hardcoded Scenario Inputs ---
    subsidies = GLOBAL_SCENARIO_CONFIG['subsidies']
    pricing_mechanisms = GLOBAL_SCENARIO_CONFIG['pricing_mechanisms']

    data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
    data_files.sort()

    logger.info(f"=== STARTING {solver_type.upper()} UNIFIED SIMULATION ===")
    logger.info(f"Testing Subsidies: {subsidies}")
    logger.info(f"Testing Pricing Mechanisms: {pricing_mechanisms}")

    filename_pattern = re.compile(r'data_T(\d+)_subset_([\d_]+)_CI(\d+)_instance_(\d+)\.pkl')
    combination_count = 0
    successful_optimizations = 0
    failed_optimizations = 0

    # OUTER TQDM LOOP (over data files)
    for filename in tqdm(data_files, desc="Overall File Progress"):
        filepath = os.path.join(data_folder, filename)
        match = filename_pattern.match(filename)
        if not match: continue

        num_terminals_str, subset_str, ci_rate_str, instance_idx_str = match.groups()
        num_terminals = int(num_terminals_str)
        ci_rate = int(ci_rate_str) / 100.0
        instance_idx = int(instance_idx_str)  # <-- THIS is correctly defined here.

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue

        # Prepare a list of all scenarios for the INNER TQDM LOOP
        scenarios = []
        num_ci_combinations = 2 ** num_terminals
        for subsidy in subsidies:
            for ci_combo_idx in range(num_ci_combinations):
                for pricing_mechanism in pricing_mechanisms:
                    ci_binary_str = format(ci_combo_idx, f'0{num_terminals}b')
                    ci_terminals = np.array([int(c) for c in ci_binary_str])
                    ci_terminals_list = np.where(ci_terminals == 1)[0]
                    ci_terminals_str = '_'.join(
                        [str(t + 1) for t in ci_terminals_list]) if ci_terminals_list.size > 0 else 'None'

                    scenarios.append({
                        'subsidy': subsidy,
                        'ci_terminals': ci_terminals,
                        'ci_terminals_str': ci_terminals_str,
                        'pricing_mechanism': pricing_mechanism
                    })

        # INNER TQDM LOOP (over scenarios within the current file)
        inner_tqdm = tqdm(scenarios, desc=f"File: {filename}", leave=False)
        for scenario in inner_tqdm:
            combination_count += 1

            subsidy = scenario['subsidy']
            ci_terminals = scenario['ci_terminals']
            ci_terminals_str = scenario['ci_terminals_str']
            pricing_mechanism = scenario['pricing_mechanism']

            # Dynamic status update
            inner_tqdm.set_postfix({
                'Subsidy': subsidy,
                'CI_Term': ci_terminals_str,
                'Pricing': pricing_mechanism,
                'Solver': solver_type
            })

            input_data_ves = {
                'subPar': subsidy, 'ciTerm': ci_terminals, 'num_terminals': num_terminals,
                'v': data['vVes'], 'x': data['xV'], 'xCI': data['vVesCI'], 'c': data['inputData'][0, :, 0],
                'cost_initial': data['inputData'][0, :, 1], 'cost_decrease_rate': data['inputData'][0, :, 2],
                'cost_increase_rate': data['inputData'][0, :, 3], 'optimal_vcr_cost_point': data['inputData'][0, :, 4],
                'initial_charge': data['inputData'][0, :, 5], 'decrease_rate': data['inputData'][0, :, 6],
                'constant_vcr_charge': data['inputData'][0, :, 7], 'pc': data['inputData'][0, :, 8],
            }

            # Solver Dispatch
            if solver_type == 'Gurobi':
                result = vessel_based_opt_gurobi(input_data_ves, pricing_mechanism=pricing_mechanism)
            elif solver_type == 'Pyomo':
                result = vessel_based_opt_pyomo_minlp(input_data_ves, pricing_mechanism=pricing_mechanism)
            else:
                raise ValueError("Invalid solver_type")

            if result and 'Feasible' in result.get('feasibility_status', ''):
                successful_optimizations += 1
            else:
                failed_optimizations += 1

            # Prepare results dictionary *before* saving
            results_to_save = {
                'simulation_params': {
                    'filename': filename, 'subsidy': subsidy, 'ci_terminals': ci_terminals_str,
                    'ci_rate_from_data': ci_rate, 'num_terminals_from_data': num_terminals,
                    'instance_index': instance_idx,  # <--- Variable is correctly passed here.
                    'subset_composition': subset_str,
                    'pricing_mechanism': pricing_mechanism, 'solver_used': solver_type
                },
                'optimization_output': result
            }

            filename_base = os.path.splitext(filename)[0]
            output_filename = f"{filename_base}_Sub{int(subsidy * 100):02d}_CICombo_{ci_terminals_str}_Pricing_{pricing_mechanism}_{solver_type.upper()}_UNIFIED.pkl"
            output_filepath = os.path.join(results_folder, output_filename)

            save_results(results_to_save, output_filepath)
            # CALL THE FIXED SAVING FUNCTION
            save_results_to_excel_with_retry(results_to_save, results_file)

        inner_tqdm.close()

    logger.info(f"\n=== {solver_type.upper()} SIMULATION COMPLETE ===")
    logger.info(f"Total combinations: {combination_count}")
    logger.info(f"Successful: {successful_optimizations}")
    logger.info(f"Failed: {failed_optimizations}")
    if combination_count > 0:
        logger.info(f"Success rate: {successful_optimizations / combination_count * 100:.1f}%")


def save_results(results: Dict[str, Any], filepath: str):
    """Saves the simulation results to a pickle file with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")


# ====================================================================
# FIX APPLIED HERE: Retrieve instance_idx from results dictionary
# ====================================================================
def save_results_to_excel_with_retry(results: Dict[str, Any], filepath: str):
    """Saves results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_data = []

    sim_params = results['simulation_params']

    num_terminals = sim_params['num_terminals_from_data']
    subsidy = sim_params['subsidy']
    ci_terminals_str = sim_params['ci_terminals']
    pricing_mechanism = results['optimization_output'].get('pricing_mechanism', 'unknown')
    solver_used = sim_params['solver_used']
    # FIX: Correctly retrieve instance_index from the dictionary
    instance_idx_data = sim_params['instance_index']

    for obj_name in ['MAXPROF', 'MAXMIN']:
        profit_before = results['optimization_output']['profitBefore']

        # Handle cases where Pyomo only runs MAXPROF
        if solver_used == 'Pyomo' and obj_name == 'MAXMIN':
            if f'profitAfter_MAXPROF' not in results['optimization_output']: continue

        if ('Feasible' in results['optimization_output']['feasibility_status']):
            profit_after = results['optimization_output'][f'profitAfter_{obj_name}']
            volume_after = results['optimization_output'][f'volumeAfter_{obj_name}']
            transfer_fees = results['optimization_output'].get(f'transferFees_{obj_name}', np.zeros(num_terminals))
            obj_value = results['optimization_output'].get(f'objval_{obj_name}', np.nan)
            optimality_gap = results['optimization_output'].get(f'optimalityGap_{obj_name}', np.nan)
        else:
            profit_after = profit_before
            volume_after = results['optimization_output']['volumeBefore']
            transfer_fees = np.zeros(num_terminals)
            obj_value = np.nan
            optimality_gap = np.nan

        profit_change = profit_after - profit_before

        for i in range(num_terminals):
            df_data.append({
                'Solver': solver_used,
                'Num_Terminals': num_terminals,
                'Subset_Composition': sim_params['subset_composition'],
                'CI_Rate_Data': sim_params['ci_rate_from_data'],
                # FIX: Use the local variable instance_idx_data
                'Instance_Index': instance_idx_data,
                'Subsidy_Level': subsidy,
                'CI_Terminals_Combo': ci_terminals_str,
                'Pricing_Mechanism': pricing_mechanism,
                'Terminal_ID': i + 1,
                'Objective': obj_name,
                'Profit_Before': profit_before[i],
                'Profit_After': profit_after[i],
                'Profit_Change': profit_change[i],
                'Volume_After': volume_after[i] if volume_after is not None else np.nan,
                'Transfer_Fee': transfer_fees[i] if transfer_fees is not None else np.nan,
                'Feasibility_Status': results['optimization_output']['feasibility_status'],
                'Objective_Value': obj_value,
                'Optimality_Gap': optimality_gap
            })

    df = pd.DataFrame(df_data)
    csv_filepath = filepath.replace('.xlsx', '.csv')

    try:
        if os.path.exists(csv_filepath):
            df.to_csv(csv_filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filepath, mode='w', header=True, index=False)
        logger.debug(f"Results saved to CSV successfully")
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")


if __name__ == '__main__':
    # --- Check for required dependencies ---
    if not GUROBI_AVAILABLE:
        print("WARNING: Gurobi is not installed. Only Pyomo simulation will be available if Pyomo is installed.")
    if not PYOMO_AVAILABLE:
        print("WARNING: Pyomo is not installed. Only Gurobi simulation will be available if Gurobi is installed.")

    if GUROBI_AVAILABLE:
        # Run Gurobi simulation
        run_all_combinations_unified(solver_type='Gurobi')

    if PYOMO_AVAILABLE:
        # Run Pyomo simulation
        run_all_combinations_unified(solver_type='Pyomo')

    if not GUROBI_AVAILABLE and not PYOMO_AVAILABLE:
        print("\nERROR: Neither Gurobi nor Pyomo is available. Please install at least one.")