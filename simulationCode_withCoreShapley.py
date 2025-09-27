"""
UNIFIED TERMINAL COOPERATION SIMULATION FRAMEWORK WITH GAME THEORY ANALYSIS

This script provides optimization paths for terminal cooperation plus Core and Shapley value analysis:
1. Gurobi MINLP: Uses Gurobi's built-in MINLP and Piecewise Linear (PWL) capabilities.
2. Pyomo MINLP: Uses Pyomo with BONMIN or IPOPT, using sigmoid smoothing.
3. Game Theory Analysis: Calculates Core membership and Shapley values.

WARNING: Game theory analysis requires 2^n-1 optimization problems. Use only for small n.
"""

import os
import pickle
import numpy as np
import re
import pandas as pd
import logging
import time
import math
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
from itertools import combinations, chain

# =============================================================================
# GLOBAL CONFIGURATION SECTION
# =============================================================================

GLOBAL_SCENARIO_CONFIG = {
    'subsidies': np.array([0, 50, 100]),
    'pricing_mechanisms': ['optimized'],
    # Game theory analysis settings
    'enable_game_theory': True,  # Set to False to disable for large problems
    'max_terminals_for_game_theory': 5,  # Maximum terminals for full game theory analysis
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
            logging.FileHandler('simulation_game_theory.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# UTILITY FUNCTIONS (UNCHANGED)
# =============================================================================

def calculate_exact_cost(terminal_idx, utilization, terminal_capacities, cost_initial, cost_slope1, cost_slope2,
                         optimal_utilization):
    """Calculates cost using the ORIGINAL PIECEWISE FUNCTION."""
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
        for key in required_keys:
            if key not in input_data_ves:
                logger.error(f"Missing required key: {key}")
                return False

        num_terminals = input_data_ves['num_terminals']
        if len(input_data_ves['c']) != num_terminals:
            logger.error(f"Capacity array length doesn't match num_terminals")
            return False

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
                R_ci_subsidy[i] += vessel_volumes[j] * vessel_ci_capabilities[j, i] * ci_terminals[i] * subsidy

    revenue_before = revenue_per_container_before * volume_before + R_ci_subsidy

    # Calculate cost before cooperation
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
        logger.info("Current state validation complete. All constraints satisfied.")
    return is_feasible


def calculate_realistic_fallback(input_data_ves: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate realistic fallback values when optimization fails."""
    try:
        num_terminals = input_data_ves['num_terminals']
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


# =============================================================================
# GAME THEORY FUNCTIONS (NEW)
# =============================================================================

def generate_all_coalitions(num_terminals: int) -> List[List[int]]:
    """Generate all possible non-empty subsets of terminals."""
    coalitions = []
    for size in range(1, num_terminals + 1):
        for coalition in combinations(range(num_terminals), size):
            coalitions.append(list(coalition))
    return coalitions


def vessel_based_opt_coalition(input_data_ves: Dict[str, Any],
                               coalition_terminals: List[int],
                               pricing_mechanism: str = 'optimized') -> Optional[Dict[str, Any]]:
    """
    Modified optimization that only considers cooperation among terminals in the coalition.
    Non-coalition terminals operate independently (keep original assignments).
    """
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is required for coalition optimization.")
        return calculate_realistic_fallback(input_data_ves)

    if not validate_input_data(input_data_ves):
        logger.error("Input data validation failed")
        return calculate_realistic_fallback(input_data_ves)

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
    num_vessels = len(vessel_volumes)

    coalition_set = set(coalition_terminals)
    non_coalition_terminals = [i for i in range(num_terminals) if i not in coalition_set]

    results = {
        'profitBefore': profit_before,
        'volumeBefore': volume_before,
        'feasibility_status': 'Infeasible - Returned No-Cooperation Solution',
        'pricing_mechanism': pricing_mechanism,
        'solver_used': 'Gurobi_Coalition',
        'coalition': coalition_terminals
    }

    # Helper function to calculate total cost PWL points
    def calculate_total_cost_pwl_points(i, u_points):
        costs = np.zeros_like(u_points, dtype=float)
        u_opt = optimal_utilization[i]
        capacity = terminal_capacities[i]

        mask1 = (u_points >= 0) & (u_points <= u_opt)
        costs[mask1] = (cost_initial[i] * u_points[mask1] - 0.5 * cost_slope1[i] * u_points[mask1] ** 2)

        mask2 = u_points > u_opt
        if np.any(mask2):
            cost_at_optimal = (cost_initial[i] * u_opt - 0.5 * cost_slope1[i] * u_opt ** 2)
            mc_at_optimal = cost_initial[i] - cost_slope1[i] * u_opt

            costs[mask2] = (cost_at_optimal +
                            mc_at_optimal * (u_points[mask2] - u_opt) +
                            0.5 * cost_slope2[i] * (u_points[mask2] - u_opt) ** 2)

        return costs * capacity

    try:
        model = gp.Model(f"Coalition_Coop_{len(coalition_terminals)}_{pricing_mechanism}")
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
                mp = calculate_marginal_profit(i, utilization_before[i], -input_data_ves['decrease_rate'],
                                               input_data_ves['initial_charge'], optimal_utilization, cost_initial,
                                               cost_slope1, cost_slope2)
                profit_factor_after[i] = max(0, mp)

        # Constraints for all terminals
        for i in range(num_terminals):
            # Volume Calculation
            model.addConstr(volume_after[i] == gp.quicksum(
                vessel_volumes[j] * vessel_assignment[j, i] for j in range(num_vessels)), name=f"C_Q_{i}")

            # Utilization Constraint
            model.addConstr(utilization_after[i] * terminal_capacities[i] == volume_after[i], name=f"C_u_{i}")

            # Minimum Volume Constraint
            model.addConstr(volume_after[i] >= 0.1 * terminal_capacities[i], name=f"C_Q_min_{i}")

            # Cost PWL Link
            u_points = np.array(sorted(list(set(np.linspace(0, 1.0, 100)) | {optimal_utilization[i]})))
            y_points = calculate_total_cost_pwl_points(i, u_points)
            model.addGenConstrPWL(utilization_after[i], production_cost[i], u_points, y_points,
                                  name=f'C_TC_PWL_{i}')

            # Transfer Profit Calculation - MODIFIED FOR COALITION
            transfer_expr = gp.LinExpr(0)

            if i in coalition_set:
                # Terminal i is in coalition - can receive transfers from and send to other coalition members
                for j in range(num_vessels):
                    for k in range(num_terminals):
                        if k in coalition_set:  # Only consider transfers within coalition
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
                                transfer_expr += vessel_assignment[j, k] * (
                                        -fee_dest * vessel_volumes[j] -
                                        subsidy * ci_terminals[i] * vessel_ci_capabilities[j, i] * vessel_volumes[j])

            model.addConstr(profit_from_transfers[i] == transfer_expr, name=f"C_P_transfer_{i}")

            # Profit Calculation
            model.addConstr(profit_terminal[i] == revenue_before[i] + profit_from_transfers[i] - (
                    production_cost[i] - cost_before[i]), name=f"C_P_total_{i}")

            # Profit Stability
            model.addConstr(profit_terminal[i] >= profit_before[i], name=f"C_P_stab_{i}")

        # COALITION CONSTRAINT: Non-coalition terminals must keep original assignments
        for j in range(num_vessels):
            for i in non_coalition_terminals:
                model.addConstr(vessel_assignment[j, i] == vessel_assignments[j, i],
                                name=f"C_fixed_assignment_{j}_{i}")

        # Global Constraints
        for j in range(num_vessels):
            model.addConstr(gp.quicksum(vessel_assignment[j, i] for i in range(num_terminals)) == 1,
                            name=f"C_x_sum_{j}")

        # Volume Conservation
        model.addConstr(gp.quicksum(volume_after[i] for i in range(num_terminals)) == np.sum(volume_before),
                        name="C_Q_cons")

        # Set objective - maximize total profit of coalition members only
        coalition_profit_expr = gp.quicksum(profit_terminal[i] for i in coalition_terminals)
        model.setObjective(coalition_profit_expr, GRB.MAXIMIZE)

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            profit_after = np.array([profit_terminal[i].x for i in range(num_terminals)])
            volume_after_values = np.array([volume_after[i].x for i in range(num_terminals)])

            if pricing_mechanism in ['optimized', 'marginal_cost']:
                transfer_fees = np.array([profit_factor_after[i].x for i in range(num_terminals)])
            else:
                transfer_fees = np.array([profit_factor_after[i] for i in range(num_terminals)])

            results['profitAfter_MAXPROF'] = profit_after
            results['volumeAfter_MAXPROF'] = volume_after_values
            results['transferFees_MAXPROF'] = transfer_fees
            results['objval_MAXPROF'] = model.objVal
            results['optimalityGap_MAXPROF'] = model.MIPGap
            results['feasibility_status'] = f'Feasible Coalition ({len(coalition_terminals)} terminals)'

            logger.debug(f"Coalition optimization completed successfully for {coalition_terminals}")
        else:
            logger.warning(f"Coalition optimization failed with status: {model.status}")
            results['profitAfter_MAXPROF'] = profit_before.copy()
            results['volumeAfter_MAXPROF'] = volume_before.copy()
            results['transferFees_MAXPROF'] = np.zeros(num_terminals)
            results['objval_MAXPROF'] = np.nan
            results['optimalityGap_MAXPROF'] = np.nan

        model.dispose()

    except Exception as e:
        logger.error(f"Coalition optimization failed with error: {e}")

    return results


def calculate_no_cooperation_value(input_data_ves: Dict[str, Any], coalition_terminals: List[int]) -> float:
    """Calculate the total profit of coalition terminals under no cooperation (baseline state)."""
    _, _, _, _, profit_before = calculate_baseline_state(input_data_ves)
    return np.sum([profit_before[i] for i in coalition_terminals])


def calculate_characteristic_function(input_data_ves: Dict[str, Any]) -> Dict[tuple, float]:
    """
    Calculate the value v(S) for every possible coalition S.
    Returns a dictionary mapping coalition tuples to their total profit.
    """
    num_terminals = input_data_ves['num_terminals']
    coalitions = generate_all_coalitions(num_terminals)
    characteristic_function = {}

    logger.info(f"Calculating characteristic function for {len(coalitions)} coalitions...")

    for i, coalition in enumerate(tqdm(coalitions, desc="Coalition Analysis")):
        try:
            result = vessel_based_opt_coalition(input_data_ves, coalition)
            if result and 'Feasible' in result.get('feasibility_status', ''):
                # Sum profits of coalition members only
                coalition_profit = np.sum([result['profitAfter_MAXPROF'][j] for j in coalition])
            else:
                # Fallback to no-cooperation value for this coalition
                coalition_profit = calculate_no_cooperation_value(input_data_ves, coalition)

            characteristic_function[tuple(sorted(coalition))] = coalition_profit

        except Exception as e:
            logger.error(f"Error calculating value for coalition {coalition}: {e}")
            # Use no-cooperation fallback
            coalition_profit = calculate_no_cooperation_value(input_data_ves, coalition)
            characteristic_function[tuple(sorted(coalition))] = coalition_profit

    return characteristic_function


def check_core_membership(characteristic_function: Dict[tuple, float],
                          grand_coalition_allocation: np.ndarray,
                          tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Check if the grand coalition allocation is in the core.
    Tests if any coalition can improve by defecting.
    """
    num_terminals = len(grand_coalition_allocation)
    core_violations = []

    # Test every possible coalition except the grand coalition
    for coalition_tuple, coalition_value in characteristic_function.items():
        if len(coalition_tuple) == num_terminals:  # Skip grand coalition
            continue

        # Sum of allocations to coalition members in grand coalition
        coalition_allocation_sum = sum(grand_coalition_allocation[i] for i in coalition_tuple)

        # Check if coalition can improve by defecting
        if coalition_value > coalition_allocation_sum + tolerance:
            core_violations.append({
                'coalition': coalition_tuple,
                'coalition_value': coalition_value,
                'current_allocation': coalition_allocation_sum,
                'improvement': coalition_value - coalition_allocation_sum
            })

    return {
        'in_core': len(core_violations) == 0,
        'violations': core_violations,
        'num_violations': len(core_violations),
        'core_stability_score': 1.0 - (len(core_violations) / (2 ** num_terminals - 2))  # Normalized score
    }


def calculate_shapley_values(characteristic_function: Dict[tuple, float],
                             num_terminals: int) -> np.ndarray:
    """Calculate Shapley values using the characteristic function."""
    shapley_values = np.zeros(num_terminals)

    logger.info("Calculating Shapley values...")

    for terminal in range(num_terminals):
        marginal_contributions = []

        # Consider all possible coalitions not containing this terminal
        for coalition_size in range(num_terminals):
            for coalition in combinations([t for t in range(num_terminals) if t != terminal], coalition_size):
                coalition_tuple = tuple(sorted(coalition))
                coalition_with_terminal = tuple(sorted(coalition + (terminal,)))

                # Marginal contribution = v(S ∪ {i}) - v(S)
                v_with = characteristic_function.get(coalition_with_terminal, 0)
                v_without = characteristic_function.get(coalition_tuple, 0) if coalition_tuple else 0

                marginal_contribution = v_with - v_without

                # Weight by coalition size (Shapley formula)
                weight = (math.factorial(coalition_size) * math.factorial(num_terminals - coalition_size - 1) /
                          math.factorial(num_terminals))

                marginal_contributions.append(weight * marginal_contribution)

        shapley_values[terminal] = sum(marginal_contributions)

    return shapley_values


def run_game_theory_analysis(input_data_ves: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete game theory analysis including Core and Shapley values.
    WARNING: Computationally intensive for num_terminals > 5
    """
    num_terminals = input_data_ves['num_terminals']

    if num_terminals > GLOBAL_SCENARIO_CONFIG['max_terminals_for_game_theory']:
        logger.warning(
            f"Skipping game theory analysis: {num_terminals} terminals > {GLOBAL_SCENARIO_CONFIG['max_terminals_for_game_theory']} limit")
        return {
            'skipped': True,
            'reason': f'Too many terminals ({num_terminals}) for game theory analysis'
        }

    total_coalitions = 2 ** num_terminals - 1
    logger.info(f"Starting game theory analysis for {num_terminals} terminals ({total_coalitions} coalitions)")

    try:
        # 1. Calculate characteristic function (most expensive step)
        characteristic_function = calculate_characteristic_function(input_data_ves)

        # 2. Get grand coalition result
        grand_coalition_result = vessel_based_opt_gurobi(input_data_ves)
        if not grand_coalition_result or 'Feasible' not in grand_coalition_result.get('feasibility_status', ''):
            logger.error("Grand coalition optimization failed")
            return {'error': 'Grand coalition optimization failed'}

        grand_coalition_profits = grand_coalition_result['profitAfter_MAXPROF']

        # 3. Check Core membership
        logger.info("Checking Core membership...")
        core_analysis = check_core_membership(characteristic_function, grand_coalition_profits)

        # 4. Calculate Shapley values
        logger.info("Calculating Shapley values...")
        shapley_values = calculate_shapley_values(characteristic_function, num_terminals)

        # 5. Compare allocations
        profit_before = grand_coalition_result['profitBefore']

        return {
            'characteristic_function': characteristic_function,
            'core_analysis': core_analysis,
            'shapley_values': shapley_values,
            'grand_coalition_profits': grand_coalition_profits,
            'profit_before_cooperation': profit_before,
            'shapley_vs_grand_coalition': shapley_values - grand_coalition_profits,
            'shapley_vs_no_cooperation': shapley_values - profit_before,
            'grand_coalition_vs_no_cooperation': grand_coalition_profits - profit_before,
            'total_coalitions_analyzed': len(characteristic_function),
            'analysis_successful': True
        }

    except Exception as e:
        logger.error(f"Game theory analysis failed: {e}")
        return {
            'error': str(e),
            'analysis_successful': False
        }


# =============================================================================
# MODIFIED MAIN OPTIMIZATION FUNCTIONS
# =============================================================================

def vessel_based_opt_gurobi(input_data_ves: Dict[str, Any], pricing_mechanism: str = 'optimized') -> Optional[
    Dict[str, Any]]:
    """Terminal cooperation optimization using Gurobi (full cooperation)."""
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

    # This is essentially the coalition optimization with all terminals in the coalition
    num_terminals = input_data_ves['num_terminals']
    all_terminals = list(range(num_terminals))

    result = vessel_based_opt_coalition(input_data_ves, all_terminals, pricing_mechanism)

    # Add MAXMIN results (copy MAXPROF for simplicity)
    if result and 'profitAfter_MAXPROF' in result:
        result['profitAfter_MAXMIN'] = result['profitAfter_MAXPROF'].copy()
        result['volumeAfter_MAXMIN'] = result['volumeAfter_MAXPROF'].copy()
        result['transferFees_MAXMIN'] = result['transferFees_MAXPROF'].copy()
        result['objval_MAXMIN'] = result.get('objval_MAXPROF', np.nan)
        result['optimalityGap_MAXMIN'] = result.get('optimalityGap_MAXPROF', np.nan)

    return result


# =============================================================================
# MAIN EXECUTION BLOCK (MODIFIED)
# =============================================================================

def run_all_combinations_unified(solver_type: str = 'Gurobi'):
    """Main function to run the simulation using the specified solver."""
    if solver_type == 'Gurobi' and not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot run Gurobi simulation.")
        return

    data_folder = "generated_data"
    results_folder = f"simulation_results_{solver_type.lower()}_game_theory"
    results_file = os.path.join(results_folder, f"summary_results_{solver_type.lower()}_game_theory.csv")

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

    logger.info(f"=== STARTING {solver_type.upper()} GAME THEORY SIMULATION ===")
    logger.info(f"Testing Subsidies: {subsidies}")
    logger.info(f"Testing Pricing Mechanisms: {pricing_mechanisms}")
    logger.info(f"Game Theory Analysis: {'Enabled' if GLOBAL_SCENARIO_CONFIG['enable_game_theory'] else 'Disabled'}")

    filename_pattern = re.compile(r'data_T(\d+)_subset_([\d_]+)_CI(\d+)_instance_(\d+)\.pkl')
    combination_count = 0
    successful_optimizations = 0
    failed_optimizations = 0
    game_theory_analyses = 0

    # OUTER TQDM LOOP (over data files)
    for filename in tqdm(data_files, desc="Overall File Progress"):
        filepath = os.path.join(data_folder, filename)
        match = filename_pattern.match(filename)
        if not match: continue

        num_terminals_str, subset_str, ci_rate_str, instance_idx_str = match.groups()
        num_terminals = int(num_terminals_str)
        ci_rate = int(ci_rate_str) / 100.0
        instance_idx = int(instance_idx_str)

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue

        # Prepare scenarios (simplified - only test a subset for game theory)
        scenarios = []
        num_ci_combinations = min(8, 2 ** num_terminals)  # Limit CI combinations for game theory
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

            # Standard optimization
            result = vessel_based_opt_gurobi(input_data_ves, pricing_mechanism=pricing_mechanism)

            if result and 'Feasible' in result.get('feasibility_status', ''):
                successful_optimizations += 1
            else:
                failed_optimizations += 1

            # Game theory analysis (if enabled and feasible)
            game_theory_result = None
            if (GLOBAL_SCENARIO_CONFIG['enable_game_theory'] and
                    result and 'Feasible' in result.get('feasibility_status', '')):
                game_theory_result = run_game_theory_analysis(input_data_ves)
                if game_theory_result and game_theory_result.get('analysis_successful', False):
                    game_theory_analyses += 1

            # Prepare results dictionary
            results_to_save = {
                'simulation_params': {
                    'filename': filename, 'subsidy': subsidy, 'ci_terminals': ci_terminals_str,
                    'ci_rate_from_data': ci_rate, 'num_terminals_from_data': num_terminals,
                    'instance_index': instance_idx, 'subset_composition': subset_str,
                    'pricing_mechanism': pricing_mechanism, 'solver_used': solver_type
                },
                'optimization_output': result,
                'game_theory_analysis': game_theory_result
            }

            filename_base = os.path.splitext(filename)[0]
            output_filename = f"{filename_base}_Sub{int(subsidy * 100):02d}_CICombo_{ci_terminals_str}_Pricing_{pricing_mechanism}_{solver_type.upper()}_GAME_THEORY.pkl"
            output_filepath = os.path.join(results_folder, output_filename)

            save_results(results_to_save, output_filepath)
            save_results_to_excel_with_retry(results_to_save, results_file)

        inner_tqdm.close()

    logger.info(f"\n=== {solver_type.upper()} GAME THEORY SIMULATION COMPLETE ===")
    logger.info(f"Total combinations: {combination_count}")
    logger.info(f"Successful optimizations: {successful_optimizations}")
    logger.info(f"Failed optimizations: {failed_optimizations}")
    logger.info(f"Game theory analyses completed: {game_theory_analyses}")
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


def save_results_to_excel_with_retry(results: Dict[str, Any], filepath: str):
    """Saves results to CSV with game theory data."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_data = []

    sim_params = results['simulation_params']
    game_theory = results.get('game_theory_analysis', {})

    num_terminals = sim_params['num_terminals_from_data']
    subsidy = sim_params['subsidy']
    ci_terminals_str = sim_params['ci_terminals']
    pricing_mechanism = results['optimization_output'].get('pricing_mechanism', 'unknown')
    solver_used = sim_params['solver_used']
    instance_idx_data = sim_params['instance_index']

    # Extract game theory results
    in_core = game_theory.get('core_analysis', {}).get('in_core', np.nan)
    core_violations = game_theory.get('core_analysis', {}).get('num_violations', np.nan)
    core_stability_score = game_theory.get('core_analysis', {}).get('core_stability_score', np.nan)
    shapley_values = game_theory.get('shapley_values', np.full(num_terminals, np.nan))

    for obj_name in ['MAXPROF', 'MAXMIN']:
        profit_before = results['optimization_output']['profitBefore']

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
                'Optimality_Gap': optimality_gap,
                # Game theory columns
                'Shapley_Value': shapley_values[i] if len(shapley_values) > i else np.nan,
                'Shapley_vs_Cooperation': (shapley_values[i] - profit_after[i]) if len(shapley_values) > i else np.nan,
                'In_Core': in_core,
                'Core_Violations': core_violations,
                'Core_Stability_Score': core_stability_score
            })

    df = pd.DataFrame(df_data)
    csv_filepath = filepath.replace('.xlsx', '.csv')

    try:
        if os.path.exists(csv_filepath):
            df.to_csv(csv_filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filepath, mode='w', header=True, index=False)
        logger.debug("Results saved to CSV successfully")
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")


if __name__ == '__main__':
    # Check for required dependencies
    if not GUROBI_AVAILABLE:
        print("WARNING: Gurobi is not installed. Game theory analysis requires Gurobi.")
        print("Only basic analysis will be available.")

    # Display computational complexity warning
    print("\n" + "=" * 80)
    print("GAME THEORY ANALYSIS COMPUTATIONAL COMPLEXITY WARNING")
    print("=" * 80)
    print("Game theory analysis requires solving 2^n-1 optimization problems:")
    print("3 terminals: 7 optimizations")
    print("4 terminals: 15 optimizations")
    print("5 terminals: 31 optimizations")
    print("6 terminals: 63 optimizations")
    print("7+ terminals: EXTREMELY COMPUTATIONALLY INTENSIVE")
    print(f"\nCurrent limit: {GLOBAL_SCENARIO_CONFIG['max_terminals_for_game_theory']} terminals")
    print(f"Game theory analysis: {'ENABLED' if GLOBAL_SCENARIO_CONFIG['enable_game_theory'] else 'DISABLED'}")
    print("=" * 80 + "\n")

    if GUROBI_AVAILABLE:
        run_all_combinations_unified(solver_type='Gurobi')
    else:
        print("ERROR: Gurobi is required for game theory analysis.")