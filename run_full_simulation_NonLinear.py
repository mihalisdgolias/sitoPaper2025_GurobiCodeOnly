import os
import pickle
import numpy as np
import re
import pandas as pd
import logging
import time
import psutil
import json
from typing import Dict, Any, Optional
from tqdm import tqdm


# Configuration management
class SimulationConfig:
    DEFAULT_OPTIMALITY_TOL = 0.01
    DEFAULT_M_VALUE = 1000000
    MAX_MEMORY_MB = 4000  # Maximum memory usage in MB
    BATCH_SIZE = 50  # Process combinations in batches
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2  # seconds
    CONSERVATION_TOLERANCE = 0.01  # 1% tolerance for conservation checks


# Dependency checks
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    print("ERROR: Gurobi is required but not installed. Install with: pip install gurobipy")
    GUROBI_AVAILABLE = False
    raise ImportError("Gurobi is required but not available")
except Exception as e:
    print(f"ERROR: Gurobi license or configuration issue: {e}")
    GUROBI_AVAILABLE = False
    raise RuntimeError(f"Gurobi error: {e}")


# Setup logging
def setup_logging():
    """Configure logging for the simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def validate_input_data(input_data_ves: Dict[str, Any]) -> bool:
    """
    Validate input data structure and values.

    Args:
        input_data_ves: Input data dictionary

    Returns:
        bool: True if valid, False otherwise
    """
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

        # Validate utilization ratios
        if np.any((input_data_ves['optimal_vcr_cost_point'] < 0) |
                  (input_data_ves['optimal_vcr_cost_point'] > 1)):
            logger.error("Optimal VCR points must be between 0 and 1")
            return False

        return True

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def validate_solution_conservation(input_data_ves: Dict[str, Any], results: Dict[str, Any]) -> bool:
    """
    Validate that the optimization solution preserves conservation laws.

    Args:
        input_data_ves: Input data dictionary
        results: Optimization results dictionary

    Returns:
        bool: True if conservation checks pass, False otherwise
    """
    try:
        # Skip validation if optimization failed
        if 'volumeAfter_MAXPROF' not in results:
            logger.debug("Skipping conservation check - optimization failed")
            return True

        # Check total volume conservation
        volume_before_total = np.sum(input_data_ves['v'][:, None] * input_data_ves['x'])
        volume_after_total = np.sum(results['volumeAfter_MAXPROF'])

        volume_error = abs(
            volume_before_total - volume_after_total) / volume_before_total if volume_before_total > 0 else 0

        if volume_error > SimulationConfig.CONSERVATION_TOLERANCE:
            logger.warning(
                f"Volume conservation violation: {volume_error:.4%} error (tolerance: {SimulationConfig.CONSERVATION_TOLERANCE:.2%})")
            logger.warning(f"Volume before: {volume_before_total:,.1f}, Volume after: {volume_after_total:,.1f}")
            return False

        # Check vessel count conservation (each vessel should be assigned to exactly one terminal)
        # This is implicitly checked by Gurobi constraints, but let's validate the totals
        num_vessels_before = np.sum(input_data_ves['x'])

        # For after state, we need to check that total assignments equal number of vessels
        # This is guaranteed by constraints, but we can check the volume consistency

        # Check capacity constraints are satisfied
        terminal_capacities = input_data_ves['c']
        volumes_after = results['volumeAfter_MAXPROF']

        for i in range(len(terminal_capacities)):
            if volumes_after[i] > terminal_capacities[i]:
                logger.warning(
                    f"Capacity violation at terminal {i + 1}: {volumes_after[i]:,.1f} > {terminal_capacities[i]:,.1f}")
                return False

            if volumes_after[i] < 0.1 * terminal_capacities[i]:
                logger.warning(
                    f"Minimum volume violation at terminal {i + 1}: {volumes_after[i]:,.1f} < {0.1 * terminal_capacities[i]:,.1f}")
                return False

        # Check profit improvement constraint
        profit_before = results['profitBefore']
        profit_after = results['profitAfter_MAXPROF']

        for i in range(len(profit_before)):
            if profit_after[i] < profit_before[i] - 1e-6:  # Small tolerance for numerical errors
                logger.warning(
                    f"Profit degradation at terminal {i + 1}: {profit_after[i]:,.2f} < {profit_before[i]:,.2f}")
                return False

        logger.debug(f"Conservation checks passed - Volume error: {volume_error:.6%}, All constraints satisfied")
        return True

    except Exception as e:
        logger.error(f"Conservation validation failed due to error: {e}")
        return False


def validate_solution_consistency(input_data_ves: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive solution validation and return detailed metrics.

    Args:
        input_data_ves: Input data dictionary
        results: Optimization results dictionary

    Returns:
        Dict with validation results and metrics
    """
    validation_results = {
        'conservation_passed': False,
        'volume_error': float('inf'),
        'capacity_violations': 0,
        'profit_violations': 0,
        'total_checks': 0,
        'warnings': []
    }

    try:
        if 'volumeAfter_MAXPROF' not in results:
            validation_results['warnings'].append("Optimization failed - no solution to validate")
            return validation_results

        # Volume conservation check
        volume_before_total = np.sum(input_data_ves['v'][:, None] * input_data_ves['x'])
        volume_after_total = np.sum(results['volumeAfter_MAXPROF'])
        volume_error = abs(
            volume_before_total - volume_after_total) / volume_before_total if volume_before_total > 0 else 0
        validation_results['volume_error'] = volume_error
        validation_results['total_checks'] += 1

        # Capacity constraint checks
        terminal_capacities = input_data_ves['c']
        volumes_after = results['volumeAfter_MAXPROF']
        capacity_violations = 0

        for i in range(len(terminal_capacities)):
            if volumes_after[i] > terminal_capacities[i] + 1e-6:
                capacity_violations += 1
                validation_results['warnings'].append(
                    f"Terminal {i + 1} capacity exceeded: {volumes_after[i]:,.1f} > {terminal_capacities[i]:,.1f}")

            if volumes_after[i] < 0.1 * terminal_capacities[i] - 1e-6:
                capacity_violations += 1
                validation_results['warnings'].append(
                    f"Terminal {i + 1} minimum volume violated: {volumes_after[i]:,.1f} < {0.1 * terminal_capacities[i]:,.1f}")

        validation_results['capacity_violations'] = capacity_violations
        validation_results['total_checks'] += len(terminal_capacities) * 2

        # Profit improvement checks
        profit_before = results['profitBefore']
        profit_after = results['profitAfter_MAXPROF']
        profit_violations = 0

        for i in range(len(profit_before)):
            if profit_after[i] < profit_before[i] - 1e-6:
                profit_violations += 1
                validation_results['warnings'].append(
                    f"Terminal {i + 1} profit decreased: {profit_after[i]:,.2f} < {profit_before[i]:,.2f}")

        validation_results['profit_violations'] = profit_violations
        validation_results['total_checks'] += len(profit_before)

        # Overall assessment
        validation_results['conservation_passed'] = (
                volume_error <= SimulationConfig.CONSERVATION_TOLERANCE and
                capacity_violations == 0 and
                profit_violations == 0
        )

        return validation_results

    except Exception as e:
        validation_results['warnings'].append(f"Validation error: {str(e)}")
        return validation_results


def calculate_realistic_fallback(input_data_ves: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate realistic fallback values when optimization fails.

    Args:
        input_data_ves: Input data dictionary

    Returns:
        Dict with fallback results
    """
    try:
        num_terminals = input_data_ves['num_terminals']

        # Calculate current state
        volume_before = np.sum(input_data_ves['v'][:, None] * input_data_ves['x'], axis=0)
        utilization_before = volume_before / input_data_ves['c']

        # Estimate reasonable profit based on input parameters
        revenue_per_teu = (input_data_ves['initial_charge'] +
                           input_data_ves['decrease_rate'] * utilization_before)

        # Conservative profit estimate (10% of revenue)
        profit_before = np.maximum(revenue_per_teu * volume_before * 0.1,
                                   volume_before * 50)  # At least $50 per TEU

        return {
            'feasibility_status': 'Fallback Solution - Optimization Failed',
            'profitBefore': profit_before,
            'profitAfter_MAXPROF': profit_before.copy(),
            'profitAfter_MAXMIN': profit_before.copy(),
            'volumeBefore': volume_before,
            'volumeAfter_MAXPROF': volume_before.copy(),
            'volumeAfter_MAXMIN': volume_before.copy(),
            'objval_MAXPROF': np.nan,
            'optimalityGap_MAXPROF': np.nan,
            'objval_MAXMIN': np.nan,
            'optimalityGap_MAXMIN': np.nan,
        }

    except Exception as e:
        logger.error(f"Fallback calculation failed: {e}")
        return None


def vessel_based_opt(input_data_ves: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Terminal cooperation optimization using Gurobi with non-linear quadratic cost function.

    Key change from original:
    - Replaced piecewise linear approximation with exact quadratic cost constraints

    Args:
        input_data_ves: Dictionary containing all input parameters

    Returns:
        Dictionary with optimization results or None if failed critically
    """

    if not validate_input_data(input_data_ves):
        logger.error("Input data validation failed")
        return calculate_realistic_fallback(input_data_ves)

    try:
        # Extract input data
        subsidy = input_data_ves['subPar']
        ci_terminals = input_data_ves['ciTerm']
        num_terminals = input_data_ves['num_terminals']

        # Vessel and terminal data
        vessel_volumes = input_data_ves['v']
        vessel_assignments = input_data_ves['x']
        vessel_ci_capabilities = input_data_ves['xCI']
        terminal_capacities = input_data_ves['c']

        # Terminal model parameters
        revenue_decrease_rate = -input_data_ves['decrease_rate']
        revenue_initial_charge = input_data_ves['initial_charge']
        cost_initial = input_data_ves['cost_initial']
        cost_slope1 = input_data_ves['cost_decrease_rate']
        cost_slope2 = input_data_ves['cost_increase_rate']
        optimal_utilization = input_data_ves['optimal_vcr_cost_point']

        big_M = input_data_ves.get('M', SimulationConfig.DEFAULT_M_VALUE)
        num_vessels = len(vessel_volumes)

        # Calculate before-cooperation state
        volume_before = np.sum(vessel_volumes[:, None] * vessel_assignments, axis=0)
        utilization_before = volume_before / terminal_capacities

        # FIXED: Calculate base revenue rate that stays constant (like GAMS pfBefore)
        revenue_per_container_before = revenue_decrease_rate * utilization_before + revenue_initial_charge
        ci_subsidy_revenue_before = np.sum(vessel_volumes[:, None] * vessel_ci_capabilities *
                                           ci_terminals[None, :] * subsidy, axis=0)
        revenue_before = revenue_per_container_before * volume_before + ci_subsidy_revenue_before

        # Calculate cost before cooperation
        cost_before = np.zeros(num_terminals)
        for i in range(num_terminals):
            utilization = utilization_before[i]
            if utilization <= optimal_utilization[i]:
                total_cost = (cost_initial[i] * utilization -
                              0.5 * cost_slope1[i] * utilization ** 2) * terminal_capacities[i]
            else:
                cost_at_optimal = (cost_initial[i] * optimal_utilization[i] -
                                   0.5 * cost_slope1[i] * optimal_utilization[i] ** 2) * terminal_capacities[i]
                additional_cost = ((cost_initial[i] - cost_slope1[i] * optimal_utilization[i]) *
                                   (utilization - optimal_utilization[i]) +
                                   0.5 * cost_slope2[i] * (utilization - optimal_utilization[i]) ** 2) * \
                                  terminal_capacities[i]
                total_cost = cost_at_optimal + additional_cost
            cost_before[i] = total_cost

        profit_before = revenue_before - cost_before

        results = {
            'profitBefore': profit_before,
            'volumeBefore': volume_before,
            'feasibility_status': 'Infeasible - Returned No-Cooperation Solution'
        }
        # Solve for both objectives
        for objective_name in ['MAXPROF', 'MAXMIN']:
            try:
                model = gp.Model(f"Terminal_Cooperation_{objective_name}")
                model.setParam('OutputFlag', 0)
                model.setParam('OptimalityTol', SimulationConfig.DEFAULT_OPTIMALITY_TOL)
                model.setParam('TimeLimit', 300)  # 5 minute time limit

                # Decision variables
                vessel_assignment = model.addVars(num_vessels, num_terminals, vtype=GRB.BINARY,
                                                  name="vessel_assignment")
                volume_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="volume_after")
                profit_terminal = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="profit_terminal")
                production_cost = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="production_cost")
                profit_from_transfers = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="profit_from_transfers")
                profit_factor_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, lb=0.0,
                                                    name="profit_factor_after")
                extra_profit = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="extra_profit")
                extra_loss = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="extra_loss")
                volume_difference = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="volume_difference")
                receives_vessels = model.addVars(num_terminals, vtype=GRB.BINARY, name="receives_vessels")
                sends_vessels = model.addVars(num_terminals, vtype=GRB.BINARY, name="sends_vessels")
                utilization_after = model.addVars(num_terminals, vtype=GRB.CONTINUOUS, name="utilization_after")

                if objective_name == 'MAXMIN':
                    min_profit = model.addVar(vtype=GRB.CONTINUOUS, name="min_profit")

                # Build constraints for each terminal
                for i in range(num_terminals):
                    # Volume constraint
                    model.addConstr(volume_after[i] == gp.quicksum(vessel_volumes[j] * vessel_assignment[j, i]
                                                                   for j in range(num_vessels)),
                                    name=f"volume_constraint_{i}")

                    # Utilization constraint
                    model.addConstr(utilization_after[i] * terminal_capacities[i] == volume_after[i],
                                    name=f"utilization_constraint_{i}")

                    # Extra profit from receiving vessels
                    extra_profit_expr = gp.LinExpr()
                    for j in range(num_vessels):
                        for ii in range(num_terminals):
                            if vessel_assignments[j, ii] == 1 and i != ii:
                                extra_profit_expr += vessel_assignment[j, i] * (
                                        profit_factor_after[ii] * vessel_volumes[j] +
                                        ci_terminals[i] * subsidy * vessel_ci_capabilities[j, ii] * vessel_volumes[j])

                    model.addConstr(extra_profit[i] == extra_profit_expr, name=f"extra_profit_constraint_{i}")

                    # Extra loss from sending vessels
                    extra_loss_expr = gp.LinExpr()
                    for j in range(num_vessels):
                        for ii in range(num_terminals):
                            if vessel_assignments[j, i] == 1 and i != ii:
                                extra_loss_expr += vessel_assignment[j, ii] * (
                                        -profit_factor_after[i] * vessel_volumes[j] -
                                        subsidy * ci_terminals[i] * vessel_ci_capabilities[j, i] * vessel_volumes[j])

                    model.addConstr(extra_loss[i] == extra_loss_expr, name=f"extra_loss_constraint_{i}")

                    # Total profit from transfers
                    model.addConstr(profit_from_transfers[i] == extra_profit[i] + extra_loss[i],
                                    name=f"profit_transfers_constraint_{i}")

                    # NON-LINEAR QUADRATIC COST FUNCTION IMPLEMENTATION
                    # Replace PWL approximation with exact quadratic constraints

                    # Binary variable to track which piece of the cost function is active
                    below_optimal_i = model.addVar(vtype=GRB.BINARY, name=f"below_optimal_{i}")

                    # Auxiliary variables for the two cost pieces
                    cost_piece1_i = model.addVar(vtype=GRB.CONTINUOUS, name=f"cost_piece1_{i}")
                    cost_piece2_i = model.addVar(vtype=GRB.CONTINUOUS, name=f"cost_piece2_{i}")

                    # Constraints to enforce utilization bounds based on binary variable
                    big_M_util = 1.0  # Since utilization is bounded by 1.0
                    model.addConstr(utilization_after[i] <= optimal_utilization[i] + big_M_util * (1 - below_optimal_i),
                                    name=f"utilization_upper_piece1_{i}")
                    model.addConstr(utilization_after[i] >= optimal_utilization[i] - big_M_util * below_optimal_i,
                                    name=f"utilization_lower_piece2_{i}")

                    # Quadratic cost for piece 1 (u <= u_optimal)
                    # When below_optimal_i = 1: cost = (cost_initial[i] * u - 0.5 * cost_slope1[i] * u^2) * capacity * 0.99
                    # When below_optimal_i = 0: cost = 0
                    # We need to use auxiliary variables to handle the binary multiplication

                    # Auxiliary variable for utilization when in piece 1
                    util_piece1 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"util_piece1_{i}")
                    model.addConstr(util_piece1 <= utilization_after[i], name=f"util_piece1_upper_{i}")
                    model.addConstr(util_piece1 <= below_optimal_i, name=f"util_piece1_binary_{i}")
                    model.addConstr(util_piece1 >= utilization_after[i] - (1 - below_optimal_i),
                                    name=f"util_piece1_lower_{i}")

                    # Quadratic constraint for piece 1
                    model.addQConstr(cost_piece1_i == 0.99 * terminal_capacities[i] * (
                            cost_initial[i] * util_piece1 -
                            0.5 * cost_slope1[i] * util_piece1 * util_piece1
                    ), name=f"quadratic_cost_piece1_{i}")

                    # For piece 2 (u > u_optimal), we need to handle the excess utilization
                    cost_at_optimal_val = 0.99 * terminal_capacities[i] * (
                            cost_initial[i] * optimal_utilization[i] -
                            0.5 * cost_slope1[i] * optimal_utilization[i] ** 2
                    )
                    marginal_cost_at_optimal = cost_initial[i] - cost_slope1[i] * optimal_utilization[i]

                    # Auxiliary variable for excess utilization when in piece 2
                    excess_util_piece2 = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"excess_util_piece2_{i}")
                    model.addConstr(excess_util_piece2 <= utilization_after[i] - optimal_utilization[
                        i] + big_M_util * below_optimal_i,
                                    name=f"excess_util_upper_{i}")
                    model.addConstr(excess_util_piece2 <= big_M_util * (1 - below_optimal_i),
                                    name=f"excess_util_binary_{i}")
                    model.addConstr(excess_util_piece2 >= utilization_after[i] - optimal_utilization[
                        i] - big_M_util * below_optimal_i,
                                    name=f"excess_util_lower_{i}")

                    # Binary variable for whether we're in piece 2
                    in_piece2 = model.addVar(vtype=GRB.BINARY, name=f"in_piece2_{i}")
                    model.addConstr(in_piece2 == 1 - below_optimal_i, name=f"piece2_indicator_{i}")

                    # Quadratic constraint for piece 2
                    model.addQConstr(cost_piece2_i == in_piece2 * cost_at_optimal_val +
                                     0.99 * terminal_capacities[i] * (
                                             marginal_cost_at_optimal * excess_util_piece2 +
                                             0.5 * cost_slope2[i] * excess_util_piece2 * excess_util_piece2
                                     ), name=f"quadratic_cost_piece2_{i}")

                    # Total production cost is the sum of both pieces
                    model.addConstr(production_cost[i] == cost_piece1_i + cost_piece2_i,
                                    name=f"total_production_cost_{i}")

                    # FIXED: Revenue calculation with FIXED rate per container (like GAMS)
                    # Base revenue uses the fixed rate from before cooperation
                    base_revenue_fixed = revenue_per_container_before[i] * volume_after[i]

                    # CI revenue from this terminal's vessels (unchanged)
                    ci_revenue = gp.quicksum(ci_terminals[i] * subsidy * vessel_ci_capabilities[j, ii] *
                                             vessel_assignment[j, i] * vessel_volumes[j]
                                             for j in range(num_vessels) for ii in range(num_terminals)
                                             if vessel_assignments[j, ii] == 1)

                    # Total profit = fixed base revenue + CI subsidies + transfers - costs
                    model.addConstr(
                        profit_terminal[i] == base_revenue_fixed + ci_revenue + profit_from_transfers[i] -
                        production_cost[i],
                        name=f"profit_constraint_{i}")

                    # Profit improvement constraint
                    model.addConstr(profit_terminal[i] >= profit_before[i], name=f"profit_improvement_constraint_{i}")

                    # Volume difference tracking
                    model.addConstr(volume_difference[i] == volume_after[i] - volume_before[i],
                                    name=f"volume_diff_constraint_{i}")

                    # Binary constraints for vessel transfers
                    model.addConstr(sends_vessels[i] + receives_vessels[i] <= 1,
                                    name=f"transfer_exclusivity_constraint_{i}")

                    # Incoming vessel indicator
                    incoming_expr = gp.LinExpr()
                    for j in range(num_vessels):
                        for ii in range(num_terminals):
                            if vessel_assignments[j, ii] == 1 and i != ii:
                                incoming_expr += vessel_assignment[j, i]

                    model.addConstr(incoming_expr <= big_M * receives_vessels[i], name=f"receives_indicator_upper_{i}")
                    model.addConstr(incoming_expr >= receives_vessels[i], name=f"receives_indicator_lower_{i}")

                    # Outgoing vessel indicator
                    outgoing_expr = gp.LinExpr()
                    for j in range(num_vessels):
                        for ii in range(num_terminals):
                            if vessel_assignments[j, i] == 1 and i != ii:
                                outgoing_expr += vessel_assignment[j, ii]

                    model.addConstr(outgoing_expr <= big_M * sends_vessels[i], name=f"sends_indicator_upper_{i}")
                    model.addConstr(outgoing_expr >= sends_vessels[i], name=f"sends_indicator_lower_{i}")

                # Global constraints
                # Each vessel assigned to exactly one terminal
                for j in range(num_vessels):
                    model.addConstr(gp.quicksum(vessel_assignment[j, i] for i in range(num_terminals)) == 1,
                                    name=f"vessel_assignment_constraint_{j}")

                # Conservation of total volume
                model.addConstr(
                    gp.quicksum(volume_after[i] for i in range(num_terminals)) ==
                    gp.quicksum(vessel_volumes[j] * vessel_assignments[j, i]
                                for j in range(num_vessels) for i in range(num_terminals)),
                    name="volume_conservation_constraint")

                # Conservation of vessel assignments
                model.addConstr(
                    gp.quicksum(vessel_assignments[j, i] for j in range(num_vessels) for i in range(num_terminals)) ==
                    gp.quicksum(vessel_assignment[j, i] for j in range(num_vessels) for i in range(num_terminals)),
                    name="assignment_conservation_constraint")

                # Capacity constraints
                for i in range(num_terminals):
                    model.addConstr(volume_after[i] <= terminal_capacities[i], name=f"capacity_upper_constraint_{i}")
                    model.addConstr(volume_after[i] >= 0.1 * terminal_capacities[i],
                                    name=f"capacity_lower_constraint_{i}")

                # Set objective
                if objective_name == 'MAXPROF':
                    model.setObjective(gp.quicksum(profit_terminal[i] for i in range(num_terminals)), GRB.MAXIMIZE)
                elif objective_name == 'MAXMIN':
                    for i in range(num_terminals):
                        model.addConstr(min_profit <= profit_terminal[i], name=f"min_profit_constraint_{i}")
                    model.setObjective(min_profit, GRB.MAXIMIZE)

                # Solve the model
                model.optimize()

                # Process results
                if model.status == GRB.OPTIMAL:
                    profit_after = np.array([profit_terminal[i].x for i in range(num_terminals)])
                    volume_after_values = np.array([volume_after[i].x for i in range(num_terminals)])
                    results[f'profitAfter_{objective_name}'] = profit_after
                    results[f'volumeAfter_{objective_name}'] = volume_after_values
                    results[f'objval_{objective_name}'] = model.objVal
                    results[f'optimalityGap_{objective_name}'] = model.MIPGap
                    logger.info(f"Optimization {objective_name} completed successfully")

                elif model.status == GRB.INFEASIBLE:
                    logger.warning(f"Optimization {objective_name} infeasible - no cooperation possible")
                    # Return exact original state (no cooperation)
                    results[f'profitAfter_{objective_name}'] = profit_before.copy()
                    results[f'volumeAfter_{objective_name}'] = volume_before.copy()
                    results[f'objval_{objective_name}'] = np.nan
                    results[f'optimalityGap_{objective_name}'] = np.nan
                    # Mark this objective as infeasible
                    results[f'{objective_name}_infeasible'] = True

                else:
                    logger.warning(f"Optimization {objective_name} failed with status: {model.status}")
                    # Return exact original state (optimization failed)
                    results[f'profitAfter_{objective_name}'] = profit_before.copy()
                    results[f'volumeAfter_{objective_name}'] = volume_before.copy()
                    results[f'objval_{objective_name}'] = np.nan
                    results[f'optimalityGap_{objective_name}'] = np.nan
                    # Mark this objective as failed
                    results[f'{objective_name}_failed'] = True

                # Clean up model
                model.dispose()

            except Exception as e:
                logger.error(f"Optimization {objective_name} failed: {e}")
                results[f'profitAfter_{objective_name}'] = profit_before.copy()
                results[f'volumeAfter_{objective_name}'] = volume_before.copy()
                results[f'objval_{objective_name}'] = np.nan
                results[f'optimalityGap_{objective_name}'] = np.nan

                # Determine overall feasibility status
                maxprof_infeasible = results.get('MAXPROF_infeasible', False)
                maxmin_infeasible = results.get('MAXMIN_infeasible', False)
                maxprof_failed = results.get('MAXPROF_failed', False)
                maxmin_failed = results.get('MAXMIN_failed', False)

                if maxprof_infeasible or maxmin_infeasible:
                    results['feasibility_status'] = 'No Cooperation - Mathematically Infeasible'
                elif maxprof_failed or maxmin_failed:
                    results['feasibility_status'] = 'No Cooperation - Solver Failed'
                else:
                    results['feasibility_status'] = 'Feasible and Sensible'

        # POST-OPTIMIZATION VALIDATION
        validation_results = validate_solution_consistency(input_data_ves, results)
        results['validation_metrics'] = validation_results

        if not validation_results['conservation_passed']:
            logger.warning(f"Solution failed validation checks:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")

            # Optionally mark as questionable but don't fail completely
            results['feasibility_status'] += ' - Validation Warnings'

        return results

    except Exception as e:
        logger.error(f"Critical optimization error: {e}")
        fallback = calculate_realistic_fallback(input_data_ves)
        if fallback is None:
            logger.error("Fallback calculation also failed")
            return None
        return fallback


def save_results_to_excel_with_retry(results: Dict[str, Any], filepath: str):
    """Saves results to Excel using a simpler, corruption-resistant approach."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df_data = []
    num_terminals = results['simulation_params']['num_terminals_from_data']
    subsidy = results['simulation_params']['subsidy']
    ci_terminals_str = results['simulation_params']['ci_terminals']

    for obj_name in ['MAXPROF', 'MAXMIN']:
        profit_before = results['optimization_output']['profitBefore']

        if ('feasibility_status' in results['optimization_output'] and
                'Infeasible' in results['optimization_output']['feasibility_status']):
            profit_after = profit_before
            volume_after = results['optimization_output']['volumeBefore']
            obj_value = np.nan
            optimality_gap = np.nan
        else:
            profit_after = results['optimization_output'][f'profitAfter_{obj_name}']
            volume_after = results['optimization_output'][f'volumeAfter_{obj_name}']
            obj_value = results['optimization_output'].get(f'objval_{obj_name}', np.nan)
            optimality_gap = results['optimization_output'].get(f'optimalityGap_{obj_name}', np.nan)

        profit_change = profit_after - profit_before

        # Handle dimension mismatch
        if isinstance(volume_after, np.ndarray) and volume_after.shape[0] < num_terminals:
            volume_after = np.pad(volume_after, (0, num_terminals - volume_after.shape[0]),
                                  'constant', constant_values=np.nan)

        # Add validation metrics if available
        validation_status = 'Not Validated'
        validation_warnings = 0
        if 'validation_metrics' in results['optimization_output']:
            vm = results['optimization_output']['validation_metrics']
            validation_status = 'Passed' if vm['conservation_passed'] else 'Failed'
            validation_warnings = len(vm['warnings'])

        for i in range(num_terminals):
            df_data.append({
                'Num_Terminals': num_terminals,
                'Subset_Composition': results['simulation_params']['subset_composition'],
                'CI_Rate_Data': results['simulation_params']['ci_rate_from_data'],
                'Instance_Index': results['simulation_params']['instance_index'],
                'Subsidy_Level': subsidy,
                'CI_Terminals_Combo': ci_terminals_str,
                'Terminal_ID': i + 1,
                'Objective': obj_name,
                'Profit_Before': profit_before[i],
                'Profit_After': profit_after[i],
                'Profit_Change': profit_change[i],
                'Volume_After': volume_after[i] if volume_after is not None else np.nan,
                'Feasibility_Status': results['optimization_output']['feasibility_status'],
                'Objective_Value': obj_value,
                'Optimality_Gap': optimality_gap,
                'Validation_Status': validation_status,
                'Validation_Warnings': validation_warnings
            })

    df = pd.DataFrame(df_data)

    # Use CSV instead of Excel to avoid corruption issues
    csv_filepath = filepath.replace('.xlsx', '.csv')

    try:
        # Simple append to CSV - much more reliable
        if os.path.exists(csv_filepath):
            df.to_csv(csv_filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filepath, mode='w', header=True, index=False)
        logger.info(f"Results saved to CSV successfully")
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")
        # Fallback: save with timestamp
        timestamp_file = csv_filepath.replace('.csv', f'_{int(time.time())}.csv')
        try:
            df.to_csv(timestamp_file, index=False)
            logger.info(f"Saved to timestamped file: {timestamp_file}")
        except Exception as e2:
            logger.error(f"All save attempts failed: {e2}")


def save_results(results: Dict[str, Any], filepath: str):
    """Saves the simulation results to a pickle file with error handling."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Results saved to '{filepath}'")
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")


def monitor_memory_usage(process: psutil.Process, start_memory: float, combination_count: int):
    """Monitor and log memory usage."""
    if combination_count % 100 == 0:
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - start_memory

        if memory_increase > 500:  # More than 500MB increase
            logger.warning(f"Memory usage: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")

        if current_memory > SimulationConfig.MAX_MEMORY_MB:
            logger.warning(f"High memory usage detected: {current_memory:.1f} MB")
            return True  # Signal to potentially run garbage collection

    return False


def run_all_combinations():
    """
    Main function to orchestrate the entire simulation with improved error handling and monitoring.
    """
    if not GUROBI_AVAILABLE:
        logger.error("Gurobi is not available. Cannot proceed with optimization.")
        return

    data_folder = "generated_data"
    results_folder = "simulation_results"
    excel_results_file = os.path.join(results_folder, "summary_results.xlsx")

    if not os.path.exists(data_folder):
        logger.error(f"The folder '{data_folder}' was not found.")
        logger.error("Please run the data generation script first to create the data files.")
        return

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    subsidies = np.array([0, 50, 100])
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
    data_files.sort()

    if not data_files:
        logger.error("No .pkl data files found in the 'generated_data' folder.")
        return

    # Calculate total combinations for progress tracking
    filename_pattern = re.compile(r'data_T(\d+)_subset_([\d_]+)_CI(\d+)_instance_(\d+)\.pkl')
    total_combinations = 0

    for filename in data_files:
        match = filename_pattern.match(filename)
        if match:
            num_terminals = int(match.groups()[0])
            total_combinations += len(subsidies) * (2 ** num_terminals)

    logger.info("=== STARTING ENHANCED SIMULATION RUNS ===")
    logger.info(f"Found {len(data_files)} data files to process")
    logger.info(f"Subsidy levels to test: {np.round(subsidies * 100)}%")
    logger.info(f"Total combinations to process: {total_combinations:,}")
    logger.info(f"Estimated runtime: {total_combinations * 0.1 / 60:.1f} minutes")

    # Memory monitoring setup
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    combination_count = 0
    successful_optimizations = 0
    failed_optimizations = 0
    validation_failures = 0

    # Process files with progress bar
    for filename in tqdm(data_files, desc="Processing data files"):
        filepath = os.path.join(data_folder, filename)
        match = filename_pattern.match(filename)

        if not match:
            logger.warning(f"Filename '{filename}' does not match expected pattern. Skipping.")
            continue

        num_terminals_str, subset_str, ci_rate_str, instance_idx_str = match.groups()
        num_terminals = int(num_terminals_str)
        ci_rate = int(ci_rate_str) / 100.0
        instance_idx = int(instance_idx_str)

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}. Skipping.")
            continue

        logger.info(
            f"\n--- Processing data for {num_terminals} terminals, CI Rate: {ci_rate:.0%}, Instance: {instance_idx} ---")

        # Process subsidies with progress bar
        for subsidy in tqdm(subsidies, desc="Processing subsidies", leave=False):
            num_ci_combinations = 2 ** num_terminals

            # Process CI combinations with progress bar
            for ci_combo_idx in tqdm(range(num_ci_combinations), desc="Processing CI combinations", leave=False):
                combination_count += 1

                # Memory monitoring
                if monitor_memory_usage(process, start_memory, combination_count):
                    import gc
                    gc.collect()

                ci_binary_str = format(ci_combo_idx, f'0{num_terminals}b')
                ci_terminals = np.array([int(c) for c in ci_binary_str])
                ci_terminals_list = np.where(ci_terminals == 1)[0]
                ci_terminals_str = '_'.join(
                    [str(t + 1) for t in ci_terminals_list]) if ci_terminals_list.size > 0 else 'None'

                logger.debug(f"  > Subsidy: {subsidy:.1%} | CI Combo: {ci_terminals_str}")

                # Prepare input data with validation
                input_data_ves = {
                    'subPar': subsidy,
                    'ciTerm': ci_terminals,
                    'num_terminals': num_terminals,
                    'v': data['vVes'],
                    'x': data['xV'],
                    'xCI': data['vVesCI'],
                    'c': data['inputData'][0, :, 0],
                    'cost_initial': data['inputData'][0, :, 1],
                    'cost_decrease_rate': data['inputData'][0, :, 2],
                    'cost_increase_rate': data['inputData'][0, :, 3],
                    'optimal_vcr_cost_point': data['inputData'][0, :, 4],
                    'initial_charge': data['inputData'][0, :, 5],
                    'decrease_rate': data['inputData'][0, :, 6],
                    'constant_vcr_charge': data['inputData'][0, :, 7],
                    'pc': data['inputData'][0, :, 8],
                }

                # Run optimization
                result = vessel_based_opt(input_data_ves)

                if result is None:
                    logger.error(f"Optimization failed critically for combination {combination_count}")
                    failed_optimizations += 1
                    continue

                if 'Feasible' in result.get('feasibility_status', ''):
                    successful_optimizations += 1

                    # Check validation results
                    if 'validation_metrics' in result and not result['validation_metrics']['conservation_passed']:
                        validation_failures += 1
                else:
                    failed_optimizations += 1

                # Save results
                results_to_save = {
                    'simulation_params': {
                        'filename': filename,
                        'subsidy': subsidy,
                        'ci_terminals': ci_terminals_str,
                        'ci_rate_from_data': ci_rate,
                        'num_terminals_from_data': num_terminals,
                        'instance_index': instance_idx,
                        'subset_composition': subset_str
                    },
                    'optimization_output': result
                }

                # Create output filename and save
                filename_base = os.path.splitext(filename)[0]
                output_filename = f"{filename_base}_Sub{int(subsidy * 100):02d}_CICombo_{ci_terminals_str}.pkl"
                output_filepath = os.path.join(results_folder, output_filename)

                save_results(results_to_save, output_filepath)
                save_results_to_excel_with_retry(results_to_save, excel_results_file)

    # Final summary
    logger.info("\n=== SIMULATION COMPLETE ===")
    logger.info(f"Total combinations processed: {combination_count:,}")
    logger.info(f"Successful optimizations: {successful_optimizations:,}")
    logger.info(f"Failed optimizations: {failed_optimizations:,}")
    logger.info(f"Validation failures: {validation_failures:,}")
    if combination_count > 0:
        logger.info(f"Success rate: {successful_optimizations / combination_count * 100:.1f}%")
        if successful_optimizations > 0:
            logger.info(f"Validation failure rate: {validation_failures / successful_optimizations * 100:.1f}%")

    final_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {final_memory:.1f} MB (increased by {final_memory - start_memory:.1f} MB)")


if __name__ == '__main__':
    run_all_combinations()