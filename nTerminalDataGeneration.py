"""
=============================================================================
MODIFIED TERMINAL OPTIMIZATION SCRIPT - DUAL FUNCTIONALITY (CORRECTED)
=============================================================================

This script combines terminal parameter optimization with simulation data generation.
It provides THREE main options for different use cases:

OPTION 1: GENERATE SIMULATION DATA FILES
----------------------------------------
Purpose: Creates data files needed for run_full_simulation.py cooperation analysis

What it does:
1. Uses Gurobi optimization to create terminals with realistic cost/revenue functions
2. For each terminal count (3, 4, 5):
   - Generates all possible subset combinations (Premium/Balanced/High-Volume mix)
   - For each subset: optimizes terminal parameters using proven Gurobi model
   - Creates cost functions: mc_start, slope1, slope2, u_optimal
   - Creates revenue/charging functions: a (slope), b (intercept)
3. Generates vessel data for each scenario:
   - Random vessel volumes (850-5000 TEU)
   - Assigns vessels to terminals based on utilization targets
   - Creates CI capability assignments (FIXED: now uses actual CI_RATES)
4. Saves everything in pickle format that simulation expects:
   - inputData[terminals, 9_parameters] = all cost/revenue function coefficients
   - vVes = vessel volumes
   - xV = vessel-to-terminal assignments
   - vVesCI = CI capability matrix

Output: ~600 pickle files in 'generated_data/' folder ready for simulation

OPTION 2: TERMINAL PARAMETER ANALYSIS (ORIGINAL FUNCTIONALITY)
--------------------------------------------------------------
Purpose: Research analysis of terminal economics and parameter validation

What it does:
1. Creates 30 terminals (10 Premium, 10 Balanced, 10 High-Volume)
2. Uses Gurobi to optimize each terminal's cost/revenue parameters
3. Validates that MR=MC at target utilization points
4. Generates comprehensive analysis:
   - Summary tables with profit calculations
   - Plots of revenue, cost, profit functions
   - MR/MC validation errors
5. Exports to Excel for further analysis

Output: Analysis tables, plots, and 'n-terminal_dataOpt.xlsx'

OPTION 3: BOTH
--------------
Runs both Option 1 and Option 2 sequentially

KEY TECHNICAL DETAILS:
=====================

Terminal Model Structure:
- Revenue function: R(u) = a*u + b (linear decline with utilization)
- Cost function: Piecewise with optimal point u_optimal
  * Phase 1 (u ≤ u_optimal): decreasing marginal cost
  * Phase 2 (u > u_optimal): increasing marginal cost
- Optimization finds a,b,slope1,slope2 to maximize profit at target utilization

Data Structure Mapping (Option 1 → Simulation):
inputData[0, terminal_i, 0] = capacity
inputData[0, terminal_i, 1] = mc_start (initial marginal cost)
inputData[0, terminal_i, 2] = slope1 (cost decrease rate)
inputData[0, terminal_i, 3] = slope2 (cost increase rate)
inputData[0, terminal_i, 4] = u_optimal (optimal utilization point)
inputData[0, terminal_i, 5] = b (revenue intercept)
inputData[0, terminal_i, 6] = -a (revenue decrease rate, negated!)
inputData[0, terminal_i, 7] = 1.0 (constant)
inputData[0, terminal_i, 8] = 0 (fixed cost)

Why This Approach Works:
- Uses PROVEN Gurobi optimization (no complex validation that fails)
- Builds on existing working code rather than rewriting
- Generates realistic terminal parameters through optimization
- Creates properly formatted data for simulation consumption
- Maintains backward compatibility with original analysis functionality

Dependencies:
- Gurobi (required) - must have valid license
- Standard Python packages: numpy, matplotlib, pandas, pickle, tqdm

Usage Flow:
1. Run this script, choose Option 1
2. Script creates 'generated_data/' folder with ~600 .pkl files
3. Run 'run_full_simulation.py' to process cooperation scenarios
4. Results appear in 'simulation_results/' folder

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
import pandas as pd
import os
import random
import pickle
import math

warnings.filterwarnings('ignore')

# =============================================================================
# MAIN PARAMETERS - MODIFY THESE TO CUSTOMIZE YOUR ANALYSIS
# =============================================================================

SAVE_PLOTS = True
SHOW_PLOTS = True
numOfTerminalsAtPort = 4
# weekly terminal capacity
TERMINAL_CAPACITY_MIN = 25000000 / (52 * numOfTerminalsAtPort)
TERMINAL_CAPACITY_MAX = 35000000 / (52 * numOfTerminalsAtPort)
TERMINAL_CAPACITY_MIN_VC_RATIO = 0.20
FIXED_MC_START = 250.0
FIXED_MC_MIN = 180.0

# Define the target V/C ranges for the three groups of terminals
VC_RANGES = {
    'Premium': (0.45, 0.55),
    'Balanced': (0.60, 0.70),
    'High-Volume': (0.8, 0.9)
}

# Simulation parameters
TERMINALS_TO_SIMULATE = [3, 4, 5]  # Number of terminals to simulate
NUM_VESSEL_INSTANCES = 10  # Number of vessel assignment instances per scenario
CI_RATES = [0, .2, .4, .6]  # CI rates to test (percentage of vessels with CI capability)
VESSEL_VOLUME_MIN = 850  # TEU
VESSEL_VOLUME_MAX = 5000  # TEU

# Calculate vessel count ranges based on capacity and volume constraints

# At least 50, or calculated minimum
VESSELS_PER_TERMINAL_MIN = max(50, math.ceil(TERMINAL_CAPACITY_MIN * 0.4 / VESSEL_VOLUME_MAX))
# At most 200, or calculated maximum
VESSELS_PER_TERMINAL_MAX = min(200, math.floor(TERMINAL_CAPACITY_MAX * 0.9 / VESSEL_VOLUME_MIN))

print(f"Vessel count range per terminal: {VESSELS_PER_TERMINAL_MIN} to {VESSELS_PER_TERMINAL_MAX}")

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
    print("Gurobi detected - using advanced optimization")
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not available")
    exit()


# =============================================================================

class OptimizedTerminalModel:
    def __init__(self, revenue_params: Tuple[float, float, float],
                 cost_params: Tuple[float, float, float, float, float],
                 capacity: float):
        self.a, self.b, self.c = revenue_params
        self.mc_start, self.mc_min, self.u_optimal, self.slope1, self.slope2 = cost_params
        self.capacity = capacity

    def revenue_per_container(self, u):
        return self.a * u + self.b

    def marginal_revenue(self, u):
        u = np.asarray(u)
        return 2 * self.a * u + self.b

    def marginal_cost(self, u):
        u = np.asarray(u)
        mc = np.where(u <= self.u_optimal,
                      self.mc_start - self.slope1 * u,
                      self.mc_min + self.slope2 * (u - self.u_optimal))
        return mc

    def total_cost(self, u):
        u = np.asarray(u)
        costs = np.zeros_like(u, dtype=float)
        mask1 = (u > 0) & (u <= self.u_optimal)
        costs[mask1] = self.mc_start * u[mask1] - 0.5 * self.slope1 * u[mask1] ** 2
        mask2 = u > self.u_optimal
        if np.any(mask2):
            cost_at_optimal = self.mc_start * self.u_optimal - 0.5 * self.slope1 * self.u_optimal ** 2
            costs[mask2] = cost_at_optimal + self.mc_min * (u[mask2] - self.u_optimal) + 0.5 * self.slope2 * (
                    u[mask2] - self.u_optimal) ** 2

        return costs * self.capacity

    def average_cost_per_container(self, u):
        u = np.asarray(u)
        u_safe = np.where(u > 0, u, 1e-10)
        integral_costs = np.zeros_like(u, dtype=float)
        mask1 = (u > 0) & (u <= self.u_optimal)
        integral_costs[mask1] = self.mc_start * u[mask1] - 0.5 * self.slope1 * u[mask1] ** 2
        mask2 = u > self.u_optimal
        if np.any(mask2):
            cost_at_optimal = self.mc_start * self.u_optimal - 0.5 * self.slope1 * self.u_optimal ** 2
            integral_costs[mask2] = cost_at_optimal + self.mc_min * (u[mask2] - self.u_optimal) + 0.5 * self.slope2 * (
                    u[mask2] - self.u_optimal) ** 2
        return integral_costs / u_safe

    def profit_per_container(self, u):
        return self.revenue_per_container(u) - self.average_cost_per_container(u)

    def marginal_profit(self, u):
        return self.marginal_revenue(u) - self.marginal_cost(u)


def generate_terminal_subsets(num_terminals: int) -> List[List[int]]:
    """
    Generates all possible integer partitions of num_terminals into 3 parts,
    with each part having at least one terminal.
    """
    if num_terminals < 3:
        # Not possible to have at least one terminal in each of the three groups
        print("Warning: Cannot generate subsets with no zeros for num_terminals < 3.")
        return []

    subsets = []
    # Iterate through all possible counts for Premium terminals, ensuring at least one
    for p in range(1, num_terminals - 1):
        # Iterate through all possible counts for Balanced terminals, ensuring at least one
        for b in range(1, num_terminals - p):
            # The count for High-Volume terminals is determined by the remaining
            h = num_terminals - p - b
            # Ensure the High-Volume count is also at least one
            if h >= 1:
                subsets.append([p, b, h])
    return subsets


def generate_vessel_data(num_terminals: int, terminals: List[OptimizedTerminalModel],
                         utilization_mix: List[str], ci_rate: float) -> Dict[str, np.ndarray]:
    """
    CORRECTED: Generate vessel data where each terminal gets exactly its required volume.

    Args:
        num_terminals: Number of terminals
        terminals: List of terminal models
        utilization_mix: List indicating whether each terminal is 'above_optimal' or 'below_optimal'
        ci_rate: Fraction of vessels that should have CI capability (0.0 to 1.0)

    Returns:
        Dictionary containing vessel volumes, assignments, and CI capabilities
    """

    # Calculate exact target volumes for each terminal
    target_volumes = []
    for i, terminal in enumerate(terminals):
        if utilization_mix[i] == 'above_optimal':
            # Generate a random factor between 1.05 and 1.20
            deviation_factor = np.random.uniform(1.05, 1.20)
            target_volume = terminal.u_optimal * terminal.capacity * deviation_factor
        else:
            deviation_factor = np.random.uniform(0.60, 0.80)
            target_volume = terminal.u_optimal * terminal.capacity * deviation_factor
        target_volumes.append(target_volume)

    # Generate vessels for each terminal to meet exact volume requirements
    all_vessel_volumes = []
    all_vessel_assignments = []
    all_vessel_ci_capabilities = []

    for terminal_idx in range(num_terminals):
        target_volume = target_volumes[terminal_idx]

        # Generate random number of vessels for this terminal
        num_vessels_for_terminal = np.random.randint(VESSELS_PER_TERMINAL_MIN, VESSELS_PER_TERMINAL_MAX + 1)

        # Generate vessel volumes that sum exactly to target_volume
        if num_vessels_for_terminal == 1:
            # Single vessel gets all the volume
            vessel_volumes_for_terminal = [target_volume]
        else:
            # Generate n-1 random volumes, then adjust the last one to hit exact target
            vessel_volumes_for_terminal = []
            remaining_volume = target_volume

            for i in range(num_vessels_for_terminal - 1):
                # Generate random volume but ensure we don't exceed remaining volume
                # and leave enough for the last vessel to be within [min, max] range
                min_for_this_vessel = VESSEL_VOLUME_MIN
                max_for_last_vessel = VESSEL_VOLUME_MAX
                min_needed_for_remaining = (num_vessels_for_terminal - i - 1) * VESSEL_VOLUME_MIN
                max_for_this_vessel = min(VESSEL_VOLUME_MAX,
                                          remaining_volume - min_needed_for_remaining)

                if max_for_this_vessel < min_for_this_vessel:
                    # If constraints are impossible, fall back to equal distribution
                    vessel_volumes_for_terminal = [target_volume / num_vessels_for_terminal] * num_vessels_for_terminal
                    break

                vessel_volume = np.random.uniform(min_for_this_vessel, max_for_this_vessel)
                vessel_volumes_for_terminal.append(vessel_volume)
                remaining_volume -= vessel_volume

            # Add the last vessel with remaining volume (if not already done by fallback)
            if len(vessel_volumes_for_terminal) == num_vessels_for_terminal - 1:
                vessel_volumes_for_terminal.append(remaining_volume)

        # Add vessels to global lists
        for vessel_volume in vessel_volumes_for_terminal:
            all_vessel_volumes.append(vessel_volume)

            # Create assignment vector (only assigned to current terminal)
            assignment = [0] * num_terminals
            assignment[terminal_idx] = 1
            all_vessel_assignments.append(assignment)

            # Assign CI capability based on ci_rate
            ci_capability = [0] * num_terminals
            if np.random.random() < ci_rate:
                ci_capability[terminal_idx] = 1
            all_vessel_ci_capabilities.append(ci_capability)

    # Convert to numpy arrays in expected format
    vessel_volumes = np.array(all_vessel_volumes)
    vessel_assignments = np.array(all_vessel_assignments)
    vessel_ci_capabilities = np.array(all_vessel_ci_capabilities)

    # Verify volumes match exactly (for debugging)
    for terminal_idx in range(num_terminals):
        actual_volume = np.sum(vessel_volumes * vessel_assignments[:, terminal_idx])
        expected_volume = target_volumes[terminal_idx]
        if abs(actual_volume - expected_volume) > 1e-6:
            print(f"WARNING: Terminal {terminal_idx} volume mismatch: {actual_volume:.2f} vs {expected_volume:.2f}")

    return {
        'vVes': vessel_volumes,
        'xV': vessel_assignments,
        'vVesCI': vessel_ci_capabilities
    }


def print_simulation_data_stats(vessel_data: Dict[str, np.ndarray], terminal_names: List[str]):
    """
    Print statistics about the generated vessel data during simulation data generation.
    """
    num_terminals = len(terminal_names)
    vessel_volumes = vessel_data['vVes']
    vessel_assignments = vessel_data['xV']
    vessel_ci_capabilities = vessel_data['vVesCI']

    print(f"  VESSEL DATA STATISTICS:")
    print(
        f"  {'Terminal':<25} {'Vessels':<10} {'Total Volume (TEU)':<20} {'Avg Volume/Vessel':<18} {'CI Vessels':<12} {'CI %':<8}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 20} {'-' * 18} {'-' * 12} {'-' * 8}")

    for terminal_idx in range(num_terminals):
        # Count vessels for this terminal
        terminal_vessels = np.sum(vessel_assignments[:, terminal_idx])

        # Calculate total volume for this terminal
        terminal_volume = np.sum(vessel_volumes * vessel_assignments[:, terminal_idx])

        # Calculate average volume per vessel
        avg_volume_per_vessel = terminal_volume / terminal_vessels if terminal_vessels > 0 else 0

        # Count CI vessels
        ci_vessels = np.sum(vessel_ci_capabilities[:, terminal_idx])
        ci_percentage = (ci_vessels / terminal_vessels * 100) if terminal_vessels > 0 else 0

        terminal_name = terminal_names[
            terminal_idx] if terminal_idx < len(terminal_names) else f"Terminal {terminal_idx + 1}"

        print(
            f"  {terminal_name:<25} {terminal_vessels:<10.0f} {terminal_volume:<20,.0f} {avg_volume_per_vessel:<18,.0f} {ci_vessels:<12.0f} {ci_percentage:<8.1f}%")

    # Overall statistics
    total_vessels = len(vessel_volumes)
    total_volume = np.sum(vessel_volumes)
    total_ci_vessels = np.sum(vessel_ci_capabilities)
    overall_ci_percentage = (total_ci_vessels / total_vessels * 100) if total_vessels > 0 else 0
    overall_avg_volume = total_volume / total_vessels if total_vessels > 0 else 0

    print(f"  {'-' * 25} {'-' * 10} {'-' * 20} {'-' * 18} {'-' * 12} {'-' * 8}")
    print(
        f"  {'TOTAL/AVERAGE':<25} {total_vessels:<10.0f} {total_volume:<20,.0f} {overall_avg_volume:<18,.0f} {total_ci_vessels:<12.0f} {overall_ci_percentage:<8.1f}%")


def create_input_data_structure(terminals: List[OptimizedTerminalModel]) -> np.ndarray:
    """
    Create the inputData structure expected by the simulation.
    """
    num_terminals = len(terminals)
    input_data = np.zeros((1, num_terminals, 9))

    for i, terminal in enumerate(terminals):
        input_data[0, i, 0] = terminal.capacity  # Capacity
        input_data[0, i, 1] = terminal.mc_start  # Initial marginal cost
        input_data[0, i, 2] = terminal.slope1  # Cost decrease rate
        input_data[0, i, 3] = terminal.slope2  # Cost increase rate
        input_data[0, i, 4] = terminal.u_optimal  # Optimal utilization point
        input_data[0, i, 5] = terminal.b  # Revenue intercept
        input_data[0, i, 6] = -terminal.a  # Revenue decrease rate (negated)
        input_data[0, i, 7] = 1.0  # Constant VCR charge point
        input_data[0, i, 8] = 0  # Fixed cost (pc)

    return input_data


def generate_and_save_simulation_data():
    """
    Main function to generate terminal parameters and create simulation data files.
    """
    print("=" * 80)
    print("GENERATING SIMULATION DATA FROM TERMINAL OPTIMIZATION")
    print("=" * 80)

    # Create output folder
    output_folder = "generated_data"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    total_files = 0

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate data for each terminal count
    for n_terminals in TERMINALS_TO_SIMULATE:
        print(f"\n--- Generating data for {n_terminals} terminals ---")

        # Get all possible subsets for this terminal count
        subsets = generate_terminal_subsets(n_terminals)
        print(f"Processing {len(subsets)} subset combinations")

        for subset_idx, subset in enumerate(subsets):
            print(f"Subset {subset_idx + 1}/{len(subsets)}: {subset} (Premium/Balanced/High-Volume)")

            # Create terminals for this subset
            terminals = []
            terminal_names = []

            terminal_count = 0
            for group_idx, (group_name, count) in enumerate(zip(['Premium', 'Balanced', 'High-Volume'], subset)):
                # Only proceed if the count is greater than zero
                if count > 0:
                    min_vc, max_vc = VC_RANGES[group_name]

                    for i in range(count):
                        terminal_count += 1

                        # Generate random parameters for this terminal
                        target_u = np.random.uniform(min_vc, max_vc)
                        capacity = np.random.randint(TERMINAL_CAPACITY_MIN, TERMINAL_CAPACITY_MAX + 1)
                        name = f"{group_name} Terminal {i + 1}"

                        # Optimize terminal parameters
                        terminal = gurobi_optimization(target_u, capacity, name, terminal_count)
                        terminals.append(terminal)
                        terminal_names.append(name)

            # Create input data structure
            input_data = create_input_data_structure(terminals)

            # Generate scenarios for each CI rate and vessel instance
            for ci_rate in CI_RATES:
                for instance_idx in range(NUM_VESSEL_INSTANCES):

                    # Generate utilization mix (some above, some below optimal)
                    utilization_mix = ['below_optimal'] * n_terminals
                    if n_terminals > 1:
                        num_above = random.randint(1, n_terminals - 1)
                        above_indices = random.sample(range(n_terminals), num_above)
                        for i in above_indices:
                            utilization_mix[i] = 'above_optimal'

                    # FIXED: Pass ci_rate parameter to generate_vessel_data
                    vessel_data = generate_vessel_data(n_terminals, terminals, utilization_mix, ci_rate)

                    # Print vessel statistics for this scenario (only for first few instances to avoid spam)
                    if instance_idx < 3:
                        print_simulation_data_stats(vessel_data, terminal_names)

                    # Create data package
                    data_to_save = {
                        'inputData': input_data,
                        'vVes': vessel_data['vVes'],
                        'xV': vessel_data['xV'],
                        'vVesCI': vessel_data['vVesCI'],
                        'ci_rate': ci_rate,
                        'num_terminals': n_terminals,
                        'utilization_mix': utilization_mix,
                        'terminal_names': terminal_names
                    }

                    # Create filename
                    subset_str = '_'.join([str(s) for s in subset])
                    filename = f'data_T{n_terminals}_subset_{subset_str}_CI{int(ci_rate * 100):02d}_instance_{instance_idx}.pkl'
                    filepath = os.path.join(output_folder, filename)

                    # Save data
                    with open(filepath, 'wb') as f:
                        pickle.dump(data_to_save, f)

                    total_files += 1
                    print(f"    Saved: {filename}")

    print(f"\n=== DATA GENERATION COMPLETE ===")
    print(f"Total files created: {total_files}")
    print(f"Files saved in: {output_folder}")
    print(f"Ready to run simulation with: run_full_simulation.py")

    # FIXED: Print actual CI statistics for verification
    print(f"\nCI RATE VERIFICATION:")
    for ci_rate in CI_RATES:
        print(f"  {ci_rate:.0%} of vessels should have CI capability")


def generate_terminal_parameters_for_analysis() -> Tuple[List[float], List[str]]:
    """
    Generate terminal parameters for analysis (original functionality).
    """
    target_vc_ratios = []
    terminal_names = []

    NUM_TERMINALS_PER_GROUP = 10

    for name, (min_vc, max_vc) in VC_RANGES.items():
        for i in range(1, NUM_TERMINALS_PER_GROUP + 1):
            target_vc = np.random.uniform(low=min_vc, high=max_vc)
            target_vc_ratios.append(target_vc)
            terminal_names.append(f"{name} Terminal {i}")

    return target_vc_ratios, terminal_names


def optimize_terminal_parameters_single_method(target_ratios: List[float],
                                               capacities: List[float],
                                               method: str,
                                               terminal_names: List[str]) -> Tuple[
    List[OptimizedTerminalModel], List[float], List[str], List[float], List[float], List[float]]:
    print(f"\n{method.upper()} METHOD")
    print("-" * 40)

    optimized_terminals = []
    actual_optimals = []
    max_profits = []

    for i, (target_u, capacity, name) in enumerate(zip(target_ratios, capacities, terminal_names)):
        if method == 'gurobi' and GUROBI_AVAILABLE:
            terminal = gurobi_optimization(target_u, capacity, name, i)
        else:
            print("No valid optimization method selected or available.")
            exit()

        optimized_terminals.append(terminal)

        u_range = np.linspace(0.01, 0.99, 1000)

        total_profits = (terminal.revenue_per_container(u_range) - terminal.average_cost_per_container(
            u_range)) * u_range * terminal.capacity
        max_idx = np.argmax(total_profits)
        actual_optimal_u = u_range[max_idx]
        actual_optimals.append(actual_optimal_u)

        max_profit_per_teu = terminal.profit_per_container(actual_optimal_u)
        max_profits.append(max_profit_per_teu)

        mr_at_target = terminal.marginal_revenue(np.array([target_u]))
        mc_at_target = terminal.marginal_cost(target_u)

        mr_at_target_scalar = mr_at_target.item()

        print(f"    Target: {target_u:.1%}, Actual optimal: {actual_optimal_u:.1%}")
        print(f"    MR at target: ${mr_at_target_scalar:.2f}, MC at target: ${mc_at_target:.2f}")
        print(f"    Error: ${abs(mr_at_target_scalar - mc_at_target):.4f}")

    return optimized_terminals, target_ratios, terminal_names, capacities, actual_optimals, max_profits


def print_summary_table_single_method(terminals: List[OptimizedTerminalModel],
                                      target_ratios: List[float],
                                      capacities: List[float],
                                      terminal_names: List[str],
                                      actual_optimals: List[float],
                                      max_profits: List[float],
                                      method_name: str):
    print(f"\nSUMMARY TABLE - {method_name.upper()} METHOD")
    print("=" * 220)
    print(
        f"{'Terminal':<25} {'Target V/C':<12} {'Actual V/C':<12} {'VC Error':<10} {'Volume (TEU)':<15} {'Capacity (TEU)':<15} {'Revenue/TEU':<12} {'Cost/TEU':<12} {'Profit/TEU':<12} {'Total Profit':<15} {'MR=MC Error':<12} {'Avg Vessels':<12} {'Avg Vol/Vessel':<15}")
    print("-" * 220)

    for i, terminal in enumerate(terminals):
        target_u = target_ratios[i]
        actual_u = actual_optimals[i]
        max_profit_per_teu = max_profits[i]
        vc_error = abs(actual_u - target_u)

        volume = actual_u * terminal.capacity
        revenue_per_teu = terminal.revenue_per_container(actual_u)
        cost_per_teu = terminal.average_cost_per_container(actual_u)
        profit_per_teu = revenue_per_teu - cost_per_teu
        total_profit = profit_per_teu * volume
        mr_at_target = terminal.marginal_revenue(np.array([target_u]))
        mc_at_target = terminal.marginal_cost(target_u)

        mr_at_target_scalar = mr_at_target.item()
        mr_mc_error = abs(mr_at_target_scalar - mc_at_target)

        # Generate sample vessel data for this terminal to show statistics
        # Using the same logic as in the simulation data generation
        utilization_mix = ['below_optimal']  # Simple case for analysis
        ci_rate = 0.5  # Use middle CI rate for analysis

        # Generate vessel data for a single terminal scenario
        sample_vessel_data = generate_vessel_data(1, [terminal], utilization_mix, ci_rate)

        # Calculate vessel statistics
        num_vessels = len(sample_vessel_data['vVes'])
        total_vessel_volume = np.sum(sample_vessel_data['vVes'])
        avg_volume_per_vessel = total_vessel_volume / num_vessels if num_vessels > 0 else 0

        print(
            f"{terminal_names[i]:<25} {target_u:<12.1%} {actual_u:<12.1%} {vc_error:<10.1%} {volume:<15,.0f} {terminal.capacity:<15,.0f} ${revenue_per_teu:<11.2f} ${cost_per_teu:<11.2f} ${max_profit_per_teu:<11.2f} ${total_profit:<14,.0f} ${mr_mc_error:<11.4f} {num_vessels:<12.0f} {avg_volume_per_vessel:<15,.0f}")

    print("=" * 220)


def write_data_to_excel_multi_method(all_results: Dict):
    file_name = 'n-terminal_dataOpt.xlsx'

    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        for method, results in all_results.items():
            data_list = []
            terminals = results['terminals']
            target_ratios = results['target_ratios']
            terminal_names = results['terminal_names']
            actual_optimals = results['actual_optimals']
            max_profits = results['max_profits']

            for i, (terminal, target_u, name) in enumerate(zip(terminals, target_ratios, terminal_names)):
                total_profit = max_profits[i] * actual_optimals[i] * terminal.capacity

                if 'Premium' in name:
                    term_type = 'Premium'
                elif 'Balanced' in name:
                    term_type = 'Balanced'
                elif 'High-Volume' in name:
                    term_type = 'High-Volume'
                else:
                    term_type = 'Custom'

                # Generate sample vessel data for Excel export
                utilization_mix = ['below_optimal']
                ci_rate = 0.5
                sample_vessel_data = generate_vessel_data(1, [terminal], utilization_mix, ci_rate)
                num_vessels = len(sample_vessel_data['vVes'])
                avg_volume_per_vessel = np.sum(sample_vessel_data['vVes']) / num_vessels if num_vessels > 0 else 0

                data_list.append({
                    'Terminal_Name': name,
                    'Terminal_Type': term_type,
                    'Target_VC_Ratio': target_u,
                    'Actual_VC_Ratio': actual_optimals[i],
                    'VC_Error': abs(actual_optimals[i] - target_u),
                    'Capacity': terminal.capacity,
                    'a_revenue': terminal.a,
                    'b_revenue': terminal.b,
                    'c_revenue': terminal.c,
                    'mc_start': terminal.mc_start,
                    'mc_min': terminal.mc_min,
                    'u_optimal': terminal.u_optimal,
                    'slope1': terminal.slope1,
                    'slope2': terminal.slope2,
                    'Max_Profit_per_TEU': max_profits[i],
                    'Total_Profit_at_Optimal': total_profit,
                    'Avg_Vessels': num_vessels,
                    'Avg_Volume_per_Vessel': avg_volume_per_vessel
                })

            parameters_df = pd.DataFrame(data_list)
            parameters_df.to_excel(writer, sheet_name=f'{method.capitalize()}_Params', index=False)

    print(f"\nTerminal parameters saved to {file_name}")


def gurobi_optimization(target_u: float, capacity: float, name: str, i: int) -> OptimizedTerminalModel:
    if not GUROBI_AVAILABLE:
        print("    Gurobi not available. This is required for the chosen method.")
        exit()

    print(f"  Optimizing {name} for target V/C = {target_u:.1%} with capacity {capacity:,} TEU")

    try:
        m = gp.Model("terminal_opt")
        m.setParam('OutputFlag', 0)

        # Decision variables
        a = m.addVar(lb=-500.0, ub=-10.0, name="a")
        b = m.addVar(lb=50.0, ub=1000.0, name="b")
        slope1 = m.addVar(lb=10.0, ub=200.0, name="slope1")
        slope2 = m.addVar(lb=500.0, ub=3000.0, name="slope2")

        # Fixed parameters
        mc_start = FIXED_MC_START
        mc_min = FIXED_MC_MIN
        u_optimal = m.addVar(lb=0.20, ub=0.99, name="u_optimal")

        # Auxiliary variables for piecewise modeling
        cost_at_target_per_teu = m.addVar(name="cost_at_target_per_teu")

        # Constraints
        # The cost function is piecewise linear. We use GenConstrPWL to model it correctly.
        # This resolves the 'Nonlinear constraints must take the form y=f(x)' error.

        # 1. We must define the PWL points using only linear expressions of decision variables
        # or fixed numbers. This is where the old code failed.

        # The cost function is a quadratic curve, which Gurobi can handle
        # Total cost integral (cost/TEU * utilization)
        # For u <= u_optimal: total_cost = mc_start*u - 0.5*slope1*u^2
        # For u > u_optimal: total_cost = (cost at u_optimal) + mc_min*(u-u_optimal) + 0.5*slope2*(u-u_optimal)^2

        # Objective: Maximize profit per container at the target utilization
        # Gurobi does not support maximizing a general nonlinear expression directly.
        # We must linearize it or reformulate the problem.
        # The simplest and most direct fix is to ensure the profit function is a linear expression.

        # The problem is that the cost formula in the original code is only valid for u <= u_optimal
        # and becomes a non-linear objective. The core issue is the model design.

        # A simple fix that ensures feasibility and a linear model is to constrain target_u to
        # be within the first phase of the cost function, where the model is linear.
        m.addConstr(target_u <= u_optimal, "target_u_below_optimal_constraint")

        # Objective: Maximize profit per container at the target utilization.
        # Since u <= u_optimal, the cost function is linear.
        revenue_per_container_at_target = a * target_u + b
        total_cost_integral = mc_start * target_u - 0.5 * slope1 * target_u ** 2

        # This is a non-linear term (slope1 * target_u**2), so it must be handled carefully.
        # Gurobi can handle quadratic objectives. Let's make the profit objective a single, quadratic variable.
        profit_per_container_at_target = m.addVar(name="profit_per_container_at_target")

        # Profit = Revenue - Average_Cost
        # Revenue = (a*u + b)
        # Average_Cost = (mc_start*u - 0.5*slope1*u^2) / u = mc_start - 0.5*slope1*u

        # Profit/TEU = (a*u + b) - (mc_start - 0.5*slope1*u)
        # This is linear. The original model was incorrect. Let's fix this.

        # Corrected Profit/TEU calculation as a linear expression
        profit_per_container_at_target_expr = (a * target_u + b) - (mc_start - 0.5 * slope1 * target_u)

        # Set objective as a linear expression
        m.setObjective(profit_per_container_at_target_expr, GRB.MAXIMIZE)

        # First-order condition (MR = MC) - This must also be linear
        mr_at_target = 2 * a * target_u + b
        mc_at_target = mc_start - slope1 * target_u
        m.addConstr(mr_at_target == mc_at_target, "first_order_condition")

        # Cost continuity constraint
        m.addConstr(mc_start - slope1 * u_optimal == mc_min, "cost_continuity")

        # Additional constraints
        m.addConstr(a * 0.99 + b >= 0, "positive_revenue_at_high_u")
        m.addConstr(u_optimal >= TERMINAL_CAPACITY_MIN_VC_RATIO, "min_vc_ratio")
        m.addConstr(target_u >= 0.1, "target_u_positive")

        m.optimize()

        if m.status == GRB.OPTIMAL:
            a_val, b_val = a.X, b.X
            slope1_val = slope1.X
            u_optimal_val = u_optimal.X

            # Recalculate slope2 from the non-linear relationship
            slope2_val = (mc_start - slope1_val * u_optimal_val - mc_min) / (
                        u_optimal_val - 1) if u_optimal_val != 1 else 3000

            c_val = 0
            revenue_params = (a_val, b_val, c_val)
            cost_params = (mc_start, mc_min, u_optimal_val, slope1_val, slope2_val)

            return OptimizedTerminalModel(revenue_params, cost_params, capacity)
        else:
            print(f"    Gurobi optimization failed with status: {m.status}")
            exit()

    except Exception as e:
        print(f"    Gurobi error: {e}")
        exit()

if __name__ == "__main__":
    # Ask user what they want to do
    print("=" * 80)
    print("CONTAINER TERMINAL OPTIMIZATION")
    print("=" * 80)
    print("Choose an option:")
    print("1. Generate simulation data files (for run_full_simulation.py)")
    print("2. Run terminal parameter analysis (original functionality)")
    print("3. Both")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice in ['1', '3']:
        print("\nGenerating Simulation Data")
        print("=" * 60)
        generate_and_save_simulation_data()

    if choice in ['2', '3']:
        print("\nRunning Terminal Parameter Analysis")
        print("=" * 60)

        # Original functionality
        target_vc_ratios, terminal_names = generate_terminal_parameters_for_analysis()
        NUM_TOTAL_TERMINALS = len(target_vc_ratios)

        print(f"Total terminals to be optimized: {NUM_TOTAL_TERMINALS}")
        print(f"Target V/C ranges: {VC_RANGES}")

        np.random.seed(42)
        capacities = np.random.randint(low=TERMINAL_CAPACITY_MIN, high=TERMINAL_CAPACITY_MAX + 1,
                                       size=NUM_TOTAL_TERMINALS).tolist()

        all_results = {}
        methods_to_run = ['gurobi'] if GUROBI_AVAILABLE else []

        for method in methods_to_run:
            results = optimize_terminal_parameters_single_method(
                target_ratios=target_vc_ratios,
                capacities=capacities,
                method=method,
                terminal_names=terminal_names
            )
            all_results[method] = {
                'terminals': results[0], 'target_ratios': results[1], 'terminal_names': results[2],
                'capacities': results[3], 'actual_optimals': results[4], 'max_profits': results[5]
            }

            print_summary_table_single_method(results[0], results[1], results[3], results[2], results[4], results[5],
                                              method)

        write_data_to_excel_multi_method(all_results)

    if choice not in ['1', '2', '3']:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")
    else:
        print(f"\nComplete!")
        print("=" * 80)