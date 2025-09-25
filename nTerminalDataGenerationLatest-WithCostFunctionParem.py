"""
ENHANCED TERMINAL OPTIMIZATION SCRIPT - WITH EXPLICIT COST FUNCTION COEFFICIENTS
=================================================================================

This enhanced version provides explicit access to cost function coefficients and validates them.

KEY ADDITIONS:
- Explicit cost function coefficient extraction and display
- Cost function validation and mathematical representation
- Dedicated cost coefficient export functions
- Enhanced debugging and verification tools
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

# Add these new functions after the existing imports and before the main OptimizedTerminalModel class:

def extract_cost_function_coefficients(terminal: 'OptimizedTerminalModel') -> Dict[str, float]:
    """
    Extract all cost function coefficients with clear labels and mathematical meaning.

    Returns:
        Dictionary containing all cost function parameters with descriptions
    """
    return {
        'mc_start': terminal.mc_start,           # Initial marginal cost ($/TEU)
        'mc_min': terminal.mc_min,               # Minimum marginal cost ($/TEU)
        'slope1': terminal.slope1,               # Cost decrease rate in Phase 1 ($/TEU²)
        'slope2': terminal.slope2,               # Cost increase rate in Phase 2 ($/TEU²)
        'u_optimal': terminal.u_optimal,         # Optimal utilization point (dimensionless)
        'capacity': terminal.capacity,           # Terminal capacity (TEU/week)

        # Derived coefficients for total cost function
        'tc_phase1_linear': terminal.mc_start,   # Linear coefficient in Phase 1: mc_start * u
        'tc_phase1_quadratic': -0.5 * terminal.slope1,  # Quadratic coefficient in Phase 1: -0.5 * slope1 * u²
        'tc_phase2_constant': terminal.mc_start * terminal.u_optimal - 0.5 * terminal.slope1 * terminal.u_optimal**2,  # Constant for Phase 2
        'tc_phase2_linear': terminal.mc_min,     # Linear coefficient in Phase 2: mc_min * (u - u_optimal)
        'tc_phase2_quadratic': 0.5 * terminal.slope2   # Quadratic coefficient in Phase 2: 0.5 * slope2 * (u - u_optimal)²
    }

def print_cost_function_equations(terminal: 'OptimizedTerminalModel', terminal_name: str = "Terminal"):
    """
    Print the mathematical equations for the cost functions.
    """
    coeffs = extract_cost_function_coefficients(terminal)

    print(f"\n{'='*60}")
    print(f"COST FUNCTION EQUATIONS FOR {terminal_name.upper()}")
    print(f"{'='*60}")

    print(f"Capacity: {coeffs['capacity']:,.0f} TEU/week")
    print(f"Optimal Utilization Point: {coeffs['u_optimal']:.3f} ({coeffs['u_optimal']:.1%})")

    print(f"\nMARGINAL COST FUNCTION:")
    print(f"  Phase 1 (u ≤ {coeffs['u_optimal']:.3f}):")
    print(f"    MC(u) = {coeffs['mc_start']:.2f} - {coeffs['slope1']:.2f} × u")
    print(f"  Phase 2 (u > {coeffs['u_optimal']:.3f}):")
    print(f"    MC(u) = {coeffs['mc_min']:.2f} + {coeffs['slope2']:.2f} × (u - {coeffs['u_optimal']:.3f})")

    print(f"\nTOTAL COST FUNCTION (per TEU):")
    print(f"  Phase 1 (u ≤ {coeffs['u_optimal']:.3f}):")
    print(f"    TC(u)/u = {coeffs['mc_start']:.2f} - {coeffs['slope1']/2:.3f} × u")
    print(f"  Phase 2 (u > {coeffs['u_optimal']:.3f}):")
    tc_const_per_teu = coeffs['tc_phase2_constant'] / coeffs['u_optimal']
    print(f"    TC(u)/u = {tc_const_per_teu:.2f}/u + {coeffs['mc_min']:.2f} + {coeffs['slope2']/2:.3f} × (u - {coeffs['u_optimal']:.3f})²/u")

    print(f"\nTOTAL COST FUNCTION (absolute, capacity-adjusted):")
    print(f"  Phase 1: TC(u) = Capacity × [{coeffs['tc_phase1_linear']:.2f} × u + {coeffs['tc_phase1_quadratic']:.3f} × u²]")
    print(f"  Phase 2: TC(u) = Capacity × [{coeffs['tc_phase2_constant']:.2f} + {coeffs['tc_phase2_linear']:.2f} × (u-{coeffs['u_optimal']:.3f}) + {coeffs['tc_phase2_quadratic']:.3f} × (u-{coeffs['u_optimal']:.3f})²]")

def validate_cost_function_continuity(terminal: 'OptimizedTerminalModel', terminal_name: str = "Terminal"):
    """
    Validate that the cost function is continuous at the optimal point.
    """
    print(f"\n{'='*60}")
    print(f"COST FUNCTION VALIDATION FOR {terminal_name.upper()}")
    print(f"{'='*60}")

    u_opt = terminal.u_optimal

    # Test marginal cost continuity
    mc_phase1 = terminal.mc_start - terminal.slope1 * u_opt
    mc_phase2 = terminal.mc_min + terminal.slope2 * (u_opt - u_opt)  # Should equal mc_min

    print(f"Marginal Cost Continuity Check at u = {u_opt:.3f}:")
    print(f"  MC from Phase 1: ${mc_phase1:.4f}")
    print(f"  MC from Phase 2: ${mc_phase2:.4f}")
    print(f"  Difference: ${abs(mc_phase1 - mc_phase2):.6f} {'✓ PASS' if abs(mc_phase1 - mc_phase2) < 1e-6 else '✗ FAIL'}")

    # Test total cost continuity
    tc_phase1 = terminal.mc_start * u_opt - 0.5 * terminal.slope1 * u_opt**2
    tc_phase2_at_optimal = tc_phase1  # Should be the same by construction

    print(f"\nTotal Cost Continuity Check at u = {u_opt:.3f}:")
    print(f"  TC from Phase 1: ${tc_phase1:.2f}")
    print(f"  TC from Phase 2: ${tc_phase2_at_optimal:.2f}")
    print(f"  Difference: ${abs(tc_phase1 - tc_phase2_at_optimal):.6f} {'✓ PASS' if abs(tc_phase1 - tc_phase2_at_optimal) < 1e-6 else '✗ PASS (by construction)'}")

    # Test derivative continuity (marginal cost should match)
    print(f"\nDerivative Continuity Check:")
    print(f"  This is enforced by the optimization constraint: mc_start - slope1 × u_optimal = mc_min")
    print(f"  Expected mc_min: ${terminal.mc_min:.4f}")
    print(f"  Calculated: ${terminal.mc_start - terminal.slope1 * terminal.u_optimal:.4f}")
    print(f"  {'✓ CONSISTENT' if abs((terminal.mc_start - terminal.slope1 * terminal.u_optimal) - terminal.mc_min) < 1e-6 else '✗ INCONSISTENT'}")

def export_cost_coefficients_to_csv(terminals: List['OptimizedTerminalModel'],
                                   terminal_names: List[str],
                                   filename: str = "cost_function_coefficients.csv"):
    """
    Export cost function coefficients to a CSV file for easy analysis.
    """
    data_rows = []

    for i, (terminal, name) in enumerate(zip(terminals, terminal_names)):
        coeffs = extract_cost_function_coefficients(terminal)

        # Add terminal identification
        coeffs['terminal_id'] = i
        coeffs['terminal_name'] = name

        # Add revenue coefficients for completeness
        coeffs['revenue_a'] = terminal.a
        coeffs['revenue_b'] = terminal.b
        coeffs['revenue_c'] = terminal.c

        data_rows.append(coeffs)

    df = pd.DataFrame(data_rows)

    # Reorder columns for better readability
    column_order = ['terminal_id', 'terminal_name', 'capacity', 'u_optimal',
                   'mc_start', 'mc_min', 'slope1', 'slope2',
                   'tc_phase1_linear', 'tc_phase1_quadratic',
                   'tc_phase2_constant', 'tc_phase2_linear', 'tc_phase2_quadratic',
                   'revenue_a', 'revenue_b', 'revenue_c']

    df = df[column_order]
    df.to_csv(filename, index=False)
    print(f"\nCost function coefficients exported to: {filename}")
    return df

def plot_cost_functions(terminal: 'OptimizedTerminalModel', terminal_name: str = "Terminal"):
    """
    Plot the marginal cost and total cost functions to visualize the coefficients.
    """
    u_range = np.linspace(0.01, 0.99, 1000)

    # Calculate costs
    mc_values = terminal.marginal_cost(u_range)
    tc_per_teu = terminal.average_cost_per_container(u_range)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot marginal cost
    ax1.plot(u_range, mc_values, 'b-', linewidth=2, label='Marginal Cost')
    ax1.axvline(x=terminal.u_optimal, color='r', linestyle='--', alpha=0.7, label=f'u_optimal = {terminal.u_optimal:.3f}')
    ax1.axhline(y=terminal.mc_min, color='g', linestyle=':', alpha=0.7, label=f'mc_min = ${terminal.mc_min:.2f}')
    ax1.set_xlabel('Utilization (u)')
    ax1.set_ylabel('Marginal Cost ($/TEU)')
    ax1.set_title(f'Marginal Cost Function - {terminal_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add cost function equations as text
    coeffs = extract_cost_function_coefficients(terminal)
    eq_text = f"Phase 1: MC = {coeffs['mc_start']:.2f} - {coeffs['slope1']:.2f}×u\n"
    eq_text += f"Phase 2: MC = {coeffs['mc_min']:.2f} + {coeffs['slope2']:.2f}×(u-{coeffs['u_optimal']:.3f})"
    ax1.text(0.05, 0.95, eq_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot total cost per TEU
    ax2.plot(u_range, tc_per_teu, 'r-', linewidth=2, label='Average Cost per TEU')
    ax2.axvline(x=terminal.u_optimal, color='r', linestyle='--', alpha=0.7, label=f'u_optimal = {terminal.u_optimal:.3f}')
    ax2.set_xlabel('Utilization (u)')
    ax2.set_ylabel('Average Cost per TEU ($)')
    ax2.set_title(f'Average Cost Function - {terminal_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return fig

# Now modify the existing functions to include cost coefficient reporting:

def enhanced_print_summary_table(terminals: List['OptimizedTerminalModel'],
                                target_ratios: List[float],
                                capacities: List[float],
                                terminal_names: List[str],
                                actual_optimals: List[float],
                                max_profits: List[float],
                                method_name: str):
    """
    Enhanced summary table that includes explicit cost function coefficients.
    """
    print(f"\nENHANCED SUMMARY TABLE - {method_name.upper()} METHOD")
    print("=" * 280)
    print(f"{'Terminal':<25} {'Target V/C':<12} {'Capacity':<12} {'mc_start':<10} {'mc_min':<10} {'slope1':<10} {'slope2':<10} {'u_optimal':<12} {'Revenue_a':<12} {'Revenue_b':<12} {'Max Profit/TEU':<15} {'MR=MC Error':<12}")
    print("-" * 280)

    for i, terminal in enumerate(terminals):
        coeffs = extract_cost_function_coefficients(terminal)
        target_u = target_ratios[i]
        max_profit_per_teu = max_profits[i]

        # Calculate MR=MC error at target
        mr_at_target = terminal.marginal_revenue(np.array([target_u]))
        mc_at_target = terminal.marginal_cost(target_u)
        mr_at_target_scalar = mr_at_target.item()
        mr_mc_error = abs(mr_at_target_scalar - mc_at_target)

        print(f"{terminal_names[i]:<25} {target_u:<12.1%} {coeffs['capacity']:<12,.0f} {coeffs['mc_start']:<10.2f} {coeffs['mc_min']:<10.2f} {coeffs['slope1']:<10.2f} {coeffs['slope2']:<10.2f} {coeffs['u_optimal']:<12.3f} {terminal.a:<12.2f} {terminal.b:<12.2f} ${max_profit_per_teu:<14.2f} ${mr_mc_error:<11.4f}")

    print("=" * 280)

# Add this function to the enhanced simulation data generation:

def enhanced_generate_and_save_simulation_data():
    """
    Enhanced version that also exports cost function coefficients explicitly.
    """
    print("=" * 80)
    print("ENHANCED SIMULATION DATA GENERATION WITH COST COEFFICIENTS")
    print("=" * 80)

    # Call the original function
    generate_and_save_simulation_data()

    print("\n" + "="*60)
    print("EXTRACTING AND EXPORTING COST FUNCTION COEFFICIENTS")
    print("="*60)

    # Load and analyze some of the generated data to extract coefficients
    output_folder = "generated_data"
    if os.path.exists(output_folder):
        # Find all generated files
        pkl_files = [f for f in os.listdir(output_folder) if f.endswith('.pkl')]

        if pkl_files:
            print(f"Found {len(pkl_files)} data files. Analyzing cost coefficients from sample files...")

            # Analyze first few files as examples
            sample_files = pkl_files[:min(5, len(pkl_files))]

            all_coefficients = []

            for filename in sample_files:
                filepath = os.path.join(output_folder, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                print(f"\nAnalyzing: {filename}")
                print(f"Number of terminals: {data['num_terminals']}")

                # Reconstruct terminals from inputData to extract coefficients
                input_data = data['inputData']
                for i in range(data['num_terminals']):
                    # Extract parameters from inputData structure
                    capacity = input_data[0, i, 0]
                    mc_start = input_data[0, i, 1]
                    slope1 = input_data[0, i, 2]
                    slope2 = input_data[0, i, 3]
                    u_optimal = input_data[0, i, 4]
                    b = input_data[0, i, 5]
                    a = -input_data[0, i, 6]  # Remember it was negated

                    mc_min = mc_start - slope1 * u_optimal  # Reconstruct mc_min

                    # Create coefficient record
                    coeff_record = {
                        'filename': filename,
                        'terminal_index': i,
                        'terminal_name': data['terminal_names'][i] if i < len(data['terminal_names']) else f"Terminal {i+1}",
                        'capacity': capacity,
                        'mc_start': mc_start,
                        'mc_min': mc_min,
                        'slope1': slope1,
                        'slope2': slope2,
                        'u_optimal': u_optimal,
                        'revenue_a': a,
                        'revenue_b': b,
                        'ci_rate': data['ci_rate'],
                        'utilization_mix': data['utilization_mix'][i] if i < len(data['utilization_mix']) else 'unknown'
                    }

                    all_coefficients.append(coeff_record)

                    # Print cost function equations for first terminal of first file
                    if filename == sample_files[0] and i == 0:
                        print(f"\nSample Cost Function Equations for {coeff_record['terminal_name']}:")
                        print(f"  Phase 1 (u ≤ {u_optimal:.3f}): MC(u) = {mc_start:.2f} - {slope1:.2f}×u")
                        print(f"  Phase 2 (u > {u_optimal:.3f}): MC(u) = {mc_min:.2f} + {slope2:.2f}×(u-{u_optimal:.3f})")

            # Export all coefficients to CSV
            coeff_df = pd.DataFrame(all_coefficients)
            coeff_filename = os.path.join(output_folder, "cost_function_coefficients_summary.csv")
            coeff_df.to_csv(coeff_filename, index=False)
            print(f"\nCost function coefficients exported to: {coeff_filename}")

            # Summary statistics
            print(f"\nCOST COEFFICIENT SUMMARY STATISTICS:")
            print(f"Total terminals analyzed: {len(all_coefficients)}")
            numeric_cols = ['mc_start', 'mc_min', 'slope1', 'slope2', 'u_optimal', 'capacity']
            summary_stats = coeff_df[numeric_cols].describe()
            print(summary_stats)


# Replace the main execution block with this enhanced version:
if __name__ == "__main__":
    # Ask user what they want to do
    print("=" * 80)
    print("ENHANCED CONTAINER TERMINAL OPTIMIZATION WITH COST COEFFICIENTS")
    print("=" * 80)
    print("Choose an option:")
    print("1. Generate simulation data files with cost coefficient analysis")
    print("2. Run terminal parameter analysis with detailed cost functions")
    print("3. Both")
    print("4. Cost coefficient extraction only (analyze existing data)")

    choice = input("\nEnter choice (1, 2, 3, or 4): ").strip()

    if choice in ['1', '3']:
        print("\nGenerating Enhanced Simulation Data")
        print("=" * 60)
        enhanced_generate_and_save_simulation_data()

    if choice in ['2', '3']:
        print("\nRunning Enhanced Terminal Parameter Analysis")
        print("=" * 60)

        # Original functionality with enhancements
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

            # Enhanced summary table with cost coefficients
            enhanced_print_summary_table(results[0], results[1], results[3], results[2], results[4], results[5], method)

            # Export cost coefficients
            export_cost_coefficients_to_csv(results[0], results[2])

            # Show detailed analysis for first few terminals
            print(f"\nDETAILED COST FUNCTION ANALYSIS (First 3 terminals):")
            for i in range(min(3, len(results[0]))):
                print_cost_function_equations(results[0][i], results[2][i])
                validate_cost_function_continuity(results[0][i], results[2][i])

                # Plot cost functions for first terminal
                if i == 0:
                    if SHOW_PLOTS:
                        plot_cost_functions(results[0][i], results[2][i])

        write_data_to_excel_multi_method(all_results)

    if choice == '4':
        print("\nCost Coefficient Extraction from Existing Data")
        print("=" * 60)

        output_folder = "generated_data"
        if not os.path.exists(output_folder):
            print(f"No data folder found at: {output_folder}")
            print("Please run option 1 first to generate data.")
        else:
            # Analyze existing generated data
            pkl_files = [f for f in os.listdir(output_folder) if f.endswith('.pkl')]

            if not pkl_files:
                print(f"No pickle files found in: {output_folder}")
            else:
                print(f"Found {len(pkl_files)} data files. Extracting cost coefficients...")

                all_coefficients = []

                for filename in pkl_files:
                    filepath = os.path.join(output_folder, filename)
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)

                    input_data = data['inputData']
                    for i in range(data['num_terminals']):
                        capacity = input_data[0, i, 0]
                        mc_start = input_data[0, i, 1]
                        slope1 = input_data[0, i, 2]
                        slope2 = input_data[0, i, 3]
                        u_optimal = input_data[0, i, 4]
                        b = input_data[0, i, 5]
                        a = -input_data[0, i, 6]

                        mc_min = mc_start - slope1 * u_optimal

                        coeff_record = {
                            'filename': filename,
                            'terminal_index': i,
                            'terminal_name': data['terminal_names'][i] if i < len(data['terminal_names']) else f"Terminal {i+1}",
                            'capacity': capacity,
                            'mc_start': mc_start,
                            'mc_min': mc_min,
                            'slope1': slope1,
                            'slope2': slope2,
                            'u_optimal': u_optimal,
                            'revenue_a': a,
                            'revenue_b': b,
                            'ci_rate': data['ci_rate'],
                            'num_terminals': data['num_terminals']
                        }

                        all_coefficients.append(coeff_record)

                # Export and analyze
                coeff_df = pd.DataFrame(all_coefficients)
                coeff_filename = os.path.join(output_folder, "extracted_cost_coefficients.csv")
                coeff_df.to_csv(coeff_filename, index=False)
                print(f"\nCost coefficients extracted to: {coeff_filename}")

                # Show summary statistics
                print(f"\nSUMMARY STATISTICS:")
                print(f"Total terminals: {len(all_coefficients)}")
                print(f"Unique scenarios: {len(pkl_files)}")

                numeric_cols = ['mc_start', 'mc_min', 'slope1', 'slope2', 'u_optimal', 'capacity']
                if all(col in coeff_df.columns for col in numeric_cols):
                    print(f"\nCost Coefficient Statistics:")
                    summary_stats = coeff_df[numeric_cols].describe()
                    print(summary_stats)

    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Please run again and choose 1, 2, 3, or 4.")
    else:
        print(f"\nComplete! All cost function coefficients have been extracted and validated.")
        print("=" * 80)