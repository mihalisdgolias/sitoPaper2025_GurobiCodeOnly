"""
LINEAR COST APPROXIMATION ANALYSIS SCRIPT
==========================================

This script reads the generated terminal data and creates linear approximations
of the piecewise quadratic cost functions. It provides:

1. Linear fit coefficients for each terminal
2. Goodness of fit analysis
3. Comparison plots of actual vs linear costs
4. Export of linear coefficients for optimization use
5. Statistical analysis of approximation quality

The actual cost function is piecewise quadratic:
- Phase 1 (u ≤ u_opt): TC = capacity * (cost_initial * u - 0.5 * cost_slope1 * u²)
- Phase 2 (u > u_opt): TC = TC_at_opt + capacity * (mc_min * (u - u_opt) + 0.5 * cost_slope2 * (u - u_opt)²)

Linear approximation: TC_linear = alpha + beta * utilization * capacity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import re
from typing import List, Tuple, Dict, Any
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns

# Configuration
DATA_FOLDER = 'generated_data'
OUTPUT_FOLDER = 'linear_cost_analysis'
UTILIZATION_RANGE = (0.1, 0.9)  # Range to fit over
NUM_SAMPLE_POINTS = 50  # Number of points for fitting
FILTER_CI_RATE = 0.0  # Only analyze CI rate = 0 data
PLOT_SAMPLE_SIZE = 20  # Number of terminals to plot individually


class TerminalCostModel:
    """Reconstructed terminal cost model for analysis."""

    def __init__(self, cost_initial: float, cost_slope1: float, cost_slope2: float,
                 optimal_utilization: float, capacity: float):
        self.cost_initial = cost_initial
        self.cost_slope1 = cost_slope1
        self.cost_slope2 = cost_slope2
        self.optimal_utilization = optimal_utilization
        self.capacity = capacity

        # Calculate mc_min from continuity constraint
        self.mc_min = cost_initial - cost_slope1 * optimal_utilization

    def total_cost(self, utilization: np.ndarray) -> np.ndarray:
        """Calculate actual piecewise quadratic total cost."""
        utilization = np.asarray(utilization)
        total_costs = np.zeros_like(utilization, dtype=float)

        # Phase 1: u <= u_optimal (decreasing marginal cost)
        mask1 = utilization <= self.optimal_utilization
        total_costs[mask1] = self.capacity * (
                self.cost_initial * utilization[mask1] -
                0.5 * self.cost_slope1 * utilization[mask1] ** 2
        )

        # Phase 2: u > u_optimal (increasing marginal cost)
        mask2 = utilization > self.optimal_utilization
        if np.any(mask2):
            # Cost at optimal point
            cost_at_optimal = self.capacity * (
                    self.cost_initial * self.optimal_utilization -
                    0.5 * self.cost_slope1 * self.optimal_utilization ** 2
            )

            # Additional cost beyond optimal
            excess_utilization = utilization[mask2] - self.optimal_utilization
            additional_cost = self.capacity * (
                    self.mc_min * excess_utilization +
                    0.5 * self.cost_slope2 * excess_utilization ** 2
            )

            total_costs[mask2] = cost_at_optimal + additional_cost

        return total_costs

    def marginal_cost(self, utilization: np.ndarray) -> np.ndarray:
        """Calculate marginal cost function."""
        utilization = np.asarray(utilization)
        mc = np.where(
            utilization <= self.optimal_utilization,
            self.cost_initial - self.cost_slope1 * utilization,
            self.mc_min + self.cost_slope2 * (utilization - self.optimal_utilization)
        )
        return mc


def calculate_linear_approximation(cost_model: TerminalCostModel,
                                   utilization_range: Tuple[float, float] = UTILIZATION_RANGE,
                                   num_points: int = NUM_SAMPLE_POINTS) -> Dict[str, Any]:
    """
    Calculate linear approximation of the cost function.

    Returns:
        Dictionary with linear coefficients and fit metrics
    """
    min_u, max_u = utilization_range

    # Create sample points
    u_points = np.linspace(min_u, max_u, num_points)
    actual_costs = cost_model.total_cost(u_points)

    # Fit linear approximation: TC = alpha + beta * utilization * capacity
    # Rearrange to: TC = alpha + (beta * capacity) * utilization
    # So we fit: TC = a + b * utilization, where b = beta * capacity

    A = np.vstack([np.ones(len(u_points)), u_points]).T
    coefficients = np.linalg.lstsq(A, actual_costs, rcond=None)[0]
    alpha = coefficients[0]
    beta_times_capacity = coefficients[1]
    beta = beta_times_capacity / cost_model.capacity

    # Calculate linear predictions
    linear_costs = alpha + beta_times_capacity * u_points

    # Calculate fit metrics
    r2 = r2_score(actual_costs, linear_costs)
    mae = mean_absolute_error(actual_costs, linear_costs)
    rmse = np.sqrt(mean_squared_error(actual_costs, linear_costs))
    max_abs_error = np.max(np.abs(actual_costs - linear_costs))
    mean_relative_error = np.mean(np.abs((actual_costs - linear_costs) / actual_costs)) * 100

    return {
        'alpha': alpha,
        'beta': beta,
        'beta_times_capacity': beta_times_capacity,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'max_abs_error': max_abs_error,
        'mean_relative_error_percent': mean_relative_error,
        'utilization_points': u_points,
        'actual_costs': actual_costs,
        'linear_costs': linear_costs
    }


def load_and_analyze_terminals() -> pd.DataFrame:
    """
    Load terminal data and calculate linear approximations.

    Returns:
        DataFrame with terminal parameters and linear approximation results
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found.")
        return pd.DataFrame()

    data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl')]
    if not data_files:
        print(f"Error: No .pkl files found in '{DATA_FOLDER}'.")
        return pd.DataFrame()

    filename_pattern = re.compile(r'data_T(\d+)_subset_([\d_]+)_CI(\d+)_instance_(\d+)\.pkl')

    results = []
    terminals_processed = 0

    print(f"Processing {len(data_files)} data files...")
    print(f"Filtering for CI rate = {FILTER_CI_RATE:.0%}")

    for filename in data_files:
        filepath = os.path.join(DATA_FOLDER, filename)

        # Parse filename
        match = filename_pattern.match(filename)
        if not match:
            print(f"Warning: Filename '{filename}' doesn't match pattern. Skipping.")
            continue

        num_terminals_str, subset_str, ci_rate_str, instance_idx_str = match.groups()
        num_terminals = int(num_terminals_str)
        ci_rate = int(ci_rate_str) / 100.0
        instance_idx = int(instance_idx_str)

        # Filter for CI rate = 0
        if ci_rate != FILTER_CI_RATE:
            continue

        # Load data
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        print(f"Processing {filename}: {num_terminals} terminals")

        # Extract terminal parameters
        input_data = data['inputData'][0]

        for i in range(num_terminals):
            try:
                # Extract parameters
                capacity = input_data[i, 0]
                cost_initial = input_data[i, 1]
                cost_slope1 = input_data[i, 2]
                cost_slope2 = input_data[i, 3]
                optimal_utilization = input_data[i, 4]

                # Revenue parameters (for completeness)
                revenue_b = input_data[i, 5]
                revenue_a = -input_data[i, 6]  # Remember it was negated

                # Create cost model
                cost_model = TerminalCostModel(
                    cost_initial, cost_slope1, cost_slope2,
                    optimal_utilization, capacity
                )

                # Calculate linear approximation
                linear_fit = calculate_linear_approximation(cost_model)

                # Store results
                terminal_result = {
                    'filename': filename,
                    'num_terminals': num_terminals,
                    'subset_composition': subset_str,
                    'ci_rate': ci_rate,
                    'instance_index': instance_idx,
                    'terminal_id': i + 1,
                    'capacity': capacity,
                    'cost_initial': cost_initial,
                    'cost_slope1': cost_slope1,
                    'cost_slope2': cost_slope2,
                    'optimal_utilization': optimal_utilization,
                    'mc_min': cost_model.mc_min,
                    'revenue_a': revenue_a,
                    'revenue_b': revenue_b,

                    # Linear approximation results
                    'linear_alpha': linear_fit['alpha'],
                    'linear_beta': linear_fit['beta'],
                    'linear_beta_times_capacity': linear_fit['beta_times_capacity'],

                    # Fit quality metrics
                    'r2_score': linear_fit['r2_score'],
                    'mae': linear_fit['mae'],
                    'rmse': linear_fit['rmse'],
                    'max_abs_error': linear_fit['max_abs_error'],
                    'mean_relative_error_percent': linear_fit['mean_relative_error_percent'],

                    # Store fit data for plotting
                    'fit_data': linear_fit
                }

                results.append(terminal_result)
                terminals_processed += 1

            except Exception as e:
                print(f"Error processing terminal {i + 1} in {filename}: {e}")
                continue

    print(f"Successfully processed {terminals_processed} terminals")
    return pd.DataFrame(results)


def create_summary_plots(df: pd.DataFrame):
    """Create summary plots of linear approximation quality."""
    if df.empty:
        print("No data to plot")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Distribution of R² scores
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Linear Cost Approximation Quality Analysis', fontsize=16, fontweight='bold')

    # R² distribution
    axes[0, 0].hist(df['r2_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['r2_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["r2_score"].mean():.4f}')
    axes[0, 0].set_xlabel('R² Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of R² Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Relative error distribution
    axes[0, 1].hist(df['mean_relative_error_percent'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df['mean_relative_error_percent'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["mean_relative_error_percent"].mean():.2f}%')
    axes[0, 1].set_xlabel('Mean Relative Error (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Relative Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # R² vs Terminal Capacity
    scatter = axes[1, 0].scatter(df['capacity'], df['r2_score'],
                                 c=df['optimal_utilization'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Terminal Capacity (TEU/week)')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R² Score vs Terminal Capacity')
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Optimal Utilization')
    axes[1, 0].grid(True, alpha=0.3)

    # Linear coefficients relationship
    axes[1, 1].scatter(df['linear_alpha'], df['linear_beta'], alpha=0.6)
    axes[1, 1].set_xlabel('Alpha (Fixed Cost Component)')
    axes[1, 1].set_ylabel('Beta (Variable Cost Coefficient)')
    axes[1, 1].set_title('Linear Coefficients Relationship')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'linear_approximation_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_individual_terminal_plots(df: pd.DataFrame, sample_size: int = PLOT_SAMPLE_SIZE):
    """Create individual plots showing actual vs linear costs for sample terminals."""
    if df.empty:
        return

    # Sample terminals for plotting
    sample_df = df.sample(min(sample_size, len(df)), random_state=42).copy()

    print(f"Creating individual plots for {len(sample_df)} sample terminals...")

    plots_folder = os.path.join(OUTPUT_FOLDER, 'individual_terminal_plots')
    os.makedirs(plots_folder, exist_ok=True)

    for idx, row in sample_df.iterrows():
        try:
            fit_data = row['fit_data']
            u_points = fit_data['utilization_points']
            actual_costs = fit_data['actual_costs']
            linear_costs = fit_data['linear_costs']

            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot 1: Cost comparison
            ax1.plot(u_points, actual_costs, 'b-', linewidth=2, label='Actual Cost (Piecewise)')
            ax1.plot(u_points, linear_costs, 'r--', linewidth=2, label='Linear Approximation')
            ax1.axvline(x=row['optimal_utilization'], color='green', linestyle=':',
                        alpha=0.7, label=f'Optimal U = {row["optimal_utilization"]:.3f}')
            ax1.set_xlabel('Utilization')
            ax1.set_ylabel('Total Cost ($)')
            ax1.set_title(f'Cost Functions Comparison\nTerminal {row["terminal_id"]} from {row["filename"]}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Error analysis
            errors = actual_costs - linear_costs
            relative_errors = (errors / actual_costs) * 100

            ax2.plot(u_points, errors, 'g-', linewidth=2, label='Absolute Error')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(u_points, relative_errors, 'orange', linestyle='--',
                          linewidth=2, label='Relative Error (%)')

            ax2.set_xlabel('Utilization')
            ax2.set_ylabel('Absolute Error ($)', color='g')
            ax2_twin.set_ylabel('Relative Error (%)', color='orange')
            ax2.set_title('Approximation Errors')
            ax2.grid(True, alpha=0.3)

            # Add metrics text
            metrics_text = f"""
Linear Approximation: TC = {row['linear_alpha']:.0f} + {row['linear_beta']:.2f} × U × Capacity
R² = {row['r2_score']:.4f}
Mean Rel. Error = {row['mean_relative_error_percent']:.2f}%
Max Abs. Error = ${row['max_abs_error']:.0f}
            """
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                     verticalalignment='top', fontfamily='monospace', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

            plt.tight_layout()

            # Save plot
            plot_filename = f"terminal_{row['filename'][:-4]}_T{row['terminal_id']}_linear_fit.png"
            plt.savefig(os.path.join(plots_folder, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating plot for terminal {row['terminal_id']}: {e}")
            continue


def export_linear_coefficients(df: pd.DataFrame):
    """Export linear approximation coefficients for use in optimization."""
    if df.empty:
        return

    # Create summary statistics
    summary_stats = {
        'Total Terminals Analyzed': len(df),
        'Mean R² Score': df['r2_score'].mean(),
        'Min R² Score': df['r2_score'].min(),
        'Max R² Score': df['r2_score'].max(),
        'Mean Relative Error (%)': df['mean_relative_error_percent'].mean(),
        'Max Relative Error (%)': df['mean_relative_error_percent'].max(),
        'Terminals with R² > 0.95': (df['r2_score'] > 0.95).sum(),
        'Terminals with R² > 0.90': (df['r2_score'] > 0.90).sum(),
        'Terminals with Rel. Error < 5%': (df['mean_relative_error_percent'] < 5).sum(),
    }

    # Export detailed coefficients
    coefficients_df = df[[
        'filename', 'terminal_id', 'capacity', 'optimal_utilization',
        'cost_initial', 'cost_slope1', 'cost_slope2', 'mc_min',
        'linear_alpha', 'linear_beta', 'linear_beta_times_capacity',
        'r2_score', 'mae', 'rmse', 'max_abs_error', 'mean_relative_error_percent'
    ]].copy()

    # Save to Excel
    excel_path = os.path.join(OUTPUT_FOLDER, 'linear_cost_coefficients.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        coefficients_df.to_excel(writer, sheet_name='Linear_Coefficients', index=False)

        # Create summary sheet
        summary_df = pd.DataFrame(list(summary_stats.items()),
                                  columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

    # Save to CSV
    coefficients_df.to_csv(os.path.join(OUTPUT_FOLDER, 'linear_cost_coefficients.csv'), index=False)

    print(f"\nLinear coefficients exported to:")
    print(f"  Excel: {excel_path}")
    print(f"  CSV: {os.path.join(OUTPUT_FOLDER, 'linear_cost_coefficients.csv')}")

    return summary_stats


def print_analysis_summary(df: pd.DataFrame, summary_stats: Dict):
    """Print comprehensive analysis summary."""
    print("\n" + "=" * 80)
    print("LINEAR COST APPROXIMATION ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nData Overview:")
    print(f"  Total terminals analyzed: {len(df)}")
    print(f"  CI rate filter: {FILTER_CI_RATE:.0%}")
    print(f"  Utilization range: {UTILIZATION_RANGE[0]:.1%} - {UTILIZATION_RANGE[1]:.1%}")
    print(f"  Sample points per fit: {NUM_SAMPLE_POINTS}")

    print(f"\nFit Quality Statistics:")
    print(f"  Mean R² Score: {summary_stats['Mean R² Score']:.4f}")
    print(f"  R² Score Range: {summary_stats['Min R² Score']:.4f} - {summary_stats['Max R² Score']:.4f}")
    print(f"  Mean Relative Error: {summary_stats['Mean Relative Error (%)']:.2f}%")
    print(f"  Max Relative Error: {summary_stats['Max Relative Error (%)']:.2f}%")

    print(f"\nFit Quality Distribution:")
    print(
        f"  Terminals with R² > 0.95: {summary_stats['Terminals with R² > 0.95']} ({summary_stats['Terminals with R² > 0.95'] / len(df) * 100:.1f}%)")
    print(
        f"  Terminals with R² > 0.90: {summary_stats['Terminals with R² > 0.90']} ({summary_stats['Terminals with R² > 0.90'] / len(df) * 100:.1f}%)")
    print(
        f"  Terminals with Rel. Error < 5%: {summary_stats['Terminals with Rel. Error < 5%']} ({summary_stats['Terminals with Rel. Error < 5%'] / len(df) * 100:.1f}%)")

    print(f"\nLinear Coefficient Ranges:")
    print(f"  Alpha (fixed cost): ${df['linear_alpha'].min():.0f} - ${df['linear_alpha'].max():.0f}")
    print(f"  Beta (variable coeff): {df['linear_beta'].min():.2f} - {df['linear_beta'].max():.2f}")

    print(f"\nTerminal Parameter Ranges:")
    print(f"  Capacity: {df['capacity'].min():,.0f} - {df['capacity'].max():,.0f} TEU/week")
    print(f"  Optimal Utilization: {df['optimal_utilization'].min():.3f} - {df['optimal_utilization'].max():.3f}")
    print(f"  Initial MC: ${df['cost_initial'].min():.2f} - ${df['cost_initial'].max():.2f}")

    # Identify problematic terminals
    poor_fit_terminals = df[df['r2_score'] < 0.9]
    if not poor_fit_terminals.empty:
        print(f"\nTerminals with Poor Linear Fit (R² < 0.9):")
        for _, terminal in poor_fit_terminals.iterrows():
            print(f"  {terminal['filename']} Terminal {terminal['terminal_id']}: R² = {terminal['r2_score']:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("LINEAR COST APPROXIMATION ANALYSIS")
    print("=" * 50)

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load and analyze terminals
    print("Loading terminal data and calculating linear approximations...")
    df = load_and_analyze_terminals()

    if df.empty:
        print("No terminal data found. Check your data generation and filtering settings.")
        exit()

    # Create plots
    print("Creating summary plots...")
    create_summary_plots(df)

    print("Creating individual terminal plots...")
    create_individual_terminal_plots(df)

    # Export coefficients
    print("Exporting linear coefficients...")
    summary_stats = export_linear_coefficients(df)

    # Print summary
    print_analysis_summary(df, summary_stats)

    print(f"\nAnalysis complete! Results saved to: {OUTPUT_FOLDER}")