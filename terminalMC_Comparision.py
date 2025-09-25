"""
MARGINAL COST COMPARISON ANALYSIS
==================================

This script analyzes marginal cost differences between terminals across all V/C ratios
to identify cooperation opportunities and understand why terminals may or may not
be cooperating in the optimization model.

Key Analysis:
1. Marginal cost curves for each terminal
2. Pairwise marginal cost differences at each V/C ratio
3. Cooperation opportunity identification
4. Current utilization vs optimal utilization analysis
5. Cost savings potential from vessel transfers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from typing import List, Tuple, Dict, Any
from itertools import combinations

# Configuration
DATA_FOLDER = 'generated_data'
OUTPUT_FOLDER = 'marginal_cost_analysis'
FILTER_CI_RATE = 0.0  # Only analyze CI rate = 0 data
VC_RANGE = (0.1, 0.95)  # V/C ratio range to analyze
NUM_POINTS = 100  # Number of points for analysis


class TerminalCostModel:
    """Reconstructed terminal cost model for marginal cost analysis."""

    def __init__(self, cost_initial: float, cost_slope1: float, cost_slope2: float,
                 optimal_utilization: float, capacity: float, terminal_name: str):
        self.cost_initial = cost_initial
        self.cost_slope1 = cost_slope1
        self.cost_slope2 = cost_slope2
        self.optimal_utilization = optimal_utilization
        self.capacity = capacity
        self.terminal_name = terminal_name

        # Calculate mc_min from continuity constraint
        self.mc_min = cost_initial - cost_slope1 * optimal_utilization

    def marginal_cost(self, utilization: np.ndarray) -> np.ndarray:
        """Calculate marginal cost at given utilization levels."""
        utilization = np.asarray(utilization)
        mc = np.where(
            utilization <= self.optimal_utilization,
            self.cost_initial - self.cost_slope1 * utilization,
            self.mc_min + self.cost_slope2 * (utilization - self.optimal_utilization)
        )
        return mc

    def total_cost(self, utilization: np.ndarray) -> np.ndarray:
        """Calculate total cost at given utilization levels."""
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
            cost_at_optimal = self.capacity * (
                    self.cost_initial * self.optimal_utilization -
                    0.5 * self.cost_slope1 * self.optimal_utilization ** 2
            )
            excess_utilization = utilization[mask2] - self.optimal_utilization
            additional_cost = self.capacity * (
                    self.mc_min * excess_utilization +
                    0.5 * self.cost_slope2 * excess_utilization ** 2
            )
            total_costs[mask2] = cost_at_optimal + additional_cost

        return total_costs


def load_terminal_data() -> List[Dict[str, Any]]:
    """Load terminal data from generated files."""
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder '{DATA_FOLDER}' not found.")
        return []

    data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl')]
    if not data_files:
        print(f"Error: No .pkl files found in '{DATA_FOLDER}'.")
        return []

    filename_pattern = re.compile(r'data_T(\d+)_subset_([\d_]+)_CI(\d+)_instance_(\d+)\.pkl')
    terminal_data = []

    print(f"Loading terminal data from {len(data_files)} files...")
    print(f"Filtering for CI rate = {FILTER_CI_RATE:.0%}")

    for filename in data_files:
        filepath = os.path.join(DATA_FOLDER, filename)

        # Parse filename
        match = filename_pattern.match(filename)
        if not match:
            continue

        num_terminals = int(match.groups()[0])
        ci_rate = int(match.groups()[2]) / 100.0
        instance_idx = int(match.groups()[3])

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

        # Extract terminal parameters and current utilization
        input_data = data['inputData'][0]
        vessel_volumes = data['vVes']
        vessel_assignments = data['xV']

        # Calculate current utilization for each terminal
        volume_before = np.sum(vessel_volumes[:, None] * vessel_assignments, axis=0)

        for i in range(num_terminals):
            capacity = input_data[i, 0]
            cost_initial = input_data[i, 1]
            cost_slope1 = input_data[i, 2]
            cost_slope2 = input_data[i, 3]
            optimal_utilization = input_data[i, 4]

            current_utilization = volume_before[i] / capacity
            terminal_name = data['terminal_names'][i] if i < len(data['terminal_names']) else f"Terminal {i + 1}"

            terminal_record = {
                'filename': filename,
                'scenario_id': f"{filename[:-4]}_T{i + 1}",
                'terminal_id': i + 1,
                'terminal_name': terminal_name,
                'num_terminals_in_scenario': num_terminals,
                'instance_index': instance_idx,
                'capacity': capacity,
                'cost_initial': cost_initial,
                'cost_slope1': cost_slope1,
                'cost_slope2': cost_slope2,
                'optimal_utilization': optimal_utilization,
                'current_utilization': current_utilization,
                'current_volume': volume_before[i],
                'cost_model': TerminalCostModel(
                    cost_initial, cost_slope1, cost_slope2,
                    optimal_utilization, capacity, terminal_name
                )
            }

            terminal_data.append(terminal_record)

    print(f"Loaded {len(terminal_data)} terminals from {len(set([t['filename'] for t in terminal_data]))} scenarios")
    return terminal_data


def analyze_marginal_cost_differences(terminal_data: List[Dict]) -> pd.DataFrame:
    """Analyze marginal cost differences between all terminal pairs."""
    vc_ratios = np.linspace(VC_RANGE[0], VC_RANGE[1], NUM_POINTS)

    analysis_results = []

    # Group terminals by scenario to analyze within-scenario cooperation opportunities
    scenarios = {}
    for terminal in terminal_data:
        scenario_key = f"{terminal['filename']}"
        if scenario_key not in scenarios:
            scenarios[scenario_key] = []
        scenarios[scenario_key].append(terminal)

    print(f"Analyzing marginal cost differences across {len(scenarios)} scenarios...")

    for scenario_key, terminals in scenarios.items():
        print(f"  Scenario: {scenario_key} ({len(terminals)} terminals)")

        # Calculate marginal costs for all terminals at all V/C ratios
        terminal_mc_data = {}
        for terminal in terminals:
            mc_values = terminal['cost_model'].marginal_cost(vc_ratios)
            terminal_mc_data[terminal['terminal_id']] = {
                'mc_values': mc_values,
                'terminal_data': terminal
            }

        # Compare all pairs of terminals within this scenario
        for term1_id, term2_id in combinations(terminal_mc_data.keys(), 2):
            term1 = terminal_mc_data[term1_id]
            term2 = terminal_mc_data[term2_id]

            mc_diff = term1['mc_values'] - term2['mc_values']  # Positive = term1 higher cost

            # Find V/C ranges where cooperation would be beneficial
            term1_higher_ranges = []
            term2_higher_ranges = []

            for i, vc in enumerate(vc_ratios):
                if mc_diff[i] > 5:  # $5 threshold for meaningful difference
                    term1_higher_ranges.append((vc, mc_diff[i]))
                elif mc_diff[i] < -5:
                    term2_higher_ranges.append((vc, -mc_diff[i]))

            # Check current utilization cooperation opportunity
            current_util_1 = term1['terminal_data']['current_utilization']
            current_util_2 = term2['terminal_data']['current_utilization']

            # Calculate marginal costs at current utilizations
            current_mc_1 = term1['terminal_data']['cost_model'].marginal_cost([current_util_1])[0]
            current_mc_2 = term2['terminal_data']['cost_model'].marginal_cost([current_util_2])[0]

            current_mc_diff = current_mc_1 - current_mc_2

            # Cooperation recommendation
            cooperation_potential = ""
            if abs(current_mc_diff) > 10:  # $10 threshold
                if current_mc_diff > 0:
                    cooperation_potential = f"Terminal {term1_id} should send vessels to Terminal {term2_id} (saves ${current_mc_diff:.2f}/TEU)"
                else:
                    cooperation_potential = f"Terminal {term2_id} should send vessels to Terminal {term1_id} (saves ${abs(current_mc_diff):.2f}/TEU)"
            else:
                cooperation_potential = "Limited cooperation benefit"

            analysis_results.append({
                'scenario': scenario_key,
                'terminal_1_id': term1_id,
                'terminal_2_id': term2_id,
                'terminal_1_name': term1['terminal_data']['terminal_name'],
                'terminal_2_name': term2['terminal_data']['terminal_name'],
                'terminal_1_current_util': current_util_1,
                'terminal_2_current_util': current_util_2,
                'terminal_1_optimal_util': term1['terminal_data']['optimal_utilization'],
                'terminal_2_optimal_util': term2['terminal_data']['optimal_utilization'],
                'terminal_1_current_mc': current_mc_1,
                'terminal_2_current_mc': current_mc_2,
                'current_mc_difference': current_mc_diff,
                'max_mc_difference': np.max(np.abs(mc_diff)),
                'avg_mc_difference': np.mean(np.abs(mc_diff)),
                'cooperation_opportunities_count': len(term1_higher_ranges) + len(term2_higher_ranges),
                'cooperation_potential': cooperation_potential,
                'terminal_1_above_optimal': current_util_1 > term1['terminal_data']['optimal_utilization'],
                'terminal_2_above_optimal': current_util_2 > term2['terminal_data']['optimal_utilization']
            })

    return pd.DataFrame(analysis_results)


def create_marginal_cost_plots(terminal_data: List[Dict]):
    """Create comprehensive marginal cost comparison plots."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Group by scenario for plotting
    scenarios = {}
    for terminal in terminal_data:
        scenario_key = f"{terminal['filename']}"
        if scenario_key not in scenarios:
            scenarios[scenario_key] = []
        scenarios[scenario_key].append(terminal)

    vc_ratios = np.linspace(VC_RANGE[0], VC_RANGE[1], NUM_POINTS)

    # Create plots for each scenario (limit to first 10 scenarios to avoid too many plots)
    scenarios_to_plot = list(scenarios.items())[:10]

    for scenario_idx, (scenario_key, terminals) in enumerate(scenarios_to_plot):
        print(f"Creating marginal cost plot for scenario {scenario_idx + 1}: {scenario_key}")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'Marginal Cost Analysis: {scenario_key}', fontsize=14, fontweight='bold')

        colors = plt.cm.Set1(np.linspace(0, 1, len(terminals)))

        # Plot 1: Marginal cost curves
        for i, terminal in enumerate(terminals):
            mc_values = terminal['cost_model'].marginal_cost(vc_ratios)
            current_util = terminal['current_utilization']
            optimal_util = terminal['optimal_utilization']

            ax1.plot(vc_ratios, mc_values, color=colors[i], linewidth=2,
                     label=f"T{terminal['terminal_id']}: {terminal['terminal_name']}")

            # Mark current utilization
            current_mc = terminal['cost_model'].marginal_cost([current_util])[0]
            ax1.scatter([current_util], [current_mc], color=colors[i], s=100,
                        marker='o', edgecolor='black', linewidth=2, zorder=5)

            # Mark optimal utilization
            optimal_mc = terminal['cost_model'].marginal_cost([optimal_util])[0]
            ax1.scatter([optimal_util], [optimal_mc], color=colors[i], s=100,
                        marker='s', edgecolor='black', linewidth=2, zorder=5)

        ax1.set_xlabel('V/C Ratio (Utilization)')
        ax1.set_ylabel('Marginal Cost ($/TEU)')
        ax1.set_title('Marginal Cost Curves (○ = Current, □ = Optimal)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Current vs Optimal utilization comparison
        terminal_names = [f"T{t['terminal_id']}" for t in terminals]
        current_utils = [t['current_utilization'] for t in terminals]
        optimal_utils = [t['optimal_utilization'] for t in terminals]
        current_mcs = [t['cost_model'].marginal_cost([t['current_utilization']])[0] for t in terminals]

        x_pos = np.arange(len(terminals))
        width = 0.35

        bars1 = ax2.bar(x_pos - width / 2, current_utils, width, label='Current Utilization', alpha=0.7, color='orange')
        bars2 = ax2.bar(x_pos + width / 2, optimal_utils, width, label='Optimal Utilization', alpha=0.7, color='blue')

        ax2.set_xlabel('Terminals')
        ax2.set_ylabel('Utilization Ratio')
        ax2.set_title('Current vs Optimal Utilization')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(terminal_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add values on bars
        for bar, util in zip(bars1, current_utils):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{util:.2f}', ha='center', va='bottom', fontsize=9)
        for bar, util in zip(bars2, optimal_utils):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{util:.2f}', ha='center', va='bottom', fontsize=9)

        # Plot 3: Current marginal costs
        bars3 = ax3.bar(terminal_names, current_mcs, color=colors[:len(terminals)], alpha=0.7)
        ax3.set_xlabel('Terminals')
        ax3.set_ylabel('Current Marginal Cost ($/TEU)')
        ax3.set_title('Current Marginal Costs')
        ax3.grid(True, alpha=0.3)

        # Add values on bars and highlight cooperation opportunities
        max_mc = max(current_mcs)
        min_mc = min(current_mcs)
        mc_diff = max_mc - min_mc

        for bar, mc, terminal in zip(bars3, current_mcs, terminals):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_mc * 0.01,
                     f'${mc:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Color code: red if above optimal, green if below optimal
            if terminal['current_utilization'] > terminal['optimal_utilization']:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
            else:
                bar.set_edgecolor('green')
                bar.set_linewidth(3)

        # Add cooperation potential text
        if mc_diff > 10:
            ax3.text(0.5, 0.95, f"Cooperation Potential: ${mc_diff:.0f}/TEU savings possible",
                     transform=ax3.transAxes, ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'marginal_cost_analysis_{scenario_idx + 1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_cooperation_opportunity_summary(analysis_df: pd.DataFrame):
    """Create summary of cooperation opportunities across all scenarios."""
    print("\n" + "=" * 80)
    print("COOPERATION OPPORTUNITY ANALYSIS")
    print("=" * 80)

    # Overall statistics
    total_pairs = len(analysis_df)
    high_potential_pairs = len(analysis_df[analysis_df['current_mc_difference'].abs() > 10])

    print(f"Total terminal pairs analyzed: {total_pairs}")
    print(f"Pairs with high cooperation potential (>$10/TEU difference): {high_potential_pairs}")
    print(f"High potential rate: {high_potential_pairs / total_pairs * 100:.1f}%")

    # Top cooperation opportunities
    print(f"\nTOP 10 COOPERATION OPPORTUNITIES:")
    print("-" * 50)
    top_opportunities = analysis_df.nlargest(10, 'current_mc_difference')

    for _, row in top_opportunities.iterrows():
        print(f"Scenario: {row['scenario']}")
        print(f"  {row['terminal_1_name']} (MC: ${row['terminal_1_current_mc']:.0f}, "
              f"Util: {row['terminal_1_current_util']:.1%}) → "
              f"{row['terminal_2_name']} (MC: ${row['terminal_2_current_mc']:.0f}, "
              f"Util: {row['terminal_2_current_util']:.1%})")
        print(f"  Savings potential: ${row['current_mc_difference']:.0f}/TEU")
        print(f"  Recommendation: {row['cooperation_potential']}")
        print()

    # Summary by scenario characteristics
    print(f"\nCOOPERATION POTENTIAL BY SCENARIO CHARACTERISTICS:")
    print("-" * 50)

    # Create scenario summary
    scenario_summary = analysis_df.groupby('scenario').agg({
        'current_mc_difference': ['max', 'mean'],
        'cooperation_opportunities_count': 'sum'
    }).round(2)
    scenario_summary.columns = ['Max_MC_Diff', 'Avg_MC_Diff', 'Total_Opportunities']

    # Sort by potential
    scenario_summary = scenario_summary.sort_values('Max_MC_Diff', ascending=False)
    print(scenario_summary.head(10))

    # Export detailed results
    analysis_df.to_csv(os.path.join(OUTPUT_FOLDER, 'cooperation_opportunities.csv'), index=False)
    scenario_summary.to_csv(os.path.join(OUTPUT_FOLDER, 'scenario_cooperation_summary.csv'))

    print(f"\nDetailed results exported to: {OUTPUT_FOLDER}")
    print(f"Files created:")
    print(f"  - cooperation_opportunities.csv")
    print(f"  - scenario_cooperation_summary.csv")
    print(f"  - marginal_cost_analysis_[N].png (for each scenario)")


def main():
    """Run the complete marginal cost comparison analysis."""
    print("MARGINAL COST COMPARISON ANALYSIS")
    print("=" * 50)

    # Load terminal data
    terminal_data = load_terminal_data()
    if not terminal_data:
        print("No terminal data found. Please check data generation.")
        return

    # Analyze marginal cost differences
    print("Analyzing marginal cost differences between terminal pairs...")
    analysis_df = analyze_marginal_cost_differences(terminal_data)

    # Create plots
    print("Creating marginal cost comparison plots...")
    create_marginal_cost_plots(terminal_data)

    # Create summary
    create_cooperation_opportunity_summary(analysis_df)

    print(f"\nAnalysis complete! Results saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()