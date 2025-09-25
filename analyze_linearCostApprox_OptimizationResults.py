"""
LINEAR COST OPTIMIZATION RESULTS ANALYSIS
==========================================

This script analyzes the results from the linear cost approximation optimization
to evaluate cooperation performance, success rates, and compare with original
piecewise cost results if available.

Key Analysis Areas:
1. Cooperation success rates by scenario
2. Profit improvements and distributions
3. Volume redistributions and capacity utilization
4. CI subsidy effectiveness
5. Feasibility and validation rates
6. Performance comparison: Linear vs Piecewise (if available)
7. Terminal-level cooperation patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Configuration
LINEAR_RESULTS_FOLDER = 'simulation_results_linear_cost'
ORIGINAL_RESULTS_FOLDER = 'simulation_results'  # For comparison if available
OUTPUT_FOLDER = 'linear_optimization_analysis'
COMPARISON_OUTPUT_FOLDER = 'linear_vs_piecewise_comparison'


class LinearOptimizationAnalyzer:
    """Comprehensive analyzer for linear cost optimization results."""

    def __init__(self, results_folder: str = LINEAR_RESULTS_FOLDER):
        self.results_folder = results_folder
        self.results_data = []
        self.summary_stats = {}

    def load_results(self) -> bool:
        """Load all optimization result files."""
        if not os.path.exists(self.results_folder):
            print(f"Error: Results folder '{self.results_folder}' not found.")
            print("Please run the linear cost optimization first.")
            return False

        result_files = [f for f in os.listdir(self.results_folder) if f.endswith('.pkl')]
        if not result_files:
            print(f"Error: No result files found in '{self.results_folder}'.")
            return False

        print(f"Loading {len(result_files)} result files...")

        filename_pattern = re.compile(r'(.+)_Sub(\d+)_CICombo_(.+)_LinearCost\.pkl')

        for filename in result_files:
            filepath = os.path.join(self.results_folder, filename)

            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                # Parse filename for metadata
                match = filename_pattern.match(filename)
                if match:
                    base_name, subsidy_str, ci_combo_str = match.groups()
                    subsidy_level = int(subsidy_str) / 100.0
                    ci_combo = ci_combo_str
                else:
                    print(f"Warning: Could not parse filename {filename}")
                    continue

                # Extract key information
                result_record = {
                    'filename': filename,
                    'base_scenario': base_name,
                    'subsidy_level': subsidy_level,
                    'ci_combo': ci_combo,
                    'num_terminals': data['simulation_params']['num_terminals_from_data'],
                    'ci_rate_data': data['simulation_params']['ci_rate_from_data'],
                    'instance_index': data['simulation_params']['instance_index'],
                    'subset_composition': data['simulation_params']['subset_composition'],
                    'optimization_results': data['optimization_output']
                }

                self.results_data.append(result_record)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

        print(f"Successfully loaded {len(self.results_data)} optimization results")
        return len(self.results_data) > 0

    def analyze_cooperation_success(self) -> pd.DataFrame:
        """Analyze cooperation success rates across different scenarios."""
        success_data = []

        for result in self.results_data:
            opt_results = result['optimization_results']
            feasibility_status = opt_results.get('feasibility_status', 'Unknown')

            # Determine if cooperation was successful
            is_feasible = 'Feasible' in feasibility_status
            is_infeasible = 'Infeasible' in feasibility_status
            has_validation_warnings = 'Validation Warnings' in feasibility_status

            # Calculate profit improvements
            profit_before = opt_results.get('profitBefore', np.array([]))
            profit_after_maxprof = opt_results.get('profitAfter_MAXPROF', np.array([]))
            profit_after_maxmin = opt_results.get('profitAfter_MAXMIN', np.array([]))

            if len(profit_before) > 0 and len(profit_after_maxprof) > 0:
                total_profit_improvement_maxprof = np.sum(profit_after_maxprof - profit_before)
                avg_profit_improvement_maxprof = np.mean(profit_after_maxprof - profit_before)
                max_profit_improvement_maxprof = np.max(profit_after_maxprof - profit_before)
                min_profit_improvement_maxprof = np.min(profit_after_maxprof - profit_before)
            else:
                total_profit_improvement_maxprof = 0
                avg_profit_improvement_maxprof = 0
                max_profit_improvement_maxprof = 0
                min_profit_improvement_maxprof = 0

            if len(profit_before) > 0 and len(profit_after_maxmin) > 0:
                total_profit_improvement_maxmin = np.sum(profit_after_maxmin - profit_before)
                avg_profit_improvement_maxmin = np.mean(profit_after_maxmin - profit_before)
            else:
                total_profit_improvement_maxmin = 0
                avg_profit_improvement_maxmin = 0

            success_data.append({
                'num_terminals': result['num_terminals'],
                'subsidy_level': result['subsidy_level'],
                'ci_combo': result['ci_combo'],
                'ci_rate_data': result['ci_rate_data'],
                'subset_composition': result['subset_composition'],
                'instance_index': result['instance_index'],
                'feasibility_status': feasibility_status,
                'is_feasible': is_feasible,
                'is_infeasible': is_infeasible,
                'has_validation_warnings': has_validation_warnings,
                'total_profit_improvement_maxprof': total_profit_improvement_maxprof,
                'avg_profit_improvement_maxprof': avg_profit_improvement_maxprof,
                'max_profit_improvement_maxprof': max_profit_improvement_maxprof,
                'min_profit_improvement_maxprof': min_profit_improvement_maxprof,
                'total_profit_improvement_maxmin': total_profit_improvement_maxmin,
                'avg_profit_improvement_maxmin': avg_profit_improvement_maxmin,
                'objval_maxprof': opt_results.get('objval_MAXPROF', np.nan),
                'objval_maxmin': opt_results.get('objval_MAXMIN', np.nan),
                'optimality_gap_maxprof': opt_results.get('optimalityGap_MAXPROF', np.nan),
                'optimality_gap_maxmin': opt_results.get('optimalityGap_MAXMIN', np.nan)
            })

        return pd.DataFrame(success_data)

    def analyze_terminal_level_impacts(self) -> pd.DataFrame:
        """Analyze terminal-level cooperation impacts."""
        terminal_data = []

        for result in self.results_data:
            opt_results = result['optimization_results']
            profit_before = opt_results.get('profitBefore', np.array([]))
            profit_after_maxprof = opt_results.get('profitAfter_MAXPROF', np.array([]))
            volume_before = opt_results.get('volumeBefore', np.array([]))
            volume_after_maxprof = opt_results.get('volumeAfter_MAXPROF', np.array([]))

            for terminal_idx in range(result['num_terminals']):
                if (len(profit_before) > terminal_idx and
                        len(profit_after_maxprof) > terminal_idx and
                        len(volume_before) > terminal_idx and
                        len(volume_after_maxprof) > terminal_idx):
                    profit_change = profit_after_maxprof[terminal_idx] - profit_before[terminal_idx]
                    volume_change = volume_after_maxprof[terminal_idx] - volume_before[terminal_idx]
                    volume_change_percent = (volume_change / volume_before[terminal_idx] * 100
                                             if volume_before[terminal_idx] > 0 else 0)

                    terminal_data.append({
                        'scenario_id': f"{result['base_scenario']}_Sub{int(result['subsidy_level'] * 100)}_CI{result['ci_combo']}",
                        'num_terminals': result['num_terminals'],
                        'terminal_id': terminal_idx + 1,
                        'subsidy_level': result['subsidy_level'],
                        'ci_combo': result['ci_combo'],
                        'ci_rate_data': result['ci_rate_data'],
                        'subset_composition': result['subset_composition'],
                        'profit_before': profit_before[terminal_idx],
                        'profit_after': profit_after_maxprof[terminal_idx],
                        'profit_change': profit_change,
                        'profit_change_percent': (profit_change / profit_before[terminal_idx] * 100
                                                  if profit_before[terminal_idx] > 0 else 0),
                        'volume_before': volume_before[terminal_idx],
                        'volume_after': volume_after_maxprof[terminal_idx],
                        'volume_change': volume_change,
                        'volume_change_percent': volume_change_percent,
                        'feasibility_status': opt_results.get('feasibility_status', 'Unknown')
                    })

        return pd.DataFrame(terminal_data)

    def create_summary_statistics(self, cooperation_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        stats = {}

        # Overall success rates
        total_scenarios = len(cooperation_df)
        feasible_scenarios = cooperation_df['is_feasible'].sum()
        infeasible_scenarios = cooperation_df['is_infeasible'].sum()
        validation_warning_scenarios = cooperation_df['has_validation_warnings'].sum()

        stats['total_scenarios'] = total_scenarios
        stats['feasible_scenarios'] = feasible_scenarios
        stats['infeasible_scenarios'] = infeasible_scenarios
        stats['validation_warning_scenarios'] = validation_warning_scenarios
        stats['feasibility_rate'] = feasible_scenarios / total_scenarios * 100 if total_scenarios > 0 else 0

        # Profit improvement statistics (only for feasible scenarios)
        feasible_df = cooperation_df[cooperation_df['is_feasible']]
        if not feasible_df.empty:
            stats['mean_total_profit_improvement'] = feasible_df['total_profit_improvement_maxprof'].mean()
            stats['median_total_profit_improvement'] = feasible_df['total_profit_improvement_maxprof'].median()
            stats['max_total_profit_improvement'] = feasible_df['total_profit_improvement_maxprof'].max()
            stats['min_total_profit_improvement'] = feasible_df['total_profit_improvement_maxprof'].min()
            stats['std_total_profit_improvement'] = feasible_df['total_profit_improvement_maxprof'].std()

            # Percentage of scenarios with positive total profit improvement
            stats['positive_improvement_rate'] = (feasible_df['total_profit_improvement_maxprof'] > 0).mean() * 100

        # Success rates by subsidy level
        stats['success_by_subsidy'] = cooperation_df.groupby('subsidy_level')['is_feasible'].agg(
            ['sum', 'count', 'mean']).to_dict('index')

        # Success rates by number of terminals
        stats['success_by_terminals'] = cooperation_df.groupby('num_terminals')['is_feasible'].agg(
            ['sum', 'count', 'mean']).to_dict('index')

        # Success rates by CI rate
        stats['success_by_ci_rate'] = cooperation_df.groupby('ci_rate_data')['is_feasible'].agg(
            ['sum', 'count', 'mean']).to_dict('index')

        self.summary_stats = stats
        return stats

    def create_summary_plots(self, cooperation_df: pd.DataFrame, terminal_df: pd.DataFrame):
        """Create comprehensive summary plots."""
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        plt.style.use('default')
        sns.set_palette("husl")

        # Plot 1: Success Rate Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Linear Cost Optimization: Cooperation Success Analysis', fontsize=16, fontweight='bold')

        # Success rates by subsidy level
        subsidy_success = cooperation_df.groupby('subsidy_level')['is_feasible'].mean() * 100
        axes[0, 0].bar(subsidy_success.index, subsidy_success.values, alpha=0.7)
        axes[0, 0].set_xlabel('Subsidy Level ($)')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_title('Cooperation Success Rate by Subsidy Level')
        axes[0, 0].grid(True, alpha=0.3)

        # Success rates by number of terminals
        terminal_success = cooperation_df.groupby('num_terminals')['is_feasible'].mean() * 100
        axes[0, 1].bar(terminal_success.index, terminal_success.values, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Number of Terminals')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Cooperation Success Rate by Terminal Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Profit improvement distribution (feasible scenarios only)
        feasible_scenarios = cooperation_df[cooperation_df['is_feasible']]
        if not feasible_scenarios.empty:
            axes[1, 0].hist(feasible_scenarios['total_profit_improvement_maxprof'], bins=30, alpha=0.7, color='green')
            axes[1, 0].axvline(feasible_scenarios['total_profit_improvement_maxprof'].mean(),
                               color='red', linestyle='--',
                               label=f'Mean: ${feasible_scenarios["total_profit_improvement_maxprof"].mean():.0f}')
            axes[1, 0].set_xlabel('Total Profit Improvement ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Profit Improvements')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # CI effectiveness
        ci_success = cooperation_df.groupby('ci_rate_data')['is_feasible'].mean() * 100
        axes[1, 1].bar(ci_success.index, ci_success.values, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('CI Rate from Data')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Cooperation Success Rate by CI Rate')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'cooperation_success_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Terminal-Level Impact Analysis
        if not terminal_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Terminal-Level Cooperation Impacts', fontsize=16, fontweight='bold')

            # Profit change distribution
            axes[0, 0].hist(terminal_df['profit_change'], bins=50, alpha=0.7)
            axes[0, 0].axvline(terminal_df['profit_change'].mean(), color='red', linestyle='--',
                               label=f'Mean: ${terminal_df["profit_change"].mean():.0f}')
            axes[0, 0].set_xlabel('Profit Change ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Terminal Profit Changes')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Volume change distribution
            axes[0, 1].hist(terminal_df['volume_change_percent'], bins=50, alpha=0.7, color='orange')
            axes[0, 1].axvline(terminal_df['volume_change_percent'].mean(), color='red', linestyle='--',
                               label=f'Mean: {terminal_df["volume_change_percent"].mean():.1f}%')
            axes[0, 1].set_xlabel('Volume Change (%)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Terminal Volume Changes')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Profit vs Volume change correlation
            axes[1, 0].scatter(terminal_df['volume_change_percent'], terminal_df['profit_change'], alpha=0.5)
            axes[1, 0].set_xlabel('Volume Change (%)')
            axes[1, 0].set_ylabel('Profit Change ($)')
            axes[1, 0].set_title('Profit Change vs Volume Change')
            axes[1, 0].grid(True, alpha=0.3)

            # Impact by subsidy level
            subsidy_impact = terminal_df.groupby('subsidy_level')['profit_change'].mean()
            axes[1, 1].bar(subsidy_impact.index, subsidy_impact.values, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Subsidy Level ($)')
            axes[1, 1].set_ylabel('Average Profit Change ($)')
            axes[1, 1].set_title('Average Terminal Profit Impact by Subsidy')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, 'terminal_level_impacts.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def export_detailed_results(self, cooperation_df: pd.DataFrame, terminal_df: pd.DataFrame):
        """Export detailed analysis results to Excel and CSV."""
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Export to Excel with multiple sheets
        excel_path = os.path.join(OUTPUT_FOLDER, 'linear_optimization_analysis.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            cooperation_df.to_excel(writer, sheet_name='Cooperation_Analysis', index=False)
            terminal_df.to_excel(writer, sheet_name='Terminal_Level_Analysis', index=False)

            # Summary statistics sheet
            summary_data = []
            for key, value in self.summary_stats.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        summary_data.append({'Metric': f"{key}_{sub_key}", 'Value': str(sub_value)})
                else:
                    summary_data.append({'Metric': key, 'Value': value})

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        # Also save as CSV
        cooperation_df.to_csv(os.path.join(OUTPUT_FOLDER, 'cooperation_analysis.csv'), index=False)
        terminal_df.to_csv(os.path.join(OUTPUT_FOLDER, 'terminal_analysis.csv'), index=False)

        print(f"Detailed results exported to:")
        print(f"  Excel: {excel_path}")
        print(f"  CSV files in: {OUTPUT_FOLDER}")

    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "=" * 80)
        print("LINEAR COST OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)

        stats = self.summary_stats

        print(f"\nOverall Performance:")
        print(f"  Total scenarios analyzed: {stats['total_scenarios']}")
        print(f"  Feasible (successful) scenarios: {stats['feasible_scenarios']}")
        print(f"  Infeasible scenarios: {stats['infeasible_scenarios']}")
        print(f"  Scenarios with validation warnings: {stats['validation_warning_scenarios']}")
        print(f"  Overall feasibility rate: {stats['feasibility_rate']:.1f}%")

        if 'mean_total_profit_improvement' in stats:
            print(f"\nProfit Improvement Analysis (Feasible Scenarios):")
            print(f"  Mean total profit improvement: ${stats['mean_total_profit_improvement']:,.0f}")
            print(f"  Median total profit improvement: ${stats['median_total_profit_improvement']:,.0f}")
            print(f"  Maximum improvement: ${stats['max_total_profit_improvement']:,.0f}")
            print(f"  Minimum improvement: ${stats['min_total_profit_improvement']:,.0f}")
            print(f"  Scenarios with positive improvement: {stats['positive_improvement_rate']:.1f}%")

        print(f"\nSuccess Rates by Subsidy Level:")
        for subsidy, data in stats['success_by_subsidy'].items():
            print(f"  ${subsidy:.2f}: {data['mean'] * 100:.1f}% ({data['sum']}/{data['count']})")

        print(f"\nSuccess Rates by Terminal Count:")
        for terminals, data in stats['success_by_terminals'].items():
            print(f"  {terminals} terminals: {data['mean'] * 100:.1f}% ({data['sum']}/{data['count']})")

        print(f"\nSuccess Rates by CI Rate:")
        for ci_rate, data in stats['success_by_ci_rate'].items():
            print(f"  {ci_rate:.0%} CI rate: {data['mean'] * 100:.1f}% ({data['sum']}/{data['count']})")

        print("\n" + "=" * 80)

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("LINEAR COST OPTIMIZATION RESULTS ANALYSIS")
        print("=" * 50)

        # Load results
        if not self.load_results():
            return False

        # Analyze cooperation success
        print("Analyzing cooperation success patterns...")
        cooperation_df = self.analyze_cooperation_success()

        # Analyze terminal-level impacts
        print("Analyzing terminal-level impacts...")
        terminal_df = self.analyze_terminal_level_impacts()

        # Generate summary statistics
        print("Generating summary statistics...")
        self.create_summary_statistics(cooperation_df)

        # Create visualizations
        print("Creating analysis plots...")
        self.create_summary_plots(cooperation_df, terminal_df)

        # Export detailed results
        print("Exporting detailed results...")
        self.export_detailed_results(cooperation_df, terminal_df)

        # Print summary report
        self.print_summary_report()

        print(f"\nAnalysis complete! Results saved to: {OUTPUT_FOLDER}")
        return True


def compare_linear_vs_piecewise(linear_folder: str = LINEAR_RESULTS_FOLDER,
                                piecewise_folder: str = ORIGINAL_RESULTS_FOLDER):
    """Compare linear vs piecewise optimization results if both are available."""
    if not (os.path.exists(linear_folder) and os.path.exists(piecewise_folder)):
        print("Both linear and piecewise results needed for comparison.")
        print(f"Linear folder exists: {os.path.exists(linear_folder)}")
        print(f"Piecewise folder exists: {os.path.exists(piecewise_folder)}")
        return False

    print("\nCOMPARING LINEAR VS PIECEWISE OPTIMIZATION RESULTS")
    print("=" * 60)

    # This would require loading both sets of results and comparing
    # Success rates, profit improvements, solve times, etc.
    # Implementation would be similar to the main analyzer but comparing two datasets

    print("Comparison functionality would be implemented here...")
    print("This would compare success rates, profit improvements, solve times, etc.")

    return True


if __name__ == "__main__":
    # Main analysis
    analyzer = LinearOptimizationAnalyzer()
    success = analyzer.run_full_analysis()

    if success:
        # Optional: Run comparison with piecewise results if available
        print("\nChecking for piecewise results for comparison...")
        compare_linear_vs_piecewise()
    else:
        print("Analysis failed. Please check that linear optimization results are available.")