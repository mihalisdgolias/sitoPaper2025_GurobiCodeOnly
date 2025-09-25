"""
===============================================================================
TERMINAL COOPERATION RESULTS ANALYZER
===============================================================================

Analyzes simulation results to generate Excel tables and visualizations
showing the impact of CI capabilities and subsidies on terminal cooperation.

Inputs: CSV file from simulation runs
Outputs: Excel files with analysis tables + PNG graphs

===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")


class TerminalResultsAnalyzer:
    """
    Analyzes terminal cooperation simulation results and generates
    Excel tables and visualizations.
    """

    def __init__(self, results_file: str = "simulation_results/summary_results.csv"):
        """Initialize analyzer with simulation results."""
        self.results_file = Path(results_file)
        self.df = None
        self.output_tables = {}

        # These will be extracted from data
        self.subsidy_levels = None
        self.ci_rates = None
        self.terminal_counts = None

        print(f"Terminal Results Analyzer")
        print(f"Input file: {self.results_file}")

    def load_and_prepare_data(self):
        """Load simulation results and extract parameters."""
        try:
            self.df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.df)} records from simulation")

            # Extract unique parameter values from data
            self.subsidy_levels = sorted(self.df['Subsidy_Level'].unique())
            self.ci_rates = sorted(self.df['CI_Rate_Data'].unique())
            self.terminal_counts = sorted(self.df['Num_Terminals'].unique())

            print(f"Subsidy levels found: {self.subsidy_levels}")
            print(f"CI rates found: {self.ci_rates}")
            print(f"Terminal counts found: {self.terminal_counts}")

            # Create derived variables
            self.df['Cooperation_Occurred'] = self.df['Profit_Change'] > 0.01
            self.df['Profit_Improvement_Pct'] = (self.df['Profit_Change'] / self.df['Profit_Before']) * 100
            self.df['Has_CI_Terminal'] = self.df['CI_Terminals_Combo'] != 'None'

            # Add volume change metrics
            volume_before = self.df.groupby(['Subset_Composition', 'Instance_Index', 'CI_Rate_Data'])[
                'Profit_Before'].transform('sum')
            self.df['Volume_Change_Pct'] = ((self.df['Volume_After'] - volume_before) / volume_before * 100).fillna(0)

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def generate_summary_statistics_table(self):
        """Generate overall summary statistics table."""
        print("Generating summary statistics table...")

        # Overall statistics
        summary_stats = {
            'Metric': [
                'Total Scenarios', 'Total Terminal-Scenarios', 'Cooperation Rate (%)',
                'Avg Profit Before ($)', 'Avg Profit After ($)', 'Avg Profit Change ($)',
                'Avg Profit Change (%)', 'Scenarios with CI Terminals (%)'
            ],
            'Value': [
                len(self.df.groupby(['Subset_Composition', 'Instance_Index', 'Subsidy_Level', 'CI_Terminals_Combo'])),
                len(self.df),
                self.df['Cooperation_Occurred'].mean() * 100,
                self.df['Profit_Before'].mean(),
                self.df['Profit_After'].mean(),
                self.df['Profit_Change'].mean(),
                self.df['Profit_Improvement_Pct'].mean(),
                self.df['Has_CI_Terminal'].mean() * 100
            ]
        }

        summary_df = pd.DataFrame(summary_stats)
        summary_df['Value'] = summary_df['Value'].round(2)
        self.output_tables['Summary_Statistics'] = summary_df

        return summary_df

    def generate_subsidy_analysis_table(self):
        """Generate subsidy effectiveness analysis table."""
        print("Generating subsidy analysis table...")

        # Analyze by subsidy level
        subsidy_analysis = self.df.groupby('Subsidy_Level').agg({
            'Cooperation_Occurred': ['count', 'sum', 'mean'],
            'Profit_Change': ['mean', 'std'],
            'Profit_Improvement_Pct': ['mean', 'std'],
            'Volume_After': 'mean',
            'Has_CI_Terminal': 'mean'
        }).round(3)

        # Flatten column names
        subsidy_analysis.columns = [
            'Total_Scenarios', 'Cooperation_Count', 'Cooperation_Rate',
            'Avg_Profit_Change', 'Std_Profit_Change',
            'Avg_Profit_Change_Pct', 'Std_Profit_Change_Pct',
            'Avg_Volume_After', 'CI_Terminal_Rate'
        ]

        subsidy_analysis['Cooperation_Rate'] = subsidy_analysis['Cooperation_Rate'] * 100
        subsidy_analysis['CI_Terminal_Rate'] = subsidy_analysis['CI_Terminal_Rate'] * 100
        subsidy_analysis = subsidy_analysis.reset_index()

        self.output_tables['Subsidy_Analysis'] = subsidy_analysis
        return subsidy_analysis

    def generate_ci_analysis_table(self):
        """Generate CI capability analysis table."""
        print("Generating CI capability analysis table...")

        # Analyze by CI rate
        ci_analysis = self.df.groupby('CI_Rate_Data').agg({
            'Cooperation_Occurred': ['count', 'sum', 'mean'],
            'Profit_Change': ['mean', 'std'],
            'Profit_Improvement_Pct': ['mean', 'std'],
            'Volume_After': 'mean'
        }).round(3)

        # Flatten column names
        ci_analysis.columns = [
            'Total_Scenarios', 'Cooperation_Count', 'Cooperation_Rate',
            'Avg_Profit_Change', 'Std_Profit_Change',
            'Avg_Profit_Change_Pct', 'Std_Profit_Change_Pct',
            'Avg_Volume_After'
        ]

        ci_analysis['Cooperation_Rate'] = ci_analysis['Cooperation_Rate'] * 100
        ci_analysis = ci_analysis.reset_index()

        self.output_tables['CI_Analysis'] = ci_analysis
        return ci_analysis

    def generate_combined_subsidy_ci_table(self):
        """Generate combined subsidy and CI analysis table."""
        print("Generating combined subsidy-CI analysis table...")

        # Cross-analysis of subsidy and CI rates
        combined_analysis = self.df.groupby(['Subsidy_Level', 'CI_Rate_Data']).agg({
            'Cooperation_Occurred': ['count', 'mean'],
            'Profit_Change': 'mean',
            'Volume_After': 'mean'
        }).round(3)

        # Flatten and reshape
        combined_analysis.columns = ['Scenario_Count', 'Cooperation_Rate', 'Avg_Profit_Change', 'Avg_Volume_After']
        combined_analysis['Cooperation_Rate'] = combined_analysis['Cooperation_Rate'] * 100
        combined_analysis = combined_analysis.reset_index()

        self.output_tables['Combined_Subsidy_CI_Analysis'] = combined_analysis
        return combined_analysis

    def generate_terminal_count_analysis_table(self):
        """Generate terminal count (network effects) analysis table."""
        print("Generating terminal count analysis table...")

        terminal_count_analysis = self.df.groupby('Num_Terminals').agg({
            'Cooperation_Occurred': ['count', 'sum', 'mean'],
            'Profit_Change': ['mean', 'std'],
            'Profit_Improvement_Pct': ['mean', 'std'],
            'Volume_After': 'mean'
        }).round(3)

        # Flatten column names
        terminal_count_analysis.columns = [
            'Total_Scenarios', 'Cooperation_Count', 'Cooperation_Rate',
            'Avg_Profit_Change', 'Std_Profit_Change',
            'Avg_Profit_Change_Pct', 'Std_Profit_Change_Pct',
            'Avg_Volume_After'
        ]

        terminal_count_analysis['Cooperation_Rate'] = terminal_count_analysis['Cooperation_Rate'] * 100
        terminal_count_analysis = terminal_count_analysis.reset_index()

        self.output_tables['Terminal_Count_Analysis'] = terminal_count_analysis
        return terminal_count_analysis

    def generate_objective_comparison_table(self):
        """Generate MAXPROF vs MAXMIN objective comparison table."""
        print("Generating objective comparison table...")

        objective_comparison = self.df.groupby('Objective').agg({
            'Cooperation_Occurred': ['count', 'mean'],
            'Profit_Change': ['mean', 'std'],
            'Profit_Improvement_Pct': ['mean', 'std'],
            'Volume_After': 'mean',
            'Objective_Value': ['mean', 'std']
        }).round(3)

        # Flatten column names
        objective_comparison.columns = [
            'Total_Scenarios', 'Cooperation_Rate',
            'Avg_Profit_Change', 'Std_Profit_Change',
            'Avg_Profit_Change_Pct', 'Std_Profit_Change_Pct',
            'Avg_Volume_After', 'Avg_Objective_Value', 'Std_Objective_Value'
        ]

        objective_comparison['Cooperation_Rate'] = objective_comparison['Cooperation_Rate'] * 100
        objective_comparison = objective_comparison.reset_index()

        self.output_tables['Objective_Comparison'] = objective_comparison
        return objective_comparison

    def generate_detailed_scenario_table(self):
        """Generate detailed scenario-level results table."""
        print("Generating detailed scenario table...")

        # Aggregate to scenario level (remove terminal-level duplicates)
        scenario_level = self.df.groupby([
            'Num_Terminals', 'Subset_Composition', 'CI_Rate_Data',
            'Instance_Index', 'Subsidy_Level', 'CI_Terminals_Combo', 'Objective'
        ]).agg({
            'Cooperation_Occurred': 'mean',  # Should be same for all terminals in scenario
            'Profit_Change': 'sum',  # Total profit change across all terminals
            'Profit_Before': 'sum',  # Total profit before
            'Profit_After': 'sum',  # Total profit after
            'Volume_After': 'sum',  # Total volume
            'Feasibility_Status': 'first',
            'Objective_Value': 'first',
            'Optimality_Gap': 'first'
        }).round(2)

        scenario_level = scenario_level.reset_index()
        scenario_level['Profit_Improvement_Pct'] = (
                scenario_level['Profit_Change'] / scenario_level['Profit_Before'] * 100
        ).round(2)

        self.output_tables['Detailed_Scenarios'] = scenario_level
        return scenario_level

    def generate_statistical_tests_table(self):
        """Generate statistical significance tests table."""
        print("Generating statistical tests table...")

        test_results = []

        # Test 1: Subsidy effect on profit change
        try:
            low_subsidy = self.df[self.df['subsidy'] == min(self.subsidy_levels)]['total_profit_increase']
            high_subsidy = self.df[self.df['subsidy'] == max(self.subsidy_levels)]['total_profit_increase']
            t_stat, p_val = stats.ttest_ind(low_subsidy.dropna(), high_subsidy.dropna())

            test_results.append({
                'Test': 'Subsidy Effect on Profit (T-test)',
                'Statistic': t_stat,
                'P_Value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
        except Exception as e:
            print(f"Warning: Could not perform subsidy effect test: {e}")

        # Test 2: CI rate effect on profit change
        try:
            low_ci = self.df[self.df['ci_rate'] == min(self.ci_rates)]['total_profit_increase']
            high_ci = self.df[self.df['ci_rate'] == max(self.ci_rates)]['total_profit_increase']
            t_stat, p_val = stats.ttest_ind(low_ci.dropna(), high_ci.dropna())

            test_results.append({
                'Test': 'CI Rate Effect on Profit (T-test)',
                'Statistic': t_stat,
                'P_Value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
        except Exception as e:
            print(f"Warning: Could not perform CI rate effect test: {e}")

        # Test 3: Objective function comparison
        try:
            maxprof_profit = self.df[self.df['objective'] == 'MAXPROF']['total_profit_increase']
            maxmin_profit = self.df[self.df['objective'] == 'MAXMIN']['total_profit_increase']
            t_stat, p_val = stats.ttest_ind(maxprof_profit.dropna(), maxmin_profit.dropna())

            test_results.append({
                'Test': 'MAXPROF vs MAXMIN Profit (T-test)',
                'Statistic': t_stat,
                'P_Value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
        except Exception as e:
            print(f"Warning: Could not perform objective comparison test: {e}")

        # Create results DataFrame
        test_df = pd.DataFrame(test_results)
        if not test_df.empty:
            test_df[['Statistic', 'P_Value']] = test_df[['Statistic', 'P_Value']].round(4)

        self.output_tables['Statistical_Tests'] = test_df
        return test_df

    def save_all_tables_to_excel(self):
        """Save all analysis tables to Excel files."""
        print("Saving tables to Excel files...")

        # Save individual tables
        for table_name, df in self.output_tables.items():
            filename = f"{table_name}.xlsx"
            df.to_excel(filename, index=False)
            print(f"Saved {filename}")

        # Save combined workbook
        with pd.ExcelWriter('Terminal_Cooperation_Analysis.xlsx') as writer:
            for sheet_name, df in self.output_tables.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print("Saved combined workbook: Terminal_Cooperation_Analysis.xlsx")

    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")

        # Set up figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Terminal Cooperation Analysis Results', fontsize=16, fontweight='bold')

        # 1. Cooperation rate by subsidy level
        ax1 = axes[0, 0]
        subsidy_coop = self.df.groupby('Subsidy_Level')['Cooperation_Occurred'].mean()
        ax1.plot(subsidy_coop.index, subsidy_coop.values * 100, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Subsidy Level ($/TEU)')
        ax1.set_ylabel('Cooperation Rate (%)')
        ax1.set_title('Cooperation Rate vs Subsidy Level')
        ax1.grid(True, alpha=0.3)

        # 2. Profit change by subsidy level
        ax2 = axes[0, 1]
        subsidy_profit = self.df.groupby('Subsidy_Level')['Profit_Change'].mean()
        ax2.plot(subsidy_profit.index, subsidy_profit.values, 's-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Subsidy Level ($/TEU)')
        ax2.set_ylabel('Average Profit Change ($)')
        ax2.set_title('Profit Impact vs Subsidy Level')
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap: Cooperation rate by subsidy and CI rate
        ax3 = axes[0, 2]
        pivot_coop = self.df.pivot_table(
            values='Cooperation_Occurred',
            index='CI_Rate_Data',
            columns='Subsidy_Level',
            aggfunc='mean'
        )
        im = ax3.imshow(pivot_coop.values, aspect='auto', cmap='RdYlBu_r')
        ax3.set_xticks(range(len(pivot_coop.columns)))
        ax3.set_xticklabels(pivot_coop.columns)
        ax3.set_yticks(range(len(pivot_coop.index)))
        ax3.set_yticklabels([f"{x:.1%}" for x in pivot_coop.index])
        ax3.set_xlabel('Subsidy Level ($/TEU)')
        ax3.set_ylabel('CI Rate')
        ax3.set_title('Cooperation Rate Heatmap')
        plt.colorbar(im, ax=ax3, label='Cooperation Rate')

        # 4. CI rate effect on cooperation
        ax4 = axes[1, 0]
        ci_coop = self.df.groupby('CI_Rate_Data')['Cooperation_Occurred'].mean()
        ax4.bar(range(len(ci_coop)), ci_coop.values * 100, alpha=0.7, color='skyblue')
        ax4.set_xticks(range(len(ci_coop)))
        ax4.set_xticklabels([f"{x:.1%}" for x in ci_coop.index])
        ax4.set_xlabel('CI Rate')
        ax4.set_ylabel('Cooperation Rate (%)')
        ax4.set_title('Cooperation Rate by CI Rate')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Terminal count effect (network effects)
        ax5 = axes[1, 1]
        terminal_coop = self.df.groupby('Num_Terminals')['Cooperation_Occurred'].mean()
        ax5.bar(terminal_coop.index, terminal_coop.values * 100, alpha=0.7, color='orange')
        ax5.set_xlabel('Number of Terminals')
        ax5.set_ylabel('Cooperation Rate (%)')
        ax5.set_title('Network Effects on Cooperation')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Profit distribution by cooperation status
        ax6 = axes[1, 2]
        coop_profits = self.df[self.df['Cooperation_Occurred'] == True]['Profit_Change']
        no_coop_profits = self.df[self.df['Cooperation_Occurred'] == False]['Profit_Change']
        ax6.boxplot([coop_profits.dropna(), no_coop_profits.dropna()],
                    labels=['Cooperation', 'No Cooperation'])
        ax6.set_ylabel('Profit Change ($)')
        ax6.set_title('Profit Change Distribution')
        ax6.grid(True, alpha=0.3, axis='y')

        # 7. Objective function comparison
        ax7 = axes[2, 0]
        obj_comparison = self.df.groupby('Objective')['Cooperation_Occurred'].mean()
        ax7.bar(obj_comparison.index, obj_comparison.values * 100, alpha=0.7, color='purple')
        ax7.set_ylabel('Cooperation Rate (%)')
        ax7.set_title('MAXPROF vs MAXMIN Cooperation')
        ax7.grid(True, alpha=0.3, axis='y')

        # 8. Volume changes by subsidy
        ax8 = axes[2, 1]
        volume_changes = self.df.groupby('Subsidy_Level')['Volume_After'].mean()
        ax8.plot(volume_changes.index, volume_changes.values, '^-', linewidth=2, markersize=8, color='red')
        ax8.set_xlabel('Subsidy Level ($/TEU)')
        ax8.set_ylabel('Average Volume After (TEU)')
        ax8.set_title('Volume Changes by Subsidy Level')
        ax8.grid(True, alpha=0.3)

        # 9. Scatter: Profit change vs Volume after
        ax9 = axes[2, 2]
        cooperating = self.df[self.df['Cooperation_Occurred'] == True]
        not_cooperating = self.df[self.df['Cooperation_Occurred'] == False]
        ax9.scatter(cooperating['Volume_After'], cooperating['Profit_Change'],
                    alpha=0.6, label='Cooperation', color='green')
        ax9.scatter(not_cooperating['Volume_After'], not_cooperating['Profit_Change'],
                    alpha=0.6, label='No Cooperation', color='red')
        ax9.set_xlabel('Volume After (TEU)')
        ax9.set_ylabel('Profit Change ($)')
        ax9.set_title('Profit vs Volume Relationship')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Terminal_Cooperation_Analysis.png', dpi=300, bbox_inches='tight')
        print("Saved visualization: Terminal_Cooperation_Analysis.png")

        return fig

    def run_analysis(self):
        """Run complete analysis and generate outputs."""
        print("Starting Terminal Cooperation Results Analysis")
        print("=" * 60)

        # Load data
        if not self.load_and_prepare_data():
            print("Failed to load data. Analysis cannot proceed.")
            return

        # Generate all analysis tables
        self.generate_summary_statistics_table()
        self.generate_subsidy_analysis_table()
        self.generate_ci_analysis_table()
        self.generate_combined_subsidy_ci_table()
        self.generate_terminal_count_analysis_table()
        self.generate_objective_comparison_table()
        self.generate_detailed_scenario_table()
        self.generate_statistical_tests_table()

        # Save to Excel
        self.save_all_tables_to_excel()

        # Create visualizations
        self.create_visualizations()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print(f"Generated {len(self.output_tables)} Excel tables")
        print("Created comprehensive visualization")
        print("All files saved to current directory")
        print("=" * 60)


def main():
    """Main execution function."""
    analyzer = TerminalResultsAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()