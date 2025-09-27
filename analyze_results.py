import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# --- Configuration ---
RESULTS_FOLDER = "simulation_results"
EXCEL_OUTPUT_FILE = "analysis_summary.xlsx"
PLOT_FOLDER = "analysis_plots"


# --- Helper Functions ---

def load_all_results(folder: str) -> List[Dict[str, Any]]:
    """Recursively loads all .pkl files from a specified folder."""
    all_data = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.pkl'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        all_data.append(data)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    return all_data


def consolidate_data(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Consolidates loaded data into a single DataFrame."""
    consolidated_list = []
    for res in all_results:
        sim_params = res['simulation_params']
        opt_output = res['optimization_output']

        num_terminals = sim_params['num_terminals_from_data']
        pricing_mechanism = sim_params['pricing_mechanism']

        # Check if optimization was successful to get profit/volume after
        if opt_output.get('feasibility_status') == 'Feasible and Sensible (optimized pricing)':
            for obj_name in ['MAXPROF', 'MAXMIN']:
                profit_after = opt_output.get(f'profitAfter_{obj_name}')

                # Check if profit_after is a valid array
                if profit_after is not None and isinstance(profit_after, np.ndarray):
                    for i in range(num_terminals):
                        row = {
                            'num_terminals': num_terminals,
                            'subset_composition': sim_params['subset_composition'],
                            'instance_index': sim_params['instance_index'],
                            'subsidy_level': sim_params['subsidy'],
                            'ci_terminals_combo': sim_params['ci_terminals'],
                            'pricing_mechanism': pricing_mechanism,
                            'objective': obj_name,
                            'terminal_id': i,
                            'profit_before': opt_output['profitBefore'][i],
                            'profit_after': profit_after[i],
                            'profit_change': profit_after[i] - opt_output['profitBefore'][i],
                            'total_profit_before': np.sum(opt_output['profitBefore']),
                            'total_profit_after': np.sum(opt_output.get(f'profitAfter_{obj_name}', [0])),
                            'feasibility_status': opt_output['feasibility_status'],
                            'cooperative': True
                        }
                        consolidated_list.append(row)
        else:
            # Handle non-cooperative instances
            for i in range(num_terminals):
                row = {
                    'num_terminals': num_terminals,
                    'subset_composition': sim_params['subset_composition'],
                    'instance_index': sim_params['instance_index'],
                    'subsidy_level': sim_params['subsidy'],
                    'ci_terminals_combo': sim_params['ci_terminals'],
                    'pricing_mechanism': pricing_mechanism,
                    'objective': 'MAXPROF',  # Use MAXPROF as a proxy for non-cooperative
                    'terminal_id': i,
                    'profit_before': opt_output['profitBefore'][i],
                    'profit_after': opt_output['profitBefore'][i],
                    'profit_change': 0,
                    'total_profit_before': np.sum(opt_output['profitBefore']),
                    'total_profit_after': np.sum(opt_output['profitBefore']),
                    'feasibility_status': opt_output['feasibility_status'],
                    'cooperative': False
                }
                consolidated_list.append(row)

    return pd.DataFrame(consolidated_list)


def analyze_and_plot(df: pd.DataFrame, writer: pd.ExcelWriter):
    """Performs analysis and generates plots."""

    os.makedirs(PLOT_FOLDER, exist_ok=True)

    # --- Overall Cooperation Analysis ---
    cooperation_counts = df.drop_duplicates(
        subset=['instance_index', 'pricing_mechanism', 'subset_composition', 'subsidy_level'])
    coop_rate = cooperation_counts['cooperative'].value_counts(normalize=True)
    print("\n--- Overall Cooperation Rate ---")
    print(coop_rate)
    coop_rate.to_excel(writer, sheet_name='Cooperation Rate')

    # --- Total Profit Analysis ---
    total_profits = df.drop_duplicates(
        subset=['instance_index', 'pricing_mechanism', 'subset_composition', 'subsidy_level', 'objective'])
    total_profits = total_profits[total_profits['cooperative'] == True].copy()

    total_profit_change = total_profits.groupby(['pricing_mechanism', 'objective']).agg(
        avg_total_profit_change=('total_profit_after',
                                 lambda x: (x - total_profits.loc[x.index, 'total_profit_before']).mean()),
        median_total_profit_change=('total_profit_after',
                                    lambda x: (x - total_profits.loc[x.index, 'total_profit_before']).median())
    ).reset_index()

    print("\n--- Total Profit Change by Pricing and Objective ---")
    print(total_profit_change)
    total_profit_change.to_excel(writer, sheet_name='Total Profit Change')

    # --- Individual Terminal Profit Analysis ---
    individual_profit_change = df[df['cooperative'] == True].groupby(['pricing_mechanism', 'objective']).agg(
        avg_profit_change=('profit_change', 'mean'),
        median_profit_change=('profit_change', 'median')
    ).reset_index()

    print("\n--- Individual Terminal Profit Change ---")
    print(individual_profit_change)
    individual_profit_change.to_excel(writer, sheet_name='Individual Profit Change')

    # --- Plotting ---

    # 1. Total Profit Increase by Objective
    plt.figure(figsize=(10, 6))
    sns.barplot(data=total_profit_change, x='objective', y='avg_total_profit_change', hue='pricing_mechanism')
    plt.title('Average Total Profit Increase by Objective and Pricing')
    plt.xlabel('Objective Function')
    plt.ylabel('Average Total Profit Increase ($)')
    plt.savefig(os.path.join(PLOT_FOLDER, 'total_profit_increase.png'))
    plt.close()

    # 2. Individual Terminal Profit Histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df[df['cooperative'] == True], x='profit_change', hue='objective', bins=30, kde=True)
    plt.title('Distribution of Individual Terminal Profit Changes (Cooperative Instances)')
    plt.xlabel('Individual Terminal Profit Change ($)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOT_FOLDER, 'individual_profit_histogram.png'))
    plt.close()

    # 3. Profit Comparison Scatter Plot
    # Requires a merge to compare MAXPROF vs MAXMIN profits for the same instance
    maxprof_df = df[(df['cooperative'] == True) & (df['objective'] == 'MAXPROF')].copy()
    maxmin_df = df[(df['cooperative'] == True) & (df['objective'] == 'MAXMIN')].copy()

    # Create a unique key for merging
    maxprof_df['merge_key'] = maxprof_df.groupby(['instance_index', 'pricing_mechanism', 'terminal_id']).ngroup()
    maxmin_df['merge_key'] = maxmin_df.groupby(['instance_index', 'pricing_mechanism', 'terminal_id']).ngroup()

    comparison_df = pd.merge(maxprof_df, maxmin_df, on='merge_key', suffixes=('_MAXPROF', '_MAXMIN'))

    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=comparison_df, x='profit_after_MAXPROF', y='profit_after_MAXMIN', hue='pricing_mechanism')
    plt.plot([comparison_df['profit_after_MAXPROF'].min(), comparison_df['profit_after_MAXPROF'].max()],
             [comparison_df['profit_after_MAXPROF'].min(), comparison_df['profit_after_MAXPROF'].max()], 'r--')
    plt.title('Profit After Cooperation: MAXPROF vs. MAXMIN')
    plt.xlabel('Profit with MAXPROF Objective ($)')
    plt.ylabel('Profit with MAXMIN Objective ($)')
    plt.savefig(os.path.join(PLOT_FOLDER, 'profit_comparison_scatter.png'))
    plt.close()


# --- Main Execution ---

if __name__ == '__main__':
    print("--- Starting Results Analysis ---")

    # 1. Load data
    results_data = load_all_results(RESULTS_FOLDER)
    if not results_data:
        print(f"No .pkl files found in '{RESULTS_FOLDER}'. Please run the simulation first.")
    else:
        # 2. Consolidate into DataFrame
        df_results = consolidate_data(results_data)

        if not df_results.empty:
            # 3. Perform analysis and plotting
            with pd.ExcelWriter(EXCEL_OUTPUT_FILE) as writer:
                # Save the raw data to a sheet
                df_results.to_excel(writer, sheet_name='RawData', index=False)

                # Perform analysis and save to other sheets
                analyze_and_plot(df_results, writer)

            print(f"\nAnalysis complete! Results saved to '{EXCEL_OUTPUT_FILE}' and plots to '{PLOT_FOLDER}/'.")
        else:
            print("No valid data found for analysis.")