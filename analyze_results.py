import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Configuration ---
RESULTS_FOLDER = "simulation_results_gurobi_unified"
OUTPUT_FOLDER = "analysis_output"
SUMMARY_CSV_FILE = os.path.join(OUTPUT_FOLDER, "aggregated_metrics_summary.csv")
TERMINAL_TYPES = ['Underproductive', 'Productive', 'Overproductive']


def debug_pickle_structure(folder: str, max_files: int = 3):
    """Debug function to inspect the structure of pickle files."""
    print("=== DEBUGGING PICKLE FILE STRUCTURE ===")

    files_checked = 0
    for filename in os.listdir(folder):
        if filename.endswith(".pkl") and 'UNIFIED' in filename and files_checked < max_files:
            filepath = os.path.join(folder, filename)
            print(f"\n--- File: {filename} ---")

            try:
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)

                print("Top-level keys:")
                for key in results.keys():
                    print(f"  - {key}: {type(results[key])}")

                if 'optimization_output' in results:
                    print("\nOptimization output keys:")
                    for key in results['optimization_output'].keys():
                        print(f"  - {key}: {type(results['optimization_output'][key])}")

                if 'simulation_params' in results:
                    print("\nSimulation params keys:")
                    for key in results['simulation_params'].keys():
                        print(f"  - {key}: {type(results['simulation_params'][key])}")

                files_checked += 1

            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print("=== END DEBUG ===\n")


# --- Utility Functions ---

def calculate_gini(array: np.ndarray) -> float:
    """Calculates the Gini coefficient for profit distribution."""
    array = array.flatten()
    if np.min(array) < 0:
        array = array - np.min(array) + 1e-6

    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)

    if np.sum(array) == 0:
        return 0.0
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_cv(array: np.ndarray) -> float:
    """Calculates the Coefficient of Variation (CV) for profit distribution."""
    if np.mean(array) == 0 or np.isnan(np.mean(array)):
        return np.nan
    return np.std(array) / np.mean(array)


def map_subset_to_terminal_types(subset_str: str) -> Dict[int, str]:
    """Maps terminal index to type based on the subset composition string (e.g., '1_1_1')."""
    parts = [int(x) for x in subset_str.split('_')]
    mapping = {}
    idx = 0

    # Premium (Underproductive)
    for _ in range(parts[0]):
        mapping[idx] = 'Underproductive'
        idx += 1
    # Balanced (Productive)
    for _ in range(parts[1]):
        mapping[idx] = 'Productive'
        idx += 1
    # High-Volume (Overproductive)
    for _ in range(parts[2]):
        mapping[idx] = 'Overproductive'
        idx += 1

    return mapping


def analyze_results_folder(folder: str) -> Dict[str, Any]:
    """Analyzes the contents of the results folder and provides detailed reporting."""
    if not os.path.exists(folder):
        return {
            'folder_exists': False,
            'total_files': 0,
            'pkl_files': 0,
            'unified_files': 0,
            'file_list': []
        }

    all_files = os.listdir(folder)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    unified_files = [f for f in pkl_files if 'UNIFIED' in f]

    print(f"=== RESULTS FOLDER ANALYSIS ===")
    print(f"Folder: {folder}")
    print(f"Total files: {len(all_files)}")
    print(f"Pickle files (.pkl): {len(pkl_files)}")
    print(f"UNIFIED pickle files: {len(unified_files)}")

    if len(all_files) > 0:
        print(f"\nAll files in folder:")
        for i, file in enumerate(all_files, 1):
            print(f"  {i:2d}. {file}")

    if len(unified_files) > 0:
        print(f"\nUNIFIED files that will be processed:")
        for i, file in enumerate(unified_files, 1):
            print(f"  {i:2d}. {file}")
    else:
        print(f"\nNo UNIFIED files found!")
        if len(pkl_files) > 0:
            print("Available pickle files (none contain 'UNIFIED'):")
            for i, file in enumerate(pkl_files, 1):
                print(f"  {i:2d}. {file}")

    print("=" * 35)

    return {
        'folder_exists': True,
        'total_files': len(all_files),
        'pkl_files': len(pkl_files),
        'unified_files': len(unified_files),
        'file_list': all_files,
        'pkl_file_list': pkl_files,
        'unified_file_list': unified_files
    }


def load_all_results(folder: str) -> pd.DataFrame:
    """Loads all pickle files from the results folder and aggregates the data."""
    all_data = []

    for filename in os.listdir(folder):
        if filename.endswith(".pkl") and 'UNIFIED' in filename:
            filepath = os.path.join(folder, filename)

            try:
                # Load the pickle file
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)

                # Check if required keys exist
                if 'simulation_params' not in results or 'optimization_output' not in results:
                    print(f"Skipping {filename}: Missing required keys")
                    continue

                sim_params = results['simulation_params']
                opt_output = results['optimization_output']

                if 'Feasible' not in opt_output.get('feasibility_status', ''):
                    continue  # Skip infeasible results

                # Determine terminal type mapping for this scenario
                terminal_type_map = map_subset_to_terminal_types(sim_params['subset_composition'])

                for obj_name in ['MAXPROF', 'MAXMIN']:
                    if f'profitAfter_{obj_name}' not in opt_output:
                        continue  # Skip if objective was not solved (e.g., MAXMIN in Pyomo run)

                    profit_before = opt_output['profitBefore']
                    profit_after = opt_output[f'profitAfter_{obj_name}']
                    volume_after = opt_output[f'volumeAfter_{obj_name}']
                    transfer_fees = opt_output[f'transferFees_{obj_name}']

                    # --- Calculate System Metrics ---
                    total_profit_before = np.sum(profit_before)
                    total_profit_after = np.sum(profit_after)

                    # Calculate system efficiency gain safely
                    if total_profit_before != 0:
                        system_efficiency_gain = (total_profit_after - total_profit_before) / total_profit_before * 100
                    else:
                        system_efficiency_gain = 0

                    # Calculate equity metrics
                    profit_gini = calculate_gini(profit_after)

                    # --- Granular Terminal Data ---
                    num_terminals = sim_params.get('num_terminals_from_data', 0)
                    if num_terminals == 0: continue

                    for i in range(num_terminals):
                        terminal_id = i + 1

                        # Estimate if terminal is CI-capable based on the combo string
                        ci_terminals_list = [int(t) for t in sim_params['ci_terminals'].split('_') if t != 'None']
                        is_ci_capable = terminal_id in ci_terminals_list

                        # Calculate utilization
                        utilization_after = np.nan  # Default to NaN

                        if 'inputData' in results:
                            try:
                                capacity = results['inputData'][0, i, 0]
                                if capacity > 0 and len(volume_after) > i:
                                    utilization_after = volume_after[i] / capacity
                            except (IndexError, TypeError):
                                pass

                        # Calculate profit change percentage safely
                        # NOTE: The problematic percentage calculation will be FIXED in the post_process_data function.
                        profit_change_pct = np.nan
                        if profit_before[i] != 0:
                            profit_change_pct = (profit_after[i] - profit_before[i]) / profit_before[i] * 100
                        elif profit_after[i] != 0:  # Case where profitBefore=0 but profitAfter!=0
                            profit_change_pct = 100.0 * np.sign(profit_after[i])

                        data_row = {
                            'Filename': sim_params['filename'],
                            'Instance_Index': sim_params['instance_index'],
                            'Num_Terminals': num_terminals,
                            'Subset_Composition': sim_params['subset_composition'],
                            'Terminal_ID': terminal_id,
                            'Terminal_Type': terminal_type_map.get(i, 'Unknown'),
                            'is_CI_Capable': is_ci_capable,
                            'Subsidy_Level': sim_params['subsidy'],
                            'Pricing_Mechanism': sim_params['pricing_mechanism'],
                            'Objective': obj_name,

                            # --- Economic Metrics ---
                            'Profit_Before': profit_before[i],
                            'Profit_After': profit_after[i],
                            'Profit_Change_Abs': profit_after[i] - profit_before[i],
                            'Profit_Change_Pct': profit_change_pct,
                            # Initial (potentially negative for loss->gain) value
                            'Transfer_Fee': transfer_fees[i],

                            # --- Operational Metrics ---
                            'Volume_After': volume_after[i],
                            'Utilization_After': utilization_after,

                            # --- System/Equity Metrics ---
                            'Total_Profit_After_System': total_profit_after,
                            'System_Efficiency_Gain_Pct': system_efficiency_gain,
                            'Gini_Coefficient': profit_gini,
                            'CV_Profit': calculate_cv(profit_after)
                        }
                        all_data.append(data_row)

            except Exception as e:
                print(f"Critical Error processing {filename}: {e}")
                continue  # Skip to the next file

    return pd.DataFrame(all_data)


def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Profit_Change_Pct fix (for Profit_Before < 0)
    and calculates the Normalized_Change_Ratio KPI.
    """

    # 1. FIX Profit_Change_Pct

    # Create mask for negative baseline (where fix is needed)
    mask_negative_before = df['Profit_Before'] < 0

    # Apply fix: Profit_Change_Abs / |Profit_Before| * 100 for negative baseline
    # This overwrites the problematic values in the existing 'Profit_Change_Pct' column
    df.loc[mask_negative_before, 'Profit_Change_Pct'] = (
            df['Profit_Change_Abs'] / np.abs(df['Profit_Before']) * 100
    )

    # Handle Profit_Before = 0 case: re-apply robust calculation for 0 baseline
    mask_zero_before = (df['Profit_Before'] == 0) & (df['Profit_After'] != 0)
    # Use a large number to represent infinite growth
    df.loc[mask_zero_before, 'Profit_Change_Pct'] = np.sign(df['Profit_After']) * 1000000.0

    # 2. Calculate Normalized Profit Change Ratio (0 to 1) - New KPI

    # Define scenario grouping columns
    scenario_cols = ['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism',
                     'Objective']

    # Calculate Max Absolute Change within each scenario
    max_abs_change_per_scenario = df.groupby(scenario_cols)['Profit_Change_Abs'].transform(lambda x: np.abs(x).max())

    # Calculate Normalized Ratio
    df['Normalized_Change_Ratio'] = np.abs(df['Profit_Change_Abs']) / max_abs_change_per_scenario

    return df


def aggregate_and_prepare_tables(df: pd.DataFrame):
    """Aggregates data for the final tables and exports to Excel/CSV."""

    # Ensure terminal types are categorized correctly
    df['Terminal_Type'] = df['Terminal_Type'].astype('category').cat.set_categories(TERMINAL_TYPES)

    # 1. System Performance and Efficiency (Table 1 type)
    system_summary = df.groupby(['Objective', 'Subsidy_Level']) \
        .agg({
        'Total_Profit_After_System': [
            lambda x: x.iloc[0] / 1e6,
            lambda x: x.mean() / 1e6
        ],
        'System_Efficiency_Gain_Pct': 'mean',
        'Gini_Coefficient': 'mean'
    }) \
        .reset_index()

    # Flatten column names
    system_summary.columns = ['Objective', 'Subsidy_Level', 'Total_Profit_Before_M',
                              'Total_Profit_After_M', 'Avg_System_Efficiency_Gain', 'Avg_Gini']

    # 2. Terminal-Level Profit Distribution (Table 2 type) - UPDATED with new KPI
    profit_by_type = df.groupby(['Objective', 'Subsidy_Level', 'Terminal_Type'], observed=False) \
        .agg({
        'Profit_Change_Pct': 'mean',
        'Normalized_Change_Ratio': 'mean'  # ADDED
    }) \
        .reset_index()

    profit_by_type.columns = ['Objective', 'Subsidy_Level', 'Terminal_Type', 'Avg_Profit_Change_Pct_Fixed',
                              'Avg_Normalized_Change_Ratio']  # UPDATED

    # 3. CI Capability Impact (Table 3 type) - UPDATED with new KPI
    ci_impact = df.groupby(['Terminal_Type', 'is_CI_Capable', 'Objective'], observed=False) \
        .agg({
        'Profit_Change_Abs': 'mean',
        'Utilization_After': 'mean',
        'Normalized_Change_Ratio': 'mean'  # ADDED
    }) \
        .reset_index()

    ci_impact.columns = ['Terminal_Type', 'is_CI_Capable', 'Objective', 'Avg_Profit_Change_Abs', 'Avg_Utilization',
                         'Avg_Normalized_Change_Ratio']  # UPDATED

    return system_summary, profit_by_type, ci_impact


def generate_graphs(df: pd.DataFrame):
    """Generates key graphs for the paper, including the new Normalized Ratio plot."""

    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Efficiency-Fairness Trade-off (Gini vs. Subsidy) (Original Plot)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Subsidy_Level', y='Gini_Coefficient', hue='Objective',
                 errorbar='sd', marker='o', palette={'MAXPROF': 'red', 'MAXMIN': 'blue'})
    plt.title('Gini Coefficient by Objective and Subsidy Level (Fairness)')
    plt.xlabel('Subsidy Level ($/TEU)')
    plt.ylabel('Average Gini Coefficient')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_1_gini_vs_subsidy.png'))
    plt.close()

    # 2. Terminal Profit Change by Type (Table 2 data visualization) (Original Plot)
    df_agg = df.groupby(['Objective', 'Terminal_Type'], observed=False)['Profit_Change_Pct'].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_agg, x='Terminal_Type', y='Profit_Change_Pct', hue='Objective',
                palette={'MAXPROF': 'lightcoral', 'MAXMIN': 'cornflowerblue'})
    plt.title('Average Profit Change (%) by Terminal Type and Objective (Aggregated)')
    plt.xlabel('Terminal Type')
    plt.ylabel('Average Profit Change (%) (Fixed for loss->gain)')
    plt.axhline(0, color='black', linestyle='-')
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Objective')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_2_profit_change_by_type.png'))
    plt.close()

    # 3. CI Capability Profit Premium (Original Plot)
    df_ci = df.groupby(['Terminal_Type', 'is_CI_Capable'], observed=False)['Profit_Change_Abs'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_ci, x='Terminal_Type', y='Profit_Change_Abs', hue='is_CI_Capable',
                palette={True: 'green', False: 'gray'})
    plt.title('Profit Premium by CI Capability and Terminal Type (Absolute Change)')
    plt.xlabel('Terminal Type')
    plt.ylabel('Average Absolute Profit Change ($)')
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='CI Capable', labels=['Capable', 'Not Capable'])
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_3_CI_profit_premium.png'))
    plt.close()

    # 4. Normalized Profit Change Ratio (New Plot)
    df_norm_agg = df.groupby(['Objective', 'Terminal_Type'], observed=False)[
        'Normalized_Change_Ratio'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_norm_agg, x='Terminal_Type', y='Normalized_Change_Ratio', hue='Objective',
                palette={'MAXPROF': 'lightcoral', 'MAXMIN': 'cornflowerblue'})
    plt.title('Average Normalized Profit Change Ratio by Terminal Type and Objective')
    plt.xlabel('Terminal Type')
    plt.ylabel('Average Normalized Profit Change Ratio (0-1)')
    plt.ylim(0, df_norm_agg['Normalized_Change_Ratio'].max() * 1.05)
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Objective')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_4_normalized_change_ratio.png'))
    plt.close()

    print(f"\nGraphs generated and saved to {OUTPUT_FOLDER}. (Including the new figure_4)")


def main():
    """Main execution function."""
    print("--- Starting Post-Optimization Analysis ---")

    # 1. Load Data
    df_raw = load_all_results(RESULTS_FOLDER)

    if df_raw.empty:
        print("\n❌ ANALYSIS FAILED: No data could be loaded.")
        return

    print(f"\n✅ DATA LOADING COMPLETE")
    print(f"Successfully loaded {len(df_raw)} terminal records from pickle files.")

    # 2. Post-Process Data: Fix % calculation and calculate new KPI
    print(f"\n=== POST-PROCESSING DATA (Fixing Pct, Calculating Normalized KPI) ===")
    df_processed = post_process_data(df_raw)

    # 3. Aggregate Data and Prepare Tables
    print(f"\n=== AGGREGATING DATA ===")
    system_summary, profit_by_type, ci_impact = aggregate_and_prepare_tables(df_processed)

    # 4. Export to Files
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        with pd.ExcelWriter(os.path.join(OUTPUT_FOLDER, "analysis_tables.xlsx")) as writer:
            system_summary.to_excel(writer, sheet_name='System_Performance', index=False)
            profit_by_type.to_excel(writer, sheet_name='Profit_by_Type', index=False)
            ci_impact.to_excel(writer, sheet_name='CI_Impact', index=False)

        # Exporting the fully processed DF with the fixed percentage and new KPI
        df_processed.to_csv(SUMMARY_CSV_FILE, index=False)

        print(f"\n✅ EXPORT COMPLETE")
        print(f"[SUCCESS] Aggregated results exported to {OUTPUT_FOLDER}/analysis_tables.xlsx (3 sheets).")
        print(f"[SUCCESS] Raw aggregated CSV saved to {SUMMARY_CSV_FILE} (includes fixed Pct and Normalized KPI).")

        # 5. Generate Visualizations
        print(f"\n=== GENERATING VISUALIZATIONS ===")
        generate_graphs(df_processed)

        print(f"\n🎉 ANALYSIS COMPLETE! Check the '{OUTPUT_FOLDER}' folder for results.")

    except Exception as e:
        print(f"\n❌ EXPORT ERROR: {e}")
        print("Data was loaded successfully but could not be exported.")


if __name__ == '__main__':
    main()