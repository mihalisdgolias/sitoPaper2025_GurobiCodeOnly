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


# --- Utility Functions ---

def calculate_gini(array: np.ndarray) -> float:
    """Calculates the Gini coefficient for profit distribution."""
    array = array.flatten()
    if np.min(array) < 0:
        # Shift values to be non-negative for Gini calculation
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


def calculate_hhi(capacities: np.ndarray) -> float:
    """
    Calculates the Herfindahl-Hirschman Index (HHI) for market concentration
    based on terminal capacities. (Prompt 3.2)
    """
    total_capacity = np.sum(capacities)
    if total_capacity == 0:
        return np.nan

    # Calculate market share percentage
    shares = (capacities / total_capacity) * 100
    # HHI is the sum of the squares of the market shares
    hhi = np.sum(shares ** 2)
    return hhi


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

                # Get terminal capacity and original volume data
                terminal_capacities = np.array([])
                terminal_volume_before = np.array([])
                if 'inputData' in results:
                    try:
                        # Assuming capacity is at [0, i, 0] and Volume Before is at [0, i, 1]
                        terminal_capacities = results['inputData'][0, :, 0]
                        terminal_volume_before = results['inputData'][0, :, 1]
                    except (IndexError, TypeError):
                        pass

                # Calculate HHI (Prompt 3.2)
                hhi = calculate_hhi(terminal_capacities)

                # Determine terminal type mapping for this scenario
                terminal_type_map = map_subset_to_terminal_types(sim_params['subset_composition'])

                # Calculate CI Density for the scenario (Prompt 1.1)
                num_terminals = sim_params.get('num_terminals_from_data', 0)
                ci_terminals_str = sim_params.get('ci_terminals', 'None')
                ci_terminals_count = len([t for t in ci_terminals_str.split('_') if t != 'None'])

                ci_density = 0.0
                if num_terminals > 0:
                    ci_density = (ci_terminals_count / num_terminals) * 100  # Percentage

                # Total Subsidy for the scenario
                subsidy_level = sim_params['subsidy']
                total_subsidy_spent = subsidy_level * ci_terminals_count

                for obj_name in ['MAXPROF', 'MAXMIN']:
                    if f'profitAfter_{obj_name}' not in opt_output:
                        continue  # Skip if objective was not solved

                    profit_before = opt_output['profitBefore']
                    profit_after = opt_output[f'profitAfter_{obj_name}']
                    volume_after = opt_output[f'volumeAfter_{obj_name}']
                    transfer_fees = opt_output[f'transferFees_{obj_name}']

                    # --- Calculate System Metrics ---
                    total_profit_before = np.sum(profit_before)
                    total_profit_after = np.sum(profit_after)
                    profit_gain_abs = total_profit_after - total_profit_before

                    # Calculate system efficiency gain safely
                    system_efficiency_gain_pct = 0.0
                    if total_profit_before != 0:
                        system_efficiency_gain_pct = (profit_gain_abs / total_profit_before) * 100
                    elif profit_gain_abs > 0:
                        system_efficiency_gain_pct = 1000000.0

                    # Calculate Efficiency per Subsidy Dollar (Prompt 2.1)
                    efficiency_per_subsidy_dollar = 0.0
                    if total_subsidy_spent > 0:
                        efficiency_per_subsidy_dollar = profit_gain_abs / total_subsidy_spent

                    # Calculate equity metrics
                    profit_gini = calculate_gini(profit_after)

                    # --- Granular Terminal Data ---
                    if num_terminals == 0: continue

                    for i in range(num_terminals):
                        terminal_id = i + 1

                        # Estimate if terminal is CI-capable
                        is_ci_capable = terminal_id in [int(t) for t in sim_params['ci_terminals'].split('_') if
                                                        t != 'None']

                        # Volume Metrics
                        vol_before = terminal_volume_before[i] if len(terminal_volume_before) > i else np.nan
                        vol_after = volume_after[i] if len(volume_after) > i else np.nan
                        vol_change_abs = vol_after - vol_before

                        # Calculate utilization
                        utilization_after = np.nan

                        if len(terminal_capacities) > i and terminal_capacities[i] > 0 and len(volume_after) > i:
                            utilization_after = vol_after / terminal_capacities[i]

                        # Calculate profit change percentage (FIXED in post-processing)
                        profit_change_pct = np.nan
                        if profit_before[i] != 0:
                            profit_change_pct = (profit_after[i] - profit_before[i]) / profit_before[i] * 100
                        elif profit_after[i] != 0:
                            profit_change_pct = 100.0 * np.sign(profit_after[i])

                        data_row = {
                            'Filename': sim_params['filename'],
                            'Instance_Index': sim_params['instance_index'],
                            'Num_Terminals': num_terminals,
                            'Subset_Composition': sim_params['subset_composition'],
                            'CI_Config_Str': ci_terminals_str,
                            'CI_Density_Pct': ci_density,
                            'Terminal_ID': terminal_id,
                            'Terminal_Type': terminal_type_map.get(i, 'Unknown'),
                            'is_CI_Capable': is_ci_capable,
                            'Subsidy_Level': subsidy_level,
                            'Pricing_Mechanism': sim_params['pricing_mechanism'],
                            'Objective': obj_name,

                            # --- Economic Metrics ---
                            'Profit_Before': profit_before[i],
                            'Profit_After': profit_after[i],
                            'Profit_Change_Abs': profit_after[i] - profit_before[i],
                            'Profit_Change_Pct': profit_change_pct,
                            'Transfer_Fee': transfer_fees[i],

                            # --- Operational Metrics ---
                            'Volume_Before': vol_before,
                            'Volume_After': vol_after,
                            'Volume_Change_Abs': vol_change_abs,
                            'Utilization_After': utilization_after,

                            # --- System/Equity Metrics ---
                            'Total_Profit_After_System': total_profit_after,
                            'System_Efficiency_Gain_Pct': system_efficiency_gain_pct,
                            'Efficiency_Per_Subsidy_Dollar': efficiency_per_subsidy_dollar,
                            'Gini_Coefficient': profit_gini,
                            'CV_Profit': calculate_cv(profit_after),
                            'HHI_Market_Concentration': hhi,
                        }
                        all_data.append(data_row)

            except Exception as e:
                print(f"Critical Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip to the next file

    df = pd.DataFrame(all_data)

    # Calculate System Shuffling Metric (sum of absolute terminal volume changes)
    # NOTE ON VOLUME SHUFFLING: This metric measures **Gross Volume Movement** (Total Absolute Change).
    # It sums the absolute volume change at every terminal (loss and gain), which effectively
    # double-counts the movement of a single TEU unit. This is useful for quantifying
    # the total magnitude of operational disruption/re-planning required across the system.
    scenario_cols = ['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism',
                     'Objective']

    # Sum of absolute volume changes (per scenario, including all terminals)
    total_volume_shuffling = df.groupby(scenario_cols)['Volume_Change_Abs'].transform(lambda x: np.sum(np.abs(x)))

    # Total volume handled (for normalization)
    total_volume_system = df.groupby(scenario_cols)['Volume_After'].transform(lambda x: np.sum(x))

    df['System_Volume_Shuffling_Abs'] = total_volume_shuffling  # Gross Volume Movement (Absolute)

    # Normalized Shuffling (Gross Volume Movement / Total System Volume)
    df['System_Volume_Shuffling_Pct'] = (total_volume_shuffling / total_volume_system) * 100

    return df


def post_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the Profit_Change_Pct fix (for Profit_Before < 0)
    and calculates the Normalized_Change_Ratio KPI.
    """

    # 1. FIX Profit_Change_Pct

    # Create mask for negative baseline (where fix is needed)
    mask_negative_before = df['Profit_Before'] < 0

    # Apply fix: Profit_Change_Abs / |Profit_Before| * 100 for negative baseline
    df.loc[mask_negative_before, 'Profit_Change_Pct'] = (
            df['Profit_Change_Abs'] / np.abs(df['Profit_Before']) * 100
    )

    # Handle Profit_Before = 0 case: re-apply robust calculation for 0 baseline
    mask_zero_before = (df['Profit_Before'] == 0) & (df['Profit_After'] != 0)
    df.loc[mask_zero_before, 'Profit_Change_Pct'] = np.sign(df['Profit_After']) * 1000000.0

    # 2. Calculate Normalized Profit Change Ratio (0 to 1)

    # Define scenario grouping columns
    scenario_cols = ['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism',
                     'Objective']

    # Calculate Max Absolute Change within each scenario
    max_abs_change_per_scenario = df.groupby(scenario_cols)['Profit_Change_Abs'].transform(lambda x: np.abs(x).max())

    # Calculate Normalized Ratio
    df['Normalized_Change_Ratio'] = np.abs(df['Profit_Change_Abs']) / max_abs_change_per_scenario

    return df


def calculate_marginal_gains(df_system: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates marginal gains for system metrics when increasing the subsidy level. (Prompt 2.3)
    Compares Subsidy 50 vs 0 and Subsidy 100 vs 50.
    """

    # Define a unique scenario key, excluding Subsidy_Level
    df_system['Scenario_Key'] = df_system['Instance_Index'].astype(str) + '_' + \
                                df_system['Subset_Composition'] + '_' + \
                                df_system['Pricing_Mechanism'] + '_' + \
                                df_system['Objective'] + '_' + \
                                df_system['CI_Config_Str']

    marginal_gains = []

    # Iterate over unique scenarios defined by the key
    for key, group in df_system.groupby('Scenario_Key'):

        # Sort by subsidy level
        group = group.sort_values('Subsidy_Level')

        # Use .get() for safer row access
        rows = {level: group[group['Subsidy_Level'] == level].iloc[0]
        if not group[group['Subsidy_Level'] == level].empty else None
                for level in [0, 50, 100]}

        row_0, row_50, row_100 = rows[0], rows[50], rows[100]

        if row_0 is not None and row_50 is not None:
            gain_50 = {
                'Scenario_Key': key,
                'Subsidy_Change': '0_to_50',
                'Objective': row_0['Objective'],
                'Pricing_Mechanism': row_0['Pricing_Mechanism'],
                'Marginal_Efficiency_Gain': row_50['System_Efficiency_Gain_Pct'] - row_0['System_Efficiency_Gain_Pct'],
                'Marginal_Gini_Reduction': row_0['Gini_Coefficient'] - row_50['Gini_Coefficient'],
                # Gini reduction is good
                'Marginal_Total_Profit_Gain': row_50['Total_Profit_After_System'] - row_0['Total_Profit_After_System'],
                'Marginal_Shuffling_Change': row_50['System_Volume_Shuffling_Abs'] - row_0[
                    'System_Volume_Shuffling_Abs'],
            }
            marginal_gains.append(gain_50)

        if row_50 is not None and row_100 is not None:
            gain_100 = {
                'Scenario_Key': key,
                'Subsidy_Change': '50_to_100',
                'Objective': row_50['Objective'],
                'Pricing_Mechanism': row_50['Pricing_Mechanism'],
                'Marginal_Efficiency_Gain': row_100['System_Efficiency_Gain_Pct'] - row_50[
                    'System_Efficiency_Gain_Pct'],
                'Marginal_Gini_Reduction': row_50['Gini_Coefficient'] - row_100['Gini_Coefficient'],
                # Gini reduction is good
                'Marginal_Total_Profit_Gain': row_100['Total_Profit_After_System'] - row_50[
                    'Total_Profit_After_System'],
                'Marginal_Shuffling_Change': row_100['System_Volume_Shuffling_Abs'] - row_50[
                    'System_Volume_Shuffling_Abs'],
            }
            marginal_gains.append(gain_100)

    return pd.DataFrame(marginal_gains)


def aggregate_and_prepare_tables(df: pd.DataFrame):
    """Aggregates data for the final tables and exports to Excel/CSV."""

    # Ensure terminal types are categorized correctly
    df['Terminal_Type'] = df['Terminal_Type'].astype('category').cat.set_categories(TERMINAL_TYPES)

    # Define unique scenario data (used for system-level metrics, HHI, and cross-instance analysis)
    system_metrics_cols = ['Total_Profit_After_System', 'System_Efficiency_Gain_Pct', 'Gini_Coefficient',
                           'Efficiency_Per_Subsidy_Dollar', 'HHI_Market_Concentration',
                           'Instance_Index', 'CI_Config_Str', 'System_Volume_Shuffling_Abs',
                           'System_Volume_Shuffling_Pct']

    df_system = df.drop_duplicates(
        subset=['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism', 'Objective'])

    # 1. System Performance and Efficiency (Table 1 type)
    system_summary = df_system.groupby(['Objective', 'Subsidy_Level']) \
        [system_metrics_cols] \
        .mean() \
        .reset_index()

    # Calculate average Transfer Fee and Total Profit Before/After (Millions) for the summary
    agg_system_summary = df.groupby(['Objective', 'Subsidy_Level']).agg(
        Total_Profit_Before_M=pd.NamedAgg(column='Profit_Before', aggfunc=lambda x: (x.sum() / 1e6)),
        Total_Profit_After_M=pd.NamedAgg(column='Total_Profit_After_System', aggfunc=lambda x: (x.iloc[0] / 1e6)),
        Avg_Transfer_Fee=pd.NamedAgg(column='Transfer_Fee', aggfunc='mean')
    )
    agg_system_summary = agg_system_summary.reset_index()

    # Drop individual instance columns before merge
    system_summary = system_summary.drop(
        columns=['Total_Profit_After_System', 'Instance_Index', 'CI_Config_Str', 'HHI_Market_Concentration'])

    system_summary = pd.merge(system_summary, agg_system_summary, on=['Objective', 'Subsidy_Level'])

    # Rename and Reorder columns for clarity
    system_summary.rename(columns={'System_Volume_Shuffling_Abs': 'Avg_Shuffling_Abs',
                                   'System_Volume_Shuffling_Pct': 'Avg_Shuffling_Pct'}, inplace=True)

    system_summary = system_summary[['Objective', 'Subsidy_Level', 'Total_Profit_Before_M', 'Total_Profit_After_M',
                                     'Avg_System_Efficiency_Gain', 'Avg_Gini', 'Avg_Efficiency_Per_Subsidy',
                                     'Avg_Transfer_Fee', 'Avg_Shuffling_Abs', 'Avg_Shuffling_Pct']]

    # 2. Terminal-Level Profit, Transfer, and Volume Redistribution (Table 2 type)
    profit_by_type = df.groupby(['Objective', 'Subsidy_Level', 'Terminal_Type'], observed=False) \
        .agg({
        'Profit_Change_Pct': 'mean',
        'Normalized_Change_Ratio': 'mean',
        'Transfer_Fee': 'mean',
        'Volume_Change_Abs': 'mean'
    }) \
        .reset_index()

    profit_by_type.columns = ['Objective', 'Subsidy_Level', 'Terminal_Type', 'Avg_Profit_Change_Pct_Fixed',
                              'Avg_Normalized_Change_Ratio', 'Avg_Transfer_Fee', 'Avg_Volume_Change_Abs']

    # 3. CI Capability Impact (Table 3 type)
    ci_impact = df.groupby(['Terminal_Type', 'is_CI_Capable', 'Objective'], observed=False) \
        .agg({
        'Profit_Change_Abs': 'mean',
        'Utilization_After': 'mean',
        'Normalized_Change_Ratio': 'mean',
        'Volume_Change_Abs': 'mean'
    }) \
        .reset_index()

    ci_impact.columns = ['Terminal_Type', 'is_CI_Capable', 'Objective', 'Avg_Profit_Change_Abs', 'Avg_Utilization',
                         'Avg_Normalized_Change_Ratio', 'Avg_Volume_Change_Abs']

    # 4. CI Density Response Curve Data (Table 4 type)
    ci_density_response = df_system.groupby(['CI_Density_Pct', 'Pricing_Mechanism', 'Objective']) \
        .agg(
        Avg_System_Efficiency_Gain=pd.NamedAgg(column='System_Efficiency_Gain_Pct', aggfunc='mean'),
        Avg_Gini=pd.NamedAgg(column='Gini_Coefficient', aggfunc='mean'),
        Avg_HHI=pd.NamedAgg(column='HHI_Market_Concentration', aggfunc='mean'),
        Avg_Shuffling_Pct=pd.NamedAgg(column='System_Volume_Shuffling_Pct', aggfunc='mean')
    ) \
        .reset_index()

    # 5. Marginal Subsidy Impact (NEW TABLE - Prompt 2.3)
    df_marginal = calculate_marginal_gains(df_system)

    marginal_summary = df_marginal.groupby(['Objective', 'Subsidy_Change']).mean().reset_index()
    marginal_summary = marginal_summary[
        ['Objective', 'Subsidy_Change', 'Marginal_Efficiency_Gain', 'Marginal_Gini_Reduction',
         'Marginal_Total_Profit_Gain', 'Marginal_Shuffling_Change']]

    # 6. CI Configuration Ranking (NEW TABLE - Prompt 1.2)
    ci_ranking = df_system.groupby(
        ['CI_Config_Str', 'Subset_Composition', 'Objective', 'Pricing_Mechanism', 'Subsidy_Level']).agg(
        Avg_System_Efficiency_Gain=pd.NamedAgg(column='System_Efficiency_Gain_Pct', aggfunc='mean'),
        Avg_Gini=pd.NamedAgg(column='Gini_Coefficient', aggfunc='mean'),
        Avg_Min_Profit=pd.NamedAgg(column='Total_Profit_After_System', aggfunc='min'),
        Avg_Total_Profit=pd.NamedAgg(column='Total_Profit_After_System', aggfunc='mean'),
        Avg_Shuffling_Abs=pd.NamedAgg(column='System_Volume_Shuffling_Abs', aggfunc='mean')
    ).reset_index()

    # Create rankings (lower rank is better)
    ci_ranking['Rank_MaxProf'] = \
    ci_ranking.groupby(['Subset_Composition', 'Pricing_Mechanism', 'Subsidy_Level', 'Objective'])[
        'Avg_Total_Profit'].rank(method='dense', ascending=False)
    ci_ranking['Rank_MaxMin'] = \
    ci_ranking.groupby(['Subset_Composition', 'Pricing_Mechanism', 'Subsidy_Level', 'Objective'])[
        'Avg_Min_Profit'].rank(method='dense', ascending=False)
    ci_ranking['Rank_Fairness'] = \
    ci_ranking.groupby(['Subset_Composition', 'Pricing_Mechanism', 'Subsidy_Level', 'Objective'])['Avg_Gini'].rank(
        method='dense', ascending=True)
    ci_ranking['Rank_MinShuffling'] = \
    ci_ranking.groupby(['Subset_Composition', 'Pricing_Mechanism', 'Subsidy_Level', 'Objective'])[
        'Avg_Shuffling_Abs'].rank(method='dense', ascending=True)

    ci_ranking = ci_ranking.sort_values(
        ['Subset_Composition', 'Pricing_Mechanism', 'Subsidy_Level', 'Objective', 'Rank_MaxProf']).reset_index(
        drop=True)

    # 7. Cross-Instance Consistency (NEW TABLE - Prompt 7.1)
    consistency_summary = df_system.groupby(['Subsidy_Level', 'Pricing_Mechanism', 'Objective']).agg(
        Count_Instances=pd.NamedAgg(column='Instance_Index', aggfunc='count'),
        Mean_Efficiency=pd.NamedAgg(column='System_Efficiency_Gain_Pct', aggfunc='mean'),
        CV_Efficiency=pd.NamedAgg(column='System_Efficiency_Gain_Pct', aggfunc=calculate_cv),
        Mean_Gini=pd.NamedAgg(column='Gini_Coefficient', aggfunc='mean'),
        CV_Gini=pd.NamedAgg(column='Gini_Coefficient', aggfunc=calculate_cv),
        Mean_Shuffling_Pct=pd.NamedAgg(column='System_Volume_Shuffling_Pct', aggfunc='mean'),
        CV_Shuffling_Pct=pd.NamedAgg(column='System_Volume_Shuffling_Pct', aggfunc=calculate_cv)
    ).reset_index()
    consistency_summary = consistency_summary[['Subsidy_Level', 'Pricing_Mechanism', 'Objective', 'Count_Instances',
                                               'Mean_Efficiency', 'CV_Efficiency', 'Mean_Gini', 'CV_Gini',
                                               'Mean_Shuffling_Pct', 'CV_Shuffling_Pct']]

    return system_summary, profit_by_type, ci_impact, ci_density_response, marginal_summary, ci_ranking, consistency_summary


def generate_graphs(df: pd.DataFrame):
    """Generates key graphs for the paper, including the new Normalized Ratio plot and CI Density plot."""

    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Efficiency-Fairness Trade-off (Gini vs. Subsidy)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Subsidy_Level', y='Gini_Coefficient', hue='Objective',
                 errorbar='sd', marker='o', palette={'MAXPROF': 'red', 'MAXMIN': 'blue'})
    plt.title('Gini Coefficient by Objective and Subsidy Level (Fairness)')
    plt.xlabel('Subsidy Level ($/TEU)')
    plt.ylabel('Average Gini Coefficient')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_1_gini_vs_subsidy.png'))
    plt.close()

    # 2. Terminal Profit Change by Type
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

    # 3. CI Capability Profit Premium
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

    # 4. Normalized Profit Change Ratio
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

    # 5. CI Density Response Curve
    df_ci_resp = df.drop_duplicates(
        subset=['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism', 'Objective']) \
        .groupby(['CI_Density_Pct', 'Pricing_Mechanism', 'Objective']) \
        ['System_Efficiency_Gain_Pct'] \
        .mean() \
        .reset_index()

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_ci_resp, x='CI_Density_Pct', y='System_Efficiency_Gain_Pct',
                 hue='Pricing_Mechanism', style='Objective', errorbar=None, marker='o')
    plt.title('System Efficiency Gain vs. CI Adoption Density (Prompt 1.1)')
    plt.xlabel('CI Density (% of Terminals with CI Capability)')
    plt.ylabel('Average System Efficiency Gain (%)')
    plt.grid(True, linestyle='--')
    plt.xticks(sorted(df_ci_resp['CI_Density_Pct'].unique()))
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_5_ci_density_response_curve.png'))
    plt.close()

    # 6. Volume Shuffling by Subsidy Level (NEW Plot)
    df_shuff = df.drop_duplicates(
        subset=['Filename', 'Instance_Index', 'Subset_Composition', 'Subsidy_Level', 'Pricing_Mechanism', 'Objective']) \
        .groupby(['Subsidy_Level', 'Pricing_Mechanism', 'Objective']) \
        ['System_Volume_Shuffling_Pct'] \
        .mean() \
        .reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_shuff, x='Subsidy_Level', y='System_Volume_Shuffling_Pct',
                 hue='Pricing_Mechanism', style='Objective', errorbar=None, marker='o')
    plt.title('System Volume Shuffling (%) by Subsidy Level (Gross Movement)')
    plt.xlabel('Subsidy Level ($/TEU)')
    plt.ylabel('Average Volume Shuffling (%)')
    plt.grid(True, linestyle='--')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'figure_6_volume_shuffling_vs_subsidy.png'))
    plt.close()

    print(f"\nGraphs generated and saved to {OUTPUT_FOLDER}. (Including figure_6 for Volume Shuffling)")


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
    print(f"\n=== AGGREGATING DATA (Including Marginal Subsidy, CI Ranking, HHI, Consistency, and Volume Metrics) ===")
    system_summary, profit_by_type, ci_impact, ci_density_response, marginal_summary, ci_ranking, consistency_summary = aggregate_and_prepare_tables(
        df_processed)

    # 4. Export to Files
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        with pd.ExcelWriter(os.path.join(OUTPUT_FOLDER, "analysis_tables.xlsx")) as writer:
            system_summary.to_excel(writer, sheet_name='System_Performance', index=False)
            profit_by_type.to_excel(writer, sheet_name='Terminal_Volume_Redistribution', index=False)
            ci_impact.to_excel(writer, sheet_name='CI_Impact_and_Volume', index=False)
            ci_density_response.to_excel(writer, sheet_name='CI_Density_Response', index=False)
            marginal_summary.to_excel(writer, sheet_name='Marginal_Subsidy_Impact', index=False)
            ci_ranking.to_excel(writer, sheet_name='CI_Configuration_Ranking', index=False)
            consistency_summary.to_excel(writer, sheet_name='Cross_Instance_Consistency', index=False)

        # Exporting the fully processed DF with the fixed percentage and new KPI
        df_processed.to_csv(SUMMARY_CSV_FILE, index=False)

        print(f"\n✅ EXPORT COMPLETE")
        print(
            f"[SUCCESS] Aggregated results exported to {OUTPUT_FOLDER}/analysis_tables.xlsx (7 sheets, now including Volume metrics).")
        print(f"[SUCCESS] Raw aggregated CSV saved to {SUMMARY_CSV_FILE}.")

        # 5. Generate Visualizations
        print(f"\n=== GENERATING VISUALIZATIONS ===")
        generate_graphs(df_processed)

        print(f"\n🎉 ANALYSIS COMPLETE! Check the '{OUTPUT_FOLDER}' folder for results.")

    except Exception as e:
        print(f"\n❌ EXPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Data was loaded successfully but could not be exported.")


if __name__ == '__main__':
    main()