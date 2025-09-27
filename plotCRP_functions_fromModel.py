"""
Terminal Economics Function Plotting
Plots cost, revenue, and profit functions used in the cooperation optimization model
Uses the exact same functions as implemented in the Gurobi optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os # Necessary for path creation and file listing
import pickle # Necessary for loading data

# Define the target subdirectory for saving plots
PLOTS_SUBFOLDER = "plotsTestModelFunctions"

def calculate_piecewise_cost(utilization, mc_start, slope1, slope2, u_optimal, capacity):
    """
    Calculate total cost using the original piecewise quadratic function
    """
    u = np.asarray(utilization)
    costs = np.zeros_like(u, dtype=float)

    # Phase 1: u <= u_optimal (decreasing marginal cost)
    mask1 = (u >= 0) & (u <= u_optimal)
    costs[mask1] = (mc_start * u[mask1] - 0.5 * slope1 * u[mask1] ** 2)

    # Phase 2: u > u_optimal (increasing marginal cost)
    mask2 = u > u_optimal
    if np.any(mask2):
        cost_at_optimal = (mc_start * u_optimal - 0.5 * slope1 * u_optimal ** 2)
        mc_at_optimal = mc_start - slope1 * u_optimal

        costs[mask2] = (cost_at_optimal +
                       mc_at_optimal * (u[mask2] - u_optimal) +
                       0.5 * slope2 * (u[mask2] - u_optimal) ** 2)

    # Scale by capacity to get total cost
    total_costs = costs * capacity
    return total_costs

def calculate_marginal_cost(utilization, mc_start, slope1, slope2, u_optimal):
    """
    Calculate marginal cost at given utilization level
    """
    u = np.asarray(utilization)
    mc = np.zeros_like(u, dtype=float)

    # Phase 1: u <= u_optimal
    mask1 = u <= u_optimal
    mc[mask1] = mc_start - slope1 * u[mask1]

    # Phase 2: u > u_optimal
    mask2 = u > u_optimal
    mc[mask2] = (mc_start - slope1 * u_optimal + slope2 * (u[mask2] - u_optimal))

    return mc

def calculate_revenue(utilization, volume, revenue_decrease_rate, revenue_initial_charge, ci_subsidy=0):
    """
    Calculate total revenue (FIXED at baseline levels during cooperation)
    """
    revenue_per_teu = revenue_decrease_rate * utilization + revenue_initial_charge
    total_revenue = revenue_per_teu * volume + ci_subsidy
    return total_revenue

def calculate_profit(utilization, mc_start, slope1, slope2, u_optimal, capacity,
                    revenue_decrease_rate, revenue_initial_charge, ci_subsidy=0):
    """
    Calculate profit = revenue - cost
    """
    volume = utilization * capacity
    revenue = calculate_revenue(utilization, volume, revenue_decrease_rate, revenue_initial_charge, ci_subsidy)
    cost = calculate_piecewise_cost(utilization, mc_start, slope1, slope2, u_optimal, capacity)
    return revenue - cost

def plot_terminal_economics(terminal_params, save_plots=True, show_plots=False):
    """
    Plot cost, revenue, and profit functions for given terminal parameters
    """

    # Ensure the plots subfolder exists
    if save_plots and not os.path.exists(PLOTS_SUBFOLDER):
        os.makedirs(PLOTS_SUBFOLDER)

    # Extract parameters
    mc_start = terminal_params['mc_start']
    slope1 = terminal_params['slope1']
    slope2 = terminal_params['slope2']
    u_optimal = terminal_params['u_optimal']
    capacity = terminal_params['capacity']
    revenue_decrease_rate = terminal_params['revenue_decrease_rate']
    revenue_initial_charge = terminal_params['revenue_initial_charge']
    ci_subsidy = terminal_params.get('ci_subsidy', 0)
    terminal_name = terminal_params.get('name', 'Terminal')

    # Create utilization range
    u_range = np.linspace(0.01, 0.99, 200)

    # Calculate functions
    volumes = u_range * capacity
    total_costs = calculate_piecewise_cost(u_range, mc_start, slope1, slope2, u_optimal, capacity)
    total_revenues = calculate_revenue(u_range, volumes, revenue_decrease_rate, revenue_initial_charge, ci_subsidy)
    total_profits = total_revenues - total_costs
    marginal_costs = calculate_marginal_cost(u_range, mc_start, slope1, slope2, u_optimal)

    # Revenue per TEU
    revenue_per_teu = revenue_decrease_rate * u_range + revenue_initial_charge

    # Average cost per TEU
    average_cost_per_teu = total_costs / volumes

    # Marginal revenue (derivative of revenue per TEU * volume)
    marginal_revenue = 2 * revenue_decrease_rate * u_range + revenue_initial_charge

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{terminal_name} - Economic Functions Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Total Cost Function
    axes[0, 0].plot(u_range, total_costs / 1e6, 'b-', linewidth=2, label='Total Cost')
    axes[0, 0].axvline(x=u_optimal, color='r', linestyle='--', alpha=0.7, label=f'Optimal U = {u_optimal:.2f}')
    axes[0, 0].set_xlabel('Utilization')
    axes[0, 0].set_ylabel('Total Cost ($ Million)')
    axes[0, 0].set_title('Total Cost Function (Piecewise Quadratic)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Total Revenue Function
    axes[0, 1].plot(u_range, total_revenues / 1e6, 'g-', linewidth=2, label='Total Revenue')
    if ci_subsidy > 0:
        base_revenue = calculate_revenue(u_range, volumes, revenue_decrease_rate, revenue_initial_charge, 0)
        axes[0, 1].plot(u_range, base_revenue / 1e6, 'g--', alpha=0.7, label='Revenue (excl. CI subsidy)')
    axes[0, 1].set_xlabel('Utilization')
    axes[0, 1].set_ylabel('Total Revenue ($ Million)')
    axes[0, 1].set_title('Total Revenue Function (Fixed During Cooperation)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Total Profit Function
    axes[0, 2].plot(u_range, total_profits / 1e6, 'purple', linewidth=2, label='Total Profit')
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axvline(x=u_optimal, color='r', linestyle='--', alpha=0.7, label=f'Optimal U = {u_optimal:.2f}')

    # Find profit-maximizing utilization
    max_profit_idx = np.argmax(total_profits)
    max_profit_u = u_range[max_profit_idx]
    max_profit_value = total_profits[max_profit_idx]
    axes[0, 2].plot(max_profit_u, max_profit_value / 1e6, 'ro', markersize=8, label=f'Max Profit U = {max_profit_u:.2f}')

    axes[0, 2].set_xlabel('Utilization')
    axes[0, 2].set_ylabel('Total Profit ($ Million)')
    axes[0, 2].set_title('Total Profit Function')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # Plot 4: Marginal Cost
    axes[1, 0].plot(u_range, marginal_costs, 'b-', linewidth=2, label='Marginal Cost')
    axes[1, 0].axvline(x=u_optimal, color='r', linestyle='--', alpha=0.7, label=f'Optimal U = {u_optimal:.2f}')
    axes[1, 0].set_xlabel('Utilization')
    axes[1, 0].set_ylabel('Marginal Cost ($/TEU)')
    axes[1, 0].set_title('Marginal Cost Function')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 5: Revenue per TEU and Average Cost per TEU
    axes[1, 1].plot(u_range, revenue_per_teu, 'g-', linewidth=2, label='Revenue per TEU')
    axes[1, 1].plot(u_range, average_cost_per_teu, 'b-', linewidth=2, label='Average Cost per TEU')
    axes[1, 1].set_xlabel('Utilization')
    axes[1, 1].set_ylabel('$/TEU')
    axes[1, 1].set_title('Revenue and Average Cost per TEU')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Plot 6: Marginal Revenue vs Marginal Cost
    axes[1, 2].plot(u_range, marginal_revenue, 'g-', linewidth=2, label='Marginal Revenue')
    axes[1, 2].plot(u_range, marginal_costs, 'b-', linewidth=2, label='Marginal Cost')
    axes[1, 2].axvline(x=u_optimal, color='r', linestyle='--', alpha=0.7, label=f'Cost Optimal U = {u_optimal:.2f}')

    # Find MR = MC intersection
    mr_mc_diff = np.abs(marginal_revenue - marginal_costs)
    mr_mc_idx = np.argmin(mr_mc_diff)
    mr_mc_u = u_range[mr_mc_idx]
    axes[1, 2].plot(mr_mc_u, marginal_costs[mr_mc_idx], 'ro', markersize=8, label=f'MR=MC U = {mr_mc_u:.2f}')

    axes[1, 2].set_xlabel('Utilization')
    axes[1, 2].set_ylabel('$/TEU')
    axes[1, 2].set_title('Marginal Revenue vs Marginal Cost')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    plt.tight_layout()

    if save_plots:
        filename = f'{terminal_name.replace(" ", "_")}_economics_analysis.png'
        save_path = os.path.join(PLOTS_SUBFOLDER, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {save_path}")

    if show_plots:
        plt.show()

    # Print summary statistics
    print(f"\n{terminal_name} - Economic Analysis Summary:")
    print("=" * 60)
    print(f"Cost Function Parameters:")
    print(f"  - Initial MC: ${mc_start:.2f}/TEU")
    print(f"  - Slope 1 (decrease): ${slope1:.2f}")
    print(f"  - Slope 2 (increase): ${slope2:.2f}")
    print(f"  - Optimal Utilization: {u_optimal:.1%}")
    print(f"Revenue Function Parameters:")
    print(f"  - Initial Charge: ${revenue_initial_charge:.2f}/TEU")
    print(f"  - Decrease Rate: ${revenue_decrease_rate:.2f}/TEU per utilization")
    if ci_subsidy > 0:
        print(f"  - CI Subsidy: ${ci_subsidy:,.0f}")
    print(f"Terminal Capacity: {capacity:,.0f} TEU")
    print()
    print(f"Key Operating Points:")
    print(f"  - Cost Optimal Utilization: {u_optimal:.1%}")
    print(f"  - Profit Optimal Utilization: {max_profit_u:.1%}")
    print(f"  - MR = MC Utilization: {mr_mc_u:.1%}")
    print(f"  - Maximum Profit: ${max_profit_value:,.0f}")

    # Ensure plot object is closed
    plt.close(fig)

    return {
        'utilization_range': u_range,
        'total_costs': total_costs,
        'total_revenues': total_revenues,
        'total_profits': total_profits,
        'marginal_costs': marginal_costs,
        'marginal_revenue': marginal_revenue,
        'max_profit_utilization': max_profit_u,
        'mr_mc_utilization': mr_mc_u
    }

def compare_terminals(terminal_params_list, save_plots=True, show_plots=False):
    """
    Compare multiple terminals on the same plots
    """

    # Ensure the plots subfolder exists
    if save_plots and not os.path.exists(PLOTS_SUBFOLDER):
        os.makedirs(PLOTS_SUBFOLDER)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Terminal Comparison - Economic Functions', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    u_range = np.linspace(0.01, 0.99, 200)

    for i, params in enumerate(terminal_params_list):
        color = colors[i % len(colors)]
        name = params.get('name', f'Terminal {i+1}')

        # Extract parameters
        mc_start = params['mc_start']
        slope1 = params['slope1']
        slope2 = params['slope2']
        u_optimal = params['u_optimal']
        capacity = params['capacity']
        revenue_decrease_rate = params['revenue_decrease_rate']
        revenue_initial_charge = params['revenue_initial_charge']

        # Calculate functions
        volumes = u_range * capacity
        total_costs = calculate_piecewise_cost(u_range, mc_start, slope1, slope2, u_optimal, capacity)
        total_revenues = calculate_revenue(u_range, volumes, revenue_decrease_rate, revenue_initial_charge)
        total_profits = total_revenues - total_costs
        marginal_costs = calculate_marginal_cost(u_range, mc_start, slope1, slope2, u_optimal)
        marginal_revenue = 2 * revenue_decrease_rate * u_range + revenue_initial_charge

        # Plot comparisons
        axes[0, 0].plot(u_range, total_costs / 1e6, color=color, linewidth=2, label=name)
        axes[0, 1].plot(u_range, total_revenues / 1e6, color=color, linewidth=2, label=name)
        axes[1, 0].plot(u_range, total_profits / 1e6, color=color, linewidth=2, label=name)
        axes[1, 1].plot(u_range, marginal_costs, color=color, linewidth=2, label=f'{name} MC')
        axes[1, 1].plot(u_range, marginal_revenue, color=color, linewidth=2, linestyle='--', alpha=0.7) # Exclude MR from legend to avoid clutter

    # Configure subplots
    axes[0, 0].set_title('Total Cost Comparison')
    axes[0, 0].set_xlabel('Utilization')
    axes[0, 0].set_ylabel('Total Cost ($ Million)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title('Total Revenue Comparison')
    axes[0, 1].set_xlabel('Utilization')
    axes[0, 1].set_ylabel('Total Revenue ($ Million)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title('Total Profit Comparison')
    axes[1, 0].set_xlabel('Utilization')
    axes[1, 0].set_ylabel('Total Profit ($ Million)')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_title('Marginal Cost vs Marginal Revenue')
    axes[1, 1].set_xlabel('Utilization')
    axes[1, 1].set_ylabel('$/TEU')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='upper left')

    plt.tight_layout()

    if save_plots:
        save_path = os.path.join(PLOTS_SUBFOLDER, 'terminal_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved as: {save_path}")

    if show_plots:
        plt.show()

    plt.close(fig)

# Example usage with REAL data from input files
if __name__ == "__main__":

    print("Terminal Economics Function Analysis")
    print("===================================")
    print(f"Plots will be saved in the subfolder: {PLOTS_SUBFOLDER}\n")

    # Load actual data from generated files
    data_folder = "generated_data"

    try:
        data_files = [f for f in os.listdir(data_folder) if f.endswith('.pkl')]
        if not data_files:
            raise FileNotFoundError("No .pkl files found in generated_data folder")

        # Load the first data file
        data_file = data_files[0]
        filepath = os.path.join(data_folder, data_file)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Loaded data from: {data_file}")

        # Extract parameters from inputData
        input_data = data['inputData'][0, :, :]
        num_terminals = input_data.shape[0]

        print(f"Number of terminals in data: {num_terminals}")

        # Create terminal parameter dictionaries using REAL data
        real_terminals = []
        for i in range(num_terminals):
            terminal_params = {
                'name': f'Terminal {i+1}',
                'mc_start': input_data[i, 1],
                'slope1': input_data[i, 2],
                'slope2': input_data[i, 3],
                'u_optimal': input_data[i, 4],
                'capacity': input_data[i, 0],
                'revenue_decrease_rate': -input_data[i, 6],
                'revenue_initial_charge': input_data[i, 5],
                'ci_subsidy': 0
            }
            real_terminals.append(terminal_params)

            # Print parameter summary
            print(f"\nTerminal {i+1} Parameters:")
            print(f"  Capacity: {terminal_params['capacity']:,.0f} TEU")
            print(f"  MC Start: ${terminal_params['mc_start']:.2f}")
            print(f"  Slope1: {terminal_params['slope1']:.2f}")
            print(f"  Slope2: {terminal_params['slope2']:.2f}")
            print(f"  U Optimal: {terminal_params['u_optimal']:.2%}")
            print(f"  Revenue Intercept: ${terminal_params['revenue_initial_charge']:.2f}")
            print(f"  Revenue Slope: ${terminal_params['revenue_decrease_rate']:.2f}")

        # Plot individual terminal analysis for each real terminal
        print(f"\nGenerating individual plots for {num_terminals} real terminals...")
        for i, terminal_params in enumerate(real_terminals):
            print(f"\nAnalyzing {terminal_params['name']}...")
            # Use show_plots=False to prevent hundreds of windows opening automatically
            plot_terminal_economics(terminal_params, save_plots=True, show_plots=False)

        # Plot comparison of all real terminals
        print("\nGenerating terminal comparison plots...")
        compare_terminals(real_terminals, save_plots=True, show_plots=False)

        print("\nAnalysis complete! 🎉")
        print(f"Plots generated using ACTUAL terminal parameters and saved in '{PLOTS_SUBFOLDER}'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'generated_data' folder exists and contains .pkl files.")
        print("Run the data generation script first.")

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("Falling back to sample data...")

        # Fallback to sample data if file loading fails
        sample_terminals = [
            {
                'name': 'Sample Premium Terminal',
                'mc_start': 250.0,
                'slope1': 100.0,
                'slope2': 800.0,
                'u_optimal': 0.50,
                'capacity': 120000,
                'revenue_decrease_rate': -50.0,
                'revenue_initial_charge': 400.0,
                'ci_subsidy': 50000
            }
        ]

        # Use show_plots=True for the sample data so the user sees a result
        for terminal_params in sample_terminals:
            plot_terminal_economics(terminal_params, save_plots=False, show_plots=True)