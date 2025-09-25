import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
import pandas as pd
import os
import random

warnings.filterwarnings('ignore')

# =============================================================================
# MAIN PARAMETERS - MODIFY THESE TO CUSTOMIZE YOUR ANALYSIS
# =============================================================================

SAVE_PLOTS = True
SHOW_PLOTS = True
TERMINAL_CAPACITY_MIN = 25000000 / 52
TERMINAL_CAPACITY_MAX = 35000000 / 52
TERMINAL_CAPACITY_MIN_VC_RATIO = 0.20
FIXED_MC_START = 150.0
FIXED_MC_MIN = 80.0

# Define the target V/C ranges for the three groups of terminals
VC_RANGES = {
    'Premium': (0.50, 0.65),
    'Balanced': (0.65, 0.75),
    'High-Volume': (0.75, 0.85)
}
NUM_TERMINALS_PER_GROUP = 10
NUM_TOTAL_TERMINALS = NUM_TERMINALS_PER_GROUP * len(VC_RANGES)

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
    print("Gurobi detected - using advanced optimization")
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi not available")
    # Exit if Gurobi is not available, as it is the only method to be used.
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


def gurobi_optimization(target_u: float, capacity: float, name: str, i: int) -> OptimizedTerminalModel:
    if not GUROBI_AVAILABLE:
        print("    Gurobi not available. This is required for the chosen method.")
        exit()

    print(f"  Optimizing {name} for target V/C = {target_u:.1%} with capacity {capacity:,} TEU")

    try:
        m = gp.Model("terminal_opt")
        m.setParam('OutputFlag', 0)

        a = m.addVar(lb=-500.0, ub=-10.0, name="a")
        b = m.addVar(lb=50.0, ub=1000.0, name="b")
        slope1 = m.addVar(lb=10.0, ub=200.0, name="slope1")
        slope2 = m.addVar(lb=500.0, ub=3000.0, name="slope2")

        mc_start = FIXED_MC_START
        mc_min = FIXED_MC_MIN
        u_optimal = target_u

        mr_at_target = 2 * a * target_u + b
        mc_at_target = mc_start - slope1 * target_u
        m.addConstr(mr_at_target == mc_at_target, "first_order_condition")

        m.addConstr(mc_start - slope1 * u_optimal == mc_min, "cost_continuity")

        m.addConstr(a * 0.99 + b >= 0, "positive_revenue_at_high_u")

        m.addConstr(target_u >= TERMINAL_CAPACITY_MIN_VC_RATIO, "min_vc_ratio")

        revenue_per_container_at_target = a * target_u + b
        total_cost_integral = mc_start * target_u - 0.5 * slope1 * target_u ** 2
        avg_cost_per_container_at_target = total_cost_integral / target_u
        profit_per_container_at_target = revenue_per_container_at_target - avg_cost_per_container_at_target

        m.setObjective(profit_per_container_at_target, GRB.MAXIMIZE)
        m.optimize()

        if m.status == GRB.OPTIMAL:
            a_val, b_val = a.X, b.X
            slope1_val, slope2_val = slope1.X, slope2.X
            c_val = 0

            revenue_params = (a_val, b_val, c_val)
            cost_params = (mc_start, mc_min, u_optimal, slope1_val, slope2_val)

            return OptimizedTerminalModel(revenue_params, cost_params, capacity)
        else:
            print(f"    Gurobi optimization failed with status: {m.status}")
            exit()

    except Exception as e:
        print(f"    Gurobi error: {e}")
        exit()


def generate_terminal_parameters() -> Tuple[List[float], List[str]]:
    """
    Generates target V/C ratios and terminal names for each of the three groups.
    """
    target_vc_ratios = []
    terminal_names = []

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
    print("=" * 160)
    print(
        f"{'Terminal':<25} {'Target V/C':<12} {'Actual V/C':<12} {'VC Error':<10} {'Volume (TEU)':<15} {'Capacity (TEU)':<15} {'Revenue/TEU':<12} {'Cost/TEU':<12} {'Profit/TEU':<12} {'Total Profit':<15} {'MR=MC Error':<12}")
    print("-" * 160)
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

        print(
            f"{terminal_names[i]:<25} {target_u:<12.1%} {actual_u:<12.1%} {vc_error:<10.1%} {volume:<15,.0f} {terminal.capacity:<15,.0f} ${revenue_per_teu:<11.2f} ${cost_per_teu:<11.2f} ${max_profit_per_teu:<11.2f} ${total_profit:<14,.0f} ${mr_mc_error:<11.4f}")
    print("=" * 160)


def plot_terminal_functions_single_method(terminals: List[OptimizedTerminalModel],
                                          target_ratios: List[float],
                                          terminal_names: List[str],
                                          actual_optimals: List[float],
                                          method_name: str):
    u_range = np.linspace(0.01, 0.99, 1000)
    colors = plt.cm.viridis(np.linspace(0, 1, len(terminals)))

    if not os.path.exists("terminal_plots"):
        os.makedirs("terminal_plots")

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle(f'Terminal Functions - {method_name.upper()} Method', fontsize=20, fontweight='bold')

    for i, terminal in enumerate(terminals):
        color = colors[i]
        label = terminal_names[i]

        revenue_per_container_vals = terminal.revenue_per_container(u_range)
        total_revenue_vals = revenue_per_container_vals * u_range * terminal.capacity
        total_cost_vals = terminal.total_cost(u_range)
        total_profit_vals = total_revenue_vals - total_cost_vals

        marginal_revenue_vals = terminal.marginal_revenue(u_range)
        marginal_cost_vals = terminal.marginal_cost(u_range)
        marginal_profit_vals = marginal_revenue_vals - marginal_cost_vals

        ax1 = axes[0, 0]
        ax1.plot(u_range, total_revenue_vals, label=label, color=color, linewidth=2)
        ax1.set_title('Total Revenue')
        ax1.set_xlabel('V/C Ratio')
        ax1.set_ylabel('Total Revenue ($)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = axes[0, 1]
        ax2.plot(u_range, total_cost_vals, label=label, color=color, linewidth=2)
        ax2.set_title('Total Cost')
        ax2.set_xlabel('V/C Ratio')
        ax2.set_ylabel('Total Cost ($)')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3 = axes[0, 2]
        ax3.plot(u_range, total_profit_vals, label=label, color=color, linewidth=2)
        ax3.axvline(x=actual_optimals[i], color=color, linestyle='--', alpha=0.7,
                    label=f'{label} Optimal')
        ax3.axvline(x=target_ratios[i], color=color, linestyle=':', alpha=0.7,
                    label=f'{label} Target')
        ax3.set_title('Total Profit')
        ax3.set_xlabel('V/C Ratio')
        ax3.set_ylabel('Total Profit ($)')
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax4 = axes[1, 0]
        ax4.plot(u_range, marginal_revenue_vals, label=label, color=color, linewidth=2)
        ax4.set_title('Marginal Revenue')
        ax4.set_xlabel('V/C Ratio')
        ax4.set_ylabel('MR ($)')
        ax4.grid(True, linestyle='--', alpha=0.6)

        ax5 = axes[1, 1]
        ax5.plot(u_range, marginal_cost_vals, label=label, color=color, linewidth=2)
        ax5.set_title('Marginal Cost')
        ax5.set_xlabel('V/C Ratio')
        ax5.set_ylabel('MC ($)')
        ax5.grid(True, linestyle='--', alpha=0.6)

        ax6 = axes[1, 2]
        ax6.plot(u_range, marginal_profit_vals, label=label, color=color, linewidth=2)
        ax6.plot(u_range, [0] * len(u_range), 'k--', alpha=0.7)
        ax6.axvline(x=actual_optimals[i], color=color, linestyle='--', alpha=0.5)
        ax6.axvline(x=target_ratios[i], color=color, linestyle=':', alpha=0.5)
        ax6.set_title('Marginal Profit (MR - MC)')
        ax6.set_xlabel('V/C Ratio')
        ax6.set_ylabel('Marginal Profit ($)')
        ax6.grid(True, linestyle='--', alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    save_path = os.path.join("terminal_plots", f"terminals_{method_name.lower()}_method.png")
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=300)
        print(f"  {method_name.capitalize()} method figure saved at '{save_path}'")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_methods_comparison(all_results: Dict):
    methods = list(all_results.keys())
    if len(methods) < 2:
        return

    u_range = np.linspace(0.01, 0.99, 1000)

    if not os.path.exists("terminal_plots"):
        os.makedirs("terminal_plots")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Method Comparison - Total Profit for Each Terminal', fontsize=16, fontweight='bold')

    terminal_names = all_results[methods[0]]['terminal_names']
    colors = ['red', 'blue', 'green']

    for term_idx in range(len(terminal_names)):
        ax = axes[term_idx]
        ax.set_title(f'{terminal_names[term_idx]}')

        for method_idx, method in enumerate(methods):
            if method in all_results:
                terminal = all_results[method]['terminals'][term_idx]
                actual_optimal = all_results[method]['actual_optimals'][term_idx]
                target_ratio = all_results[method]['target_ratios'][term_idx]

                revenue_per_container_vals = terminal.revenue_per_container(u_range)
                total_revenue_vals = revenue_per_container_vals * u_range * terminal.capacity
                total_cost_vals = terminal.total_cost(u_range)
                total_profit_vals = total_revenue_vals - total_cost_vals

                ax.plot(u_range, total_profit_vals, label=f'{method.capitalize()}',
                        color=colors[method_idx], linewidth=2)
                ax.axvline(x=actual_optimal, color=colors[method_idx],
                           linestyle='--', alpha=0.5, label=f'{method.capitalize()} Optimal')
                ax.axvline(x=target_ratio, color=colors[method_idx],
                           linestyle=':', alpha=0.5, label=f'{method.capitalize()} Target')

        ax.set_xlabel('V/C Ratio')
        ax.set_ylabel('Total Profit ($)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()

    save_path = os.path.join("terminal_plots", "methods_comparison.png")
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=300)
        print(f"Methods comparison figure saved at '{save_path}'")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


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

                if abs(target_u - 0.65) < 0.05:
                    term_type = 'Premium'
                elif abs(target_u - 0.75) < 0.05:
                    term_type = 'Balanced'
                elif abs(target_u - 0.85) < 0.05:
                    term_type = 'High-Volume'
                else:
                    term_type = 'Custom'

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
                    'Total_Profit_at_Optimal': total_profit
                })

            parameters_df = pd.DataFrame(data_list)
            parameters_df.to_excel(writer, sheet_name=f'{method.capitalize()}_Params', index=False)

            summary_data_list = []
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

                summary_data_list.append({
                    'Terminal': terminal_names[i],
                    'Target_VC': target_u,
                    'Actual_VC': actual_u,
                    'VC_Error': vc_error,
                    'Volume_TEU': volume,
                    'Capacity_TEU': terminal.capacity,
                    'Revenue_per_TEU': revenue_per_teu,
                    'Cost_per_TEU': cost_per_teu,
                    'Profit_per_TEU': max_profit_per_teu,
                    'Total_Profit': total_profit,
                    'MR_MC_Error': mr_mc_error
                })

            summary_df = pd.DataFrame(summary_data_list)
            summary_df.to_excel(writer, sheet_name=f'{method.capitalize()}_Summary', index=False)

    print(f"\nAll methods data saved to {file_name}")


if __name__ == "__main__":
    print("=" * 80)
    print("CONTAINER TERMINAL OPTIMIZATION - ALL METHODS COMPARISON")
    print("=" * 80)

    # Generate terminal parameters
    target_vc_ratios, terminal_names = generate_terminal_parameters()

    print(f"Total terminals to be optimized: {NUM_TOTAL_TERMINALS}")
    print(f"Target V/C ranges: {VC_RANGES}")

    np.random.seed(42)
    capacities = np.random.randint(low=TERMINAL_CAPACITY_MIN, high=TERMINAL_CAPACITY_MAX + 1,
                                   size=NUM_TOTAL_TERMINALS).tolist()
    print(f"Terminal capacities: {capacities}")
    print("=" * 80)

    all_results = {}
    methods_to_run = []
    if GUROBI_AVAILABLE: methods_to_run.append('gurobi')

    if not methods_to_run:
        print("Neither Gurobi nor Scipy is available. Exiting.")
        exit()

    for method in methods_to_run:
        print(f"\n🚀 RUNNING {method.upper()} OPTIMIZATION")
        print("=" * 60)

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
        plot_terminal_functions_single_method(results[0], results[1], results[2], results[4], method)

    if len(all_results) > 1:
        print(f"\n📊 CREATING METHODS COMPARISON PLOT")
        print("=" * 50)
        plot_methods_comparison(all_results)

    print(f"\n💾 SAVING DATA TO EXCEL")
    print("=" * 50)
    write_data_to_excel_multi_method(all_results)

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Methods run: {', '.join([m.capitalize() for m in methods_to_run])}")
    print(f"Plots created: {len(methods_to_run)} individual method plots + 1 comparison plot")
    print(f"Data saved to: n-terminal_dataOpt.xlsx")
    print("=" * 80)
