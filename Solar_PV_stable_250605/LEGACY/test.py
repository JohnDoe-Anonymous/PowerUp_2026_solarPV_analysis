# This is the helper function file for the main.py file.
# It contains the functions that are used to generate the panel data, save it to an Excel file,
# plot the data, and perform the optimization.

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from param_estimation import PVCellArrayModel
from param_estimation import PVPanelDataProcessor
from Simulator_diode_peng import SolarFarmDefine,SolarFarmSolverSim
from param_est_peng import estimate_pv_parameters_E

# --------------------------- Function Definitions ---------------------------

def generate_panel_data(m, n, panel_params, insolation_changes, temperature_changes, ideality_change):
    panel_data = {(i, j): copy.deepcopy(panel_params) for i in range(1, m + 1) for j in range(1, n + 1)}
    
    for (i, j), new_gp in insolation_changes.items():
        if i < 1 or i > m or j < 1 or j > n:
            pass
        else:
            panel_data[(i, j)]['Gp'] = new_gp

    for (i, j), new_tp in temperature_changes.items():
        if i < 1 or i > m or j < 1 or j > n:
            pass
        else:
            panel_data[(i, j)]['Tp'] = new_tp

    # Print updated data
    for key, value in panel_data.items():
        # print(f"Panel {key}: {value['Gp']} W/m²")
        # print(f"Panel {key}: {value['Tp']} K")
        pass

    return panel_data

def save_panel_data_to_excel(panel_data, filename):
    df = pd.DataFrame.from_dict(panel_data, orient='index')
    df.to_excel(filename)
    print(f"Excel file '{filename}' saved successfully.")

def save_panel_data_to_csv(panel_data, filename):
    df = pd.DataFrame.from_dict(panel_data, orient='index')
    df.to_csv(filename)
    print(f"Excel file '{filename}' saved successfully.")

def plot_distribution(matrix, title, xlabel, ylabel, colorbar_label, cmap, n, m):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticks(np.arange(m))
    ax.set_yticklabels(np.arange(1, m + 1))
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
    ax.grid(True, color='black', linestyle='-', linewidth=2, which='minor')
    fig.colorbar(cax, label=colorbar_label)
    plt.show()

def optimize_panel_parameters(panel_data, tolerance=0.02):
    pv_array_model = PVCellArrayModel(panel_data)
    results = pv_array_model.estimate_pv_parameters_E(tolerance=tolerance)
    print("Panel parameters optimized successfully.")
    print("Optimized panel data:", results)
    
    return results

def initialize_solar_farm(results, q_val, alpha_val, K_val, T_val, R_Load_val, m, n):
    data_processor = PVPanelDataProcessor(
        results=results,
        q_val=q_val,
        alpha_val=alpha_val,
        K_val=K_val,
        T_val=T_val,
        R_Load_val=R_Load_val,
        total_row=m,
        total_col=n
    )
    panel_data_list = data_processor.generate_panel_data_list()
    solar_panel_list = [SolarFarmDefine(*panel_data) for panel_data in panel_data_list]
    print("Solar panel list initialized successfully.")
    print("Solar panel data:", solar_panel_list)
    return solar_panel_list

def change_degradation_factor(solar_panel_list,m,n, new_alpha):
    for panel in solar_panel_list:
        if panel.m == m and panel.n == n:
            panel.alpha = new_alpha
    print(f"Degradation factor changed to {new_alpha} for panel ({m}, {n}).")
    return solar_panel_list

def simulate_IV_curve(solar_panel_list, m, n, num_points=1000):
    z_values = np.logspace(-3, 3, num=num_points).tolist()
    v_batt_values, i_load_values, power_values, v_batt_all = [], [], [], {}
    initial_guess = None
    for z in z_values:
        
        for panel in solar_panel_list:
            panel.Z_Load = z
        solar_panel_solving = SolarFarmSolverSim(solar_panel_list, m, n, initial_guess, find_MPP=False)

        Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP, initial_guess = solar_panel_solving.solve()

        v_batt = VBatt_value_sol[1]
        i_load = I_MPP 
        power = v_batt * i_load

        v_batt_values.append(v_batt)
        i_load_values.append(i_load)
        power_values.append(power)
        v_batt_all[z] = VBatt_value_sol

    return z_values, v_batt_values, i_load_values, power_values

def plot_IV_and_PV_curves(v_batt_values, i_load_values, power_values):
    plt.figure(figsize=(12, 6))
    plt.plot(v_batt_values, i_load_values, label="I_Load vs VBatt[1]", color="blue")
    plt.xlabel("VBatt (V)")
    plt.ylabel("I_Load (A)")
    plt.title("I_Load vs VBatt[1]")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(v_batt_values, power_values, label="Power vs VBatt[1]", color="green")
    plt.xlabel("VBatt[1] (V)")
    plt.ylabel("Power (W)")
    plt.title("Power vs VBatt[1]")
    plt.legend()
    plt.grid()
    plt.show()

def plot_agg_and_ACM(v_batt_values, i_load_values, power_values,v_batt_values_agg, i_load_values_agg, power_values_agg,I_MPP_ACM, V_MPP_ACM, I_MPP_agg, V_MPP_agg):
    plt.figure(figsize=(12, 6))
    plt.plot(v_batt_values, i_load_values, label="I_Load vs VBatt[1] (ACM)", color="blue")
    plt.plot(v_batt_values_agg, i_load_values_agg, label="I_Load vs VBatt[1] (Aggregate)", color="orange", linestyle='--')
    
    # Plot MPPT points
    plt.axvline(x=V_MPP_ACM, color='blue', linestyle='--', label=f'MPP ACM: {V_MPP_ACM:.2f} V')
    plt.axvline(x=V_MPP_agg, color='orange', linestyle='--', label=f'MPP Aggregate: {V_MPP_agg:.2f} V')
    plt.axhline(y=I_MPP_ACM, color='blue', linestyle='--', label=f'I_MPP ACM: {I_MPP_ACM:.2f} A')
    plt.axhline(y=I_MPP_agg, color='orange', linestyle='--', label=f'I_MPP Aggregate: {I_MPP_agg:.2f} A')
    
    plt.xlabel("VBatt (V)")
    plt.ylabel("I_Load (A)")
    plt.title("I_Load vs VBatt[1] Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(v_batt_values, power_values, label="Power vs VBatt[1] (ACM)", color="green")
    plt.plot(v_batt_values_agg, power_values_agg, label="Power vs VBatt[1] (Aggregate)", color="red", linestyle='--')
    
    # Plot MPPT points
    plt.axvline(x=V_MPP_ACM, color='blue', linestyle='--', label=f'MPP ACM: {V_MPP_ACM:.2f} V')
    plt.axvline(x=V_MPP_agg, color='orange', linestyle='--', label=f'MPP Aggregate: {V_MPP_agg:.2f} V')
    plt.axhline(y=I_MPP_ACM * V_MPP_ACM, color='blue', linestyle='--', label=f'Power ACM: {I_MPP_ACM * V_MPP_ACM:.2f} W')
    plt.axhline(y=I_MPP_agg * V_MPP_agg, color='orange', linestyle='--', label=f'Power Aggregate: {I_MPP_agg * V_MPP_agg:.2f} W')
    
    plt.xlabel("VBatt[1] (V)")
    plt.ylabel("Power (W)")
    plt.title("Power vs VBatt[1] Comparison")
    plt.legend()
    plt.grid()
    plt.show()

def find_MPPT(solar_panel_list, m, n):
    initial_guess = None
    solver = SolarFarmSolverSim(solar_panel_list, m, n, initial_guess, find_MPP=True)
    Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP, initial_guess = solver.solve()
    Power_output = VBatt_value_sol[1] * I_MPP
    return Power_output, Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP

def save_MPPT_results_to_csv(m, n, Power_output, V_MPP, I_MPP,type):
    data = {
        'Power Output (W)': [Power_output],
        'V_MPP (V)': [V_MPP],
        'I_Load (MPPT)': [I_MPP],
        'Type': [type]
    }
    df = pd.DataFrame(data)
    filename = f"MPPT_results_{m}x{n}_{type}.csv"
    df.to_csv(filename, index=False)
    print(f"MPPT results saved to {filename}")

def print_matrix(data_dict, rows, columns):
    matrix = [[0] * columns for _ in range(rows)]
    for idx in range(1, len(data_dict) + 1):
        col = (idx - 1) // rows
        row = (idx - 1) % rows
        matrix[row][col] = data_dict[idx]

    for row in matrix:
        print(' '.join(f'{val:8.2f}' for val in row))

# --------------------------- Main Execution ---------------------------
# moved to main.py