from test import generate_panel_data, save_panel_data_to_csv
from test import plot_distribution, optimize_panel_parameters
from test import initialize_solar_farm, simulate_IV_curve, plot_IV_and_PV_curves
from test import find_MPPT, print_matrix,change_degradation_factor,plot_agg_and_ACM,save_MPPT_results_to_csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import scipy.stats as stats


m, n = 10,3 # Dimensions of solar farm
# m, n = 1,1 # Dimensions of solar farm

panel_params = {
    'I_mp': 9.58, 'I_sc': 10.40, 'V_mp': 41.25, 'V_oc': 49.51, 'P_max_e': 395,
    'Tp': 273 + 15, 'Gp': 1000, 'K_V': -0.271, 'K_I': 0.038, 'Ns': 72 # why is the Ns set to be 12*1? for cell in series? How can it be 12?
} # trying with ET-M672395WB

panel_data = {(i, j): copy.deepcopy(panel_params) for i in range(1, m+1) for j in range(1, n+1)}


# Dictionary specifying new insolation values for selected panels
insolation_changes = {
    # (1,1): 400,
    # (2,1): 500,
    # (1,2): 500,
    # (2,2): 600
}

# Dictionary specifying new insolation values for selected panels
temperature_changes = {
    # (1,1): 273 + 75
    (1,1): 273 + 75,
    (2,1): 273 + 55,
    (1,2): 273 + 55,
    (2,2): 273 + 55,
    (1,3): 273 + 25,
    (2,3): 273 + 25,
    (3,1): 273 + 25,
    (3,2): 273 + 25,
    (3,3): 273 + 25
}

ideality_change = {
    # (1,1): 0.5,
    # (2,1): 0.6,
    # (1,2): 0.7,
    # (2,2): 0.8
}


panel_data = generate_panel_data(m, n, panel_params, insolation_changes, temperature_changes, ideality_change)

print("Panel data generated successfully.")
print("===========================")

save_panel_data_to_csv(panel_data, f"panel_data_{m}x{n}_insolations_manufacture_parameters_Temp.csv")

# Visualize Temperature and Insolation
temp_values = np.array([[panel_data[(i,j)]['Tp'] for j in range(1, n+1)] for i in range(1, m+1)])
gp_values = np.array([[panel_data[(i,j)]['Gp'] for j in range(1, n+1)] for i in range(1, m+1)])

plot_distribution(temp_values, "Panel Temperature Distribution (K)", "Column Index", "Row Index", "Temperature (K)", "turbo", n, m)
plot_distribution(gp_values, "Panel Insolation Distribution (W/m²)", "Column Index", "Row Index", "Insolation (W/m²)", "Greys_r", n, m)

# Optimization
results = optimize_panel_parameters(panel_data)
# exit()
save_panel_data_to_csv(results, f"panel_param_{m}x{n}_insolations_manufacture_parameters_Temp.csv")

# Initialize solar farm
solar_panel_list = initialize_solar_farm(results, q_val=1.602e-19, alpha_val=1.0, K_val=1.380649e-23, T_val=298, R_Load_val=8, m=m, n=n)


# change the alpha value to be the average of the panels NOT done yet
# print("ideality_change keys:", list(ideality_change.keys()))
# for panels in solar_panel_list:
#     key = (int(panels.m), int(panels.n))
#     print(f"Panel {panels.m},{panels.n} : {key}")
#     print(f"Type of m: {type(panels.m)}, Type of n: {type(panels.n)}")
#     if key in ideality_change:
#         panels.alpha = ideality_change[key]
#         print(f"Panel {panels.m},{panels.n} : {panels.alpha}")
#     else:
#         print(f"NULL")
#         pass
# exit()

print("Solar farm initialized successfully.")
print("===========================")
print("Solar panel param details:")
for i, panel in enumerate(solar_panel_list):
    print(f"--- Panel {i+1} (ID: {panel.id}) ---")
    print(f"  Coordinates: ({panel.row}, {panel.column})")
    print(f"  I_PV: {panel.I_PV}")
    print(f"  R_S: {panel.R_S}")
    print(f"  R_P: {panel.R_P}")
    print(f"  I0: {panel.Is}")
    print(f"  alpha: {panel.alpha}")



# Simulate IV curves
z_values, v_batt_values, i_load_values, power_values = simulate_IV_curve(solar_panel_list, m, n)



df = pd.DataFrame({'Z_Load': z_values, 'V_Batt': v_batt_values, 'I_Load': i_load_values, 'Power': power_values})
df.to_csv(f"ACM_Panel_{m}x{n}_plot_DEG.csv", index=False)

# Find MPPT
Power_output, Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP = find_MPPT(solar_panel_list, m, n)

print("\nFinal Solar Farm Results ACM:")
print("Power Output:", Power_output, "Watts")
print("Vd values:")
print_matrix(Vd_value_sol, m, n)
print("VBatt values:")
print_matrix(VBatt_value_sol, m, n)
print("I values:")
print(I_value_sol)
print("Sum of I:", I_sum)
print("I_Load (MPPT):", I_MPP)

I_MPP_ACM = I_MPP
V_MPP_ACM = Power_output/ I_MPP_ACM
save_MPPT_results_to_csv(m, n, Power_output, V_MPP_ACM, I_MPP_ACM, "ACM")

plot_IV_and_PV_curves(v_batt_values, i_load_values, power_values)



# ===========   Aggregation Part ===========
run_aggregate = True  # Set to True to run the aggregation part
if run_aggregate:
    m_agg = 1
    n_agg = 1
    
    # because we assume that the panels are identical, we can just take the first panel's parameters
    # also, the solar irradiance and temperature are the average of the panels
    panel_params_agg =  {
    'I_mp': 9.58, 'I_sc': 10.40, 'V_mp': 41.25, 'V_oc': 49.51, 'P_max_e': 395,
    'Tp': 273 + 15, 'Gp': 1000, 'K_V': -0.271, 'K_I': 0.038, 'Ns': 72 # why is the Ns set to be 12*1? for cell in series? How can it be 12?
    } 
    # modify the irr and temp values to be the average of the panels
    # panel_params_agg['Gp'] = np.max(gp_values)
    # panel_params_agg['Tp'] = np.max(temp_values)
    # modify the irr and temp values to be the mode of the panels
    # panel_params_agg['Gp'] = stats.mode(gp_values).mode[0]
    # panel_params_agg['Tp'] = stats.mode(temp_values).mode[0]

    # we take the (1,1) panel's parameters as the base for the aggregated panel
    # panel_data_agg = {(i, j): copy.deepcopy(panel_params_agg) for i in range(1, m_agg+1) for j in range(1, n_agg+1)}
    
    # For aggragation, no changes in insolation, temperature, or ideality factor
    insolation_changes = {}
    temperature_changes = {}
    ideality_change = {}

    panel_data_agg = generate_panel_data(m_agg, n_agg, panel_params_agg, insolation_changes, temperature_changes, ideality_change)
    print("Panel data generated successfully.")
    print("===========================")
    save_panel_data_to_csv(panel_data_agg, f"panel_data_{m_agg}x{n_agg}_insolations_manufacture_parameters_Temp.csv")
    # Initialize solar farm
    results_agg = optimize_panel_parameters(panel_data_agg)
    solar_panel_list_agg = initialize_solar_farm(results_agg, q_val=1.602e-19, alpha_val=1.0, K_val=1.380649e-23, T_val=298, R_Load_val=8, m=m_agg, n=n_agg)
    print("Solar farm initialized successfully.")
    print("===========================")
    print("Solar panel param details:")
    for i, panel in enumerate(solar_panel_list_agg):
        print(f"--- Panel {i+1} (ID: {panel.id}) ---")
        print(f"  Coordinates: ({panel.row}, {panel.column})")
        print(f"  I_PV: {panel.I_PV}")
        print(f"  R_S: {panel.R_S}")
        print(f"  R_P: {panel.R_P}")
        print(f"  I0: {panel.Is}")
        print(f"  alpha: {panel.alpha}")
    # Simulate IV curves
    z_values_agg, v_batt_values_agg, i_load_values_agg, power_values_agg = simulate_IV_curve(solar_panel_list_agg, m_agg, n_agg)
    # This is the single panel of the aggregated solar farm, thus we need to multiply the values by m or n
    # depending on the orientation of the solar farm consider we have m x n panels, m in series and n in parallel
    v_batt_values_agg = [v * m for v in v_batt_values_agg]
    i_load_values_agg = [i * n for i in i_load_values_agg]
    power_values_agg = [p * m * n for p in power_values_agg]
    plot_IV_and_PV_curves(v_batt_values_agg, i_load_values_agg, power_values_agg)
    df_agg = pd.DataFrame({'Z_Load': z_values_agg, 'V_Batt': v_batt_values_agg, 'I_Load': i_load_values_agg, 'Power': power_values_agg})
    df_agg.to_csv(f"AGG_Panel_{m_agg}x{n_agg}_plot.csv", index=False)

    # Find MPPT
    Power_output, Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP = find_MPPT(solar_panel_list_agg, m_agg, n_agg)
    Power_output = Power_output * m * n
    Vd_value_sol = [Vd_value_sol * m for Vd_value_sol in Vd_value_sol]
    VBatt_value_sol = [VBatt_value_sol * m for VBatt_value_sol in VBatt_value_sol]
    I_value_sol = [I_value_sol * n for I_value_sol in I_value_sol]
    I_sum = I_sum * n
    I_MPP = I_MPP * n

    print("\nFinal Solar Farm Results ACM:")
    print("Power Output:", Power_output, "Watts")
    print("Vd values:")
    print(Vd_value_sol[0])
    print("VBatt values:")
    print(VBatt_value_sol[0])
    print("I values:")
    print(I_value_sol[0])
    print("Sum of I:", I_sum)
    print("I_Load (MPPT):", I_MPP)
    I_MPP_agg = I_MPP
    V_MPP_agg = Power_output/ I_MPP_agg
    save_MPPT_results_to_csv(m_agg, n_agg, Power_output, V_MPP_agg, I_MPP_agg, "AGG")
    plot_agg_and_ACM(v_batt_values, i_load_values, power_values,v_batt_values_agg, i_load_values_agg, power_values_agg,I_MPP_ACM, V_MPP_ACM, I_MPP_agg, V_MPP_agg)