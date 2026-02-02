from param_estimation import PVCellArrayModel
from param_estimation import PVPanelDataProcessor
from solar_farm_definer_class_diode_24_april import SolarFarmDefine
from solar_farm_definer_class_diode_24_april import SolarFarmSolver
from Simulator_diode_24_april import SolarFarmSolverSim
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Grouping by column

m = 3 # CHANGE Keep 1 for aggregate, change for indie
n = 2 # CHANGE Keep 1 for aggregate, change for indie


# Parameters for all panels
panel_params = {
    'I_mp': 9.52, 
    'I_sc': 10.24, 
    'V_mp': 41.5, 
    'V_oc': 49.6, 
    'P_max_e': 395,
    'Tp': 273 + 25, 
    'Gp': 1000, 
    'K_V': -0.271, 
    'K_I': 0.038, 
    'Ns': 72  #12*m only for aggregated model. For indie panel its 12*1
}


# Generating a 6x6 panel array
panel_data = {(i, j): copy.deepcopy(panel_params) for i in range(1, m+1) for j in range(1, n+1)}

# Dictionary specifying new insolation values for selected panels
insolation_changes = {
    (1,2): 0
}

# Dictionary specifying new insolation values for selected panels
temperature_changes = {
    # (1, 1): 273+40
}

# # Apply changes
for (i, j), new_gp in insolation_changes.items():
    panel_data[(i, j)]['Gp'] = new_gp

# Apply changes
for (i, j), new_tp in temperature_changes.items():
    panel_data[(i, j)]['Tp'] = new_tp

# Print updated data
for key, value in panel_data.items():
    print(f"Panel {key}: {value['Gp']} W/m²")

print(panel_data)

# Convert panel_data dictionary to a DataFrame
df = pd.DataFrame.from_dict(panel_data, orient='index')

# Save to an Excel file
df.to_excel("panel_data_3x2_insolations_manufacture_parameters_Temp_1_2_3.xlsx")

print("Excel file saved successfully!")

print("Code is paused. Press Enter to continue...")
input()  # Waits for user input before proceeding
print("Code resumed.")

# Extract Tp and Gp values for each panel
temp_values = np.zeros((m, n))  # Temperature matrix
gp_values = np.zeros((m, n))    # Insolation matrix

for (row, col), params in panel_data.items():
    temp_values[row - 1, col - 1] = params['Tp']  # Store temperature
    gp_values[row - 1, col - 1] = params['Gp']    # Store insolation

# Create subplots for temperature distribution
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(temp_values, cmap='turbo', interpolation='nearest')
ax.set_title("Panel Temperature Distribution (K)")
ax.set_xlabel("Column Index")
ax.set_ylabel("Row Index")

# Adjust tick positions to start from 1 instead of 0
ax.set_xticks(np.arange(n))  # Grid at integer positions
ax.set_xticklabels(np.arange(1, n + 1))  # Labels start from 1
ax.set_yticks(np.arange(m))  # Grid at integer positions
ax.set_yticklabels(np.arange(1, m + 1))  # Labels start from 1

# Add gridlines without extra boxes
ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
ax.grid(True, color='black', linestyle='-', linewidth=2, which='minor')

fig.colorbar(cax, label="Temperature (K)")
plt.show()

# Create subplots for insolation distribution
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(gp_values, cmap='Greys_r', interpolation='nearest')
ax.set_title("Panel Insolation Distribution (W/m²)")
ax.set_xlabel("Column Index")
ax.set_ylabel("Row Index")

# Adjust tick positions to start from 1 instead of 0
ax.set_xticks(np.arange(n))  # Grid aligns with edges
ax.set_xticklabels(np.arange(1, n + 1))  # Labels start from 1
ax.set_yticks(np.arange(m))  # Grid aligns with edges
ax.set_yticklabels(np.arange(1, m + 1))  # Labels start from 1

# Add gridlines without extra boxes
ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
ax.grid(True, color='black', linestyle='-', linewidth=2, which='minor')

fig.colorbar(cax, label="Insolation (W/m²)")
plt.show()

print(panel_data)

print("Code is paused. Press Enter to continue...")
input()  # Waits for user input before proceeding
print("Code resumed.")


# Initialize model
pv_array_model = PVCellArrayModel(panel_data)

# # Compute parameters
results = pv_array_model.compute_parameters(tolerance=0.02) # Use when different parameters are there

# Convert panel_data dictionary to a DataFrame
df2 = pd.DataFrame.from_dict(results, orient='index')

# Save to an Excel file
df2.to_excel("panel_data_3x2_insolations_parameters_estTemp_1_2_3.xlsx")

print("Excel file saved successfully!")

# """Compute parameters only once"""
# single_panel_model = PVCellArrayModel({(1, 1): panel_params})  # Use a single representative panel
# single_panel_results = single_panel_model.compute_parameters(tolerance=0.02)
 

print("Optimized Parameter Calculation Done!")

# Print results for each panel
for (m, n), params in results.items():
    print(f"Panel ({m}, {n}): {params}")

print("Code is paused. Press Enter to continue...")
input()  # Waits for user input before proceeding
print("Code resumed.")

# Constants and user-defined values
q_val = 1.602e-19  # Charge of an electron
alpha_val = 1       # Diode ideality factor
K_val = 1.380649e-23  # Boltzmann constant
T_val = 298         # Temperature in Kelvin
R_Load_val = 8       # Load resistance
total_row = m        # Total rows of panels CHANGE
total_col = n        # Total columns of panels CHANGE

# Initialize the data processor
data_processor = PVPanelDataProcessor(
    results=results,
    q_val=q_val,
    alpha_val=alpha_val,
    K_val=K_val,
    T_val=T_val,
    R_Load_val=R_Load_val,
    total_row=total_row,
    total_col=total_col
)

# Generate the panel data list
panel_data_list = data_processor.generate_panel_data_list()

# Print the generated list
for panel_data in panel_data_list:
    print(panel_data)

# Initialize 2D list for solar panels

solar_panel_list = []

for panel_data in panel_data_list:
    
    solar_cell = SolarFarmDefine(panel_data[0],panel_data[1], panel_data[2], panel_data[3], panel_data[4], panel_data[5], panel_data[6], panel_data[7], panel_data[8], panel_data[9], panel_data[10], panel_data[11], panel_data[12])

    solar_panel_list.append(solar_cell)





# List to store results
v_batt_values = []
i_load_values = []
power_values = []

# Dictionary to store V_Batt values for each Z_Load
v_batt_all = {}

# Range of Z values to test
z_values = np.logspace(-7, 6, num=1000).tolist()  # From 10⁻³ to 10⁶

i = 0

# Iterate over Z values
for z in z_values:
    i = i+1
    # Update Z_Load in solar panel data
    for panel in solar_panel_list:
        panel.Z_Load = z  # Update Z value for all panels

    # Solve the model for this Z value
    solar_panel_solving = SolarFarmSolverSim(solar_panel_list, m, n)
    Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP = solar_panel_solving.solve()

    # Extract results
    v_batt = VBatt_value_sol[1]  # Voltage of the first panel
    i_load = I_MPP               # Load current (I_MPP)
    power = v_batt * i_load      # Power output

    # Append results to lists
    v_batt_values.append(v_batt)
    i_load_values.append(i_load)
    power_values.append(power)

    # Store V_Batt values for this Z_Load
    v_batt_all[z] = VBatt_value_sol
    # print("The iteration number is ", i)

# Plot I_Load vs V_Batt[1]
plt.figure(figsize=(12, 6))
plt.plot(v_batt_values, i_load_values, label="I_Load vs VBatt[1]", color="blue")
plt.xlabel("VBatt (V)")
plt.ylabel("I_Load (A)")
plt.title("I_Load vs VBatt[1]")
plt.legend()
plt.grid()
plt.show()

# Plot Power vs V_Batt[1]
plt.figure(figsize=(12, 6))
plt.plot(v_batt_values, power_values, label="Power vs VBatt[1]", color="green")
plt.xlabel("VBatt[1] (V)")
plt.ylabel("Power (W)")
plt.title("Power vs VBatt[1]")
plt.legend()
plt.grid()
plt.show()

# Prepare the data for DataFrame
data = {
    'Z_Load': z_values,
    'V_Batt': v_batt_values,
    'I_Load': i_load_values,
    'Power': power_values
}

# Create DataFrame
df3 = pd.DataFrame(data)

# Save DataFrame to Excel
excel_filename = "base_Panel_3x2Temp_1_2_3.xlsx"
df3.to_excel(excel_filename, index = 'True')

"""This section of the code is to find MPPT Values"""

solar_panel_solving_MPPT = SolarFarmSolver(solar_panel_list,m,n)

Vd_value_sol, VBatt_value_sol, I_value_sol, I_sum, I_MPP = solar_panel_solving_MPPT.solve()


solar_potential = VBatt_value_sol[1]


I_PV_Column, R_S_Column, R_P_Column, q_Column, alpha_Column, K_Column, T_Column, Is_Column, V_Load = solar_panel_solving_MPPT.extract_all_data()

def print_matrix(data_dict, rows, columns):
    """Print dictionary data as a matrix, filled column-wise."""
    # Initialize matrix
    matrix = [[0] * columns for _ in range(rows)]
    
    # Fill the matrix with values from the dictionary
    for idx in range(1, len(data_dict) + 1):
        col = (idx - 1) // rows
        row = (idx - 1) % rows
        matrix[row][col] = data_dict[idx]
    
    # Print the matrix
    for row in matrix:
        print(' '.join(f'{val:8.2f}' for val in row))


print("Vd values (matrix):")
print_matrix(Vd_value_sol, m, n)

print("\nVBatt values (matrix):")
print_matrix(VBatt_value_sol, m, n)

print("I values:", I_value_sol)

I_Load = I_MPP 

print("I Load value:", I_Load)
print("Sum of I values:", I_sum)

I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col, R_Load_Col = solar_panel_solving_MPPT.extract_all_data()

Power_output = solar_potential*I_MPP

print("Power Output of the solar farm is: ", Power_output, "Watts")

