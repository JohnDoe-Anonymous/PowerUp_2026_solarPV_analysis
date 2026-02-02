# Here the same as the Legacy code but I will represent each panel as a powersource object
# This is for simplyfication and future I can import each Vmpp*Impp from each panel object directly as the powersource

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

m = 2 # rows/panels in series
n = 2 # columns/panels in parallel

class panel_as_power_source:
    def __init__(self, Pmp, row, column):
        self.Pmp = Pmp
        # self.eta = eta # Assume power converter (DC-DC) is ideal for now moved to conv_as_voltage_source class
        self.row = row
        self.column = column
        self.coordinate = (row, column)

class conv_as_voltage_source: 
    # Maybe the best way is to represent it as a current controlled voltage source
    # The inverter will draw less current then Sum(Vinv) < Vdc_ragulation
    def __init__(self, eta_conv, row, column):
        self.Vinv = None
        self.Iinv = None 
        self.eta_conv = eta_conv # Assume power converter (DC-DC) is ideal for now 
        self.row = row
        self.column = column
        self.coordinate = (row, column)
        self.Pinv = None

    def get_power_output_from_power_source(self, panel_data):
        self.Pinv = panel_data[(self.row, self.column)].Pmp * self.eta_conv

class Inv_controller:
    # Here there are some regulations that the inverter and converter must follow
    # 1. Iinv = Pinv / Vinv = eta_conv * Pmp / Vinv = I_string
    # 2. Sum(Vinv) = Vdc_regulation
    def __init__(self, conv_data, m, n, Vdc_regulation):
        self.conv_data = conv_data
        self.m = m
        self.n = n
        self.Vdc_regulation = Vdc_regulation
        # I_string should have n values, one for each parallel string
        


def generate_panel_data_as_power_source(m, n):
    # Now I assume 350W panel with ideal converter for simplicity for all panels
    panel_data = {(i, j): panel_as_power_source(350, i, j) for i in range(0, m) for j in range(0, n)}
    # For specific panels, I can change the Pmp and eta values
    panel_data[(0, 0)].Pmp = 400
    conv_data = {(i, j): conv_as_voltage_source(48, 1.0, i, j) for i in range(0, m) for j in range(0, n)}
    for i in range(0, m):
        for j in range(0, n):
            conv_data[(i, j)].get_power_output_from_power_source(panel_data)
    return panel_data, conv_data

# Visualize power distribution
def visualize_power_distribution(panel_data, m, n):
    try:
        power_values = np.array([[panel_data[(i,j)].Pmp for j in range(0, n)] for i in range(0, m)])
    except:
        power_values = np.array([[panel_data[(i,j)].Pdc for j in range(0, n)] for i in range(0, m)])
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(power_values, cmap='viridis', interpolation='nearest')
    ax.set_title('Power Distribution Across Solar Farm Panels')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(0, n))
    ax.set_yticks(np.arange(m))
    ax.set_yticklabels(np.arange(0, m))
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(m + 1) - 0.5, minor=True)
    ax.grid(True, color='black', linestyle='-', linewidth=2, which='minor')
    fig.colorbar(cax, label='Power (W)')

    
    plt.show()

# +++ San Check +++ # 

# panel_data, conv_data = generate_panel_data_as_power_source(m, n)
# visualize_power_distribution(panel_data, m, n)
# visualize_power_distribution(conv_data, m, n)

# +++ End San Check +++ #

from pyomo.environ import ConcreteModel, Var, Param, RangeSet, SolverFactory, value
