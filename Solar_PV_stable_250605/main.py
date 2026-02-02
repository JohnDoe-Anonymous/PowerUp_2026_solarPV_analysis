from pyomo.environ import ConcreteModel, Var, Constraint, Objective, SolverFactory, value, minimize
from RESTRCTURED.solar_pv_panel import SolarPanel
import matplotlib.pyplot as plt
import numpy as np
import copy
import json

# --- Configuration ---
m, n = 1, 1  # panel layout
panel_specs = {
    'I_mp': 9.58, 'I_sc': 10.4, 'V_mp': 41.25, 'V_oc': 49.51, 'P_max_e': 395,
    'Tp': 298, 'Gp': 1000, 'K_V': -0.271, 'K_I': 0.038, 'Ns': 72
}

# --- Instantiate and estimate parameters ---
panels = []
for i in range(1, m + 1):
    for j in range(1, n + 1):
        panel = SolarPanel(row=i, col=j, parameters=copy.deepcopy(panel_specs))
        if (i, j) == (2, 1):
            panel.apply_environment(Gp=800)  # example variation
        panel.estimate_parameters()
        panels.append(panel)

# exit()

# --- IV sweep ---
z_sweep = np.logspace(-7, 7, 300)
v_results, i_results, p_results = [], [], []

for Z_load in z_sweep:
    model_z_sweep = ConcreteModel()
    model_z_sweep.V_terminal = Var(initialize=0.1, bounds=(0, 500))
    model_z_sweep.I_terminal = Var(initialize=10.0)
    model_z_sweep.V_D = Var(initialize=0.1, shape=(m,n))
    model_z_sweep.V_panel = Var(initialize=0.1, shape=(m,n))

    current_vars = []

    for idx, panel in enumerate(panels):
        model_z_sweep.add_component(f"I_panel_{idx}", Var(initialize=1.0))
        I_var = getattr(model_z_sweep, f"I_panel_{idx}")
        current_vars.append(I_var)

        panel.add_to_pyomo_model(model_z_sweep, index=idx)
        model_z_sweep.add_component(f"link_vbatt_{idx}", Constraint(expr=panel.vars['VBatt'] == model_z_sweep.V_terminal))

        Rs = panel.export_parameters()['R_S']
        model_z_sweep.add_component(
            f"current_eq_{idx}",
            Constraint(expr=I_var == (panel.vars['Vd'] - panel.vars['VBatt']) / Rs)
        )

    model_z_sweep.total_current_eq = Constraint(expr=model_z_sweep.I_terminal == sum(current_vars))
    model_z_sweep.load_eq = Constraint(expr=model_z_sweep.V_terminal == Z_load * model_z_sweep.I_terminal)
    model_z_sweep.obj = Objective(expr=model_z_sweep.V_terminal * model_z_sweep.I_terminal, sense=minimize)

    try:
        SolverFactory("ipopt").solve(model_z_sweep)
        V = value(model_z_sweep.V_terminal)
        I = value(model_z_sweep.I_terminal)
        v_results.append(V)
        i_results.append(I)
        p_results.append(V * I)
    except:
        v_results.append(None)
        i_results.append(None)
        p_results.append(None)

# --- Plot results ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(v_results, i_results, label='I-V Curve')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(v_results, p_results, label='P-V Curve', color='green')
plt.xlabel('Voltage (V)')
plt.ylabel('Power (W)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()