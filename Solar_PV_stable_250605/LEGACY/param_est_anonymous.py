import pyomo.environ as pyo
import numpy as np

# Constants
q = 1.602e-19  # Charge of electron [C]
k = 1.381e-23  # Boltzmann constant [J/K]
T = 298.15     # Temperature [K]
Ns = 72        # Number of cells in series

def estimate_pv_parameters(Isc, Voc, Imp, Vmp, fixed_params=None, verbose=False):
    """
    Estimate PV diode model parameters using Pyomo and IPOPT.

    Parameters:
    - Isc, Voc, Imp, Vmp: STC data (floats)
    - fixed_params: dict, any parameters to fix (e.g., {'Iph': 11.29, 'n': 1.2})
    - verbose: bool, whether to print solver output

    Returns:
    - dict with estimated parameter values
    """
    model = pyo.ConcreteModel()

    model.Iph = pyo.Var(bounds=(10, 12), initialize=Isc)
    model.I0 = pyo.Var(bounds=(1e-12, 1e-6), initialize=1e-10)
    model.n = pyo.Var(bounds=(1.0, 2.0), initialize=1.2)
    model.Rs = pyo.Var(bounds=(0, 2), initialize=0.5)
    model.Rsh = pyo.Var(bounds=(10, 2000), initialize=1000)

    if fixed_params:
        for param_name, value in fixed_params.items():
            getattr(model, param_name).fix(value)

    def diode_eq(V, I, Iph, I0, n, Rs, Rsh):
        Vt = n * k * T * Ns / q
        return Iph - I0 * (pyo.exp((V + I * Rs) / Vt) - 1) - (V + I * Rs) / Rsh

    def obj_rule(m):
        eq1 = diode_eq(0, Isc, m.Iph, m.I0, m.n, m.Rs, m.Rsh) - Isc
        eq2 = diode_eq(Voc, 0, m.Iph, m.I0, m.n, m.Rs, m.Rsh)
        eq3 = diode_eq(Vmp, Imp, m.Iph, m.I0, m.n, m.Rs, m.Rsh) - Imp
        return eq1**2 + eq2**2 + eq3**2

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    solver = pyo.SolverFactory('ipopt')
    solver.solve(model, tee=verbose)

    result = {
        'Iph': pyo.value(model.Iph),
        'I0': pyo.value(model.I0),
        'n': pyo.value(model.n),
        'Rs': pyo.value(model.Rs),
        'Rsh': pyo.value(model.Rsh),
    }
    return result


def estimate_pv_parameters_E(i_sc,v_oc,i_mp, v_mp,cells_in_series, T=298.15, n=1.1):
    k = 1.380649e-23  # Boltzmann constant [J/K]
    q = 1.602176634e-19  # Elementary charge [C]
    V_th = n * k * T / q * cells_in_series
    I_0 = i_sc / (np.exp(v_oc / V_th) - 1)
    I_ph = i_sc
    R_s = (V_th * np.log((I_ph - i_mp) / I_0) - v_mp) / i_mp
    exp_term = np.exp((v_mp + i_mp * R_s) / V_th)
    R_sh = (v_mp + i_mp * R_s) / (I_ph - i_mp - I_0 * exp_term)
    print(f"R_s: {R_s:.4f} Ohm")
    print(f"R_sh: {R_sh:.4f} Ohm")
    print(f"I_0: {I_0:.4e} A")
    
    result = {
        'Iph': I_ph,
        'I0': I_0,
        'n': n,
        'Rs': R_s,
        'Rsh': R_sh
    }
    return result

if __name__ == "__main__":
    # STC data
    Isc = 11.29
    Voc = 43.6
    Imp = 10.69
    Vmp = 37.0

    # You can fix certain parameters if desired
    fixed_params = {
        # 'Iph': Isc,
        # 'n': 1.2,
    }

    results = estimate_pv_parameters_E(Isc, Voc, Imp, Vmp, fixed_params = 72, verbose=True)

    for key, val in results.items():
        print(f"{key} = {val:.6g}")