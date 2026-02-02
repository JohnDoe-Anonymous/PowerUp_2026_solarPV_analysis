from pyomo.environ import Var, Param, Constraint, Expression, exp, value
import numpy as np

class SolarPanel:
    def __init__(self, row, col, parameters):
        self.col = col
        self.row = row
        self.parameters = parameters.copy()
        self.estimated_params = None
        self.vars = {}
        self.constraints = []

    def apply_environment(self, Gp=None, Tp=None):
        if Gp is not None:
            self.parameters['Gp'] = Gp
        if Tp is not None:
            self.parameters['Tp'] = Tp

    def estimate_parameters(self, tolerance=0.02):
        # Direct estimation without PVCellArrayModel
        p = self.parameters
        T = p.get('Tp', 298)
        G = p.get('Gp', 1000)
        Tn = 298
        Gn = 1000

        I_mp = p['I_mp']
        I_sc = p['I_sc']
        V_mp = p['V_mp']
        V_oc = p['V_oc']
        K_V = p['K_V']/100 * V_oc
        K_I = p['K_I']/100 * I_sc
        Ns = p['Ns']

        q = 1.602e-19
        k = 1.380649e-23
        a = 1.02

        delT = T - Tn
        V_t = Ns * k * T / q
        V_t_n = Ns * k * Tn / q

        I_ph = (I_sc + K_I * delT) * (G / Gn)
        I_0 = (I_sc / (np.exp(V_oc / V_t_n) - 1))

        delta_current = max(I_ph - I_mp, 1e-10)
        R_s = ((V_t * a * np.log(delta_current / I_0) - V_mp) / I_mp)

        exp_term = np.exp((V_mp + I_mp * R_s) / (a * V_t))
        R_sh = ((V_mp + I_mp * R_s) / (I_ph - I_mp - I_0 * exp_term))

        self.estimated_params = {
            "R_s": R_s,
            "R_p": R_sh,
            "a": a,
            "I_PV": I_ph,
            "I0": I_0,
            "Ns": Ns
        }
        print(f"Estimated parameters for panel ({self.row}, {self.col}):")
        print(f"  R_s: {R_s:.4f} Ohm")
        print(f"  R_p: {R_sh:.4f} Ohm")
        print(f"  I0: {I_0:.4e} A")
        print(f"  I_PV: {I_ph:.4f} A")
        print(f"  a: {a}")
        print(f"  Ns: {Ns}")
    

    def export_parameters(self):
        if self.estimated_params is None:
            raise ValueError("No parameters estimated yet")
        return {
            'I_PV': self.estimated_params['I_PV'],
            'R_S': self.estimated_params['R_s'],
            'R_P': self.estimated_params['R_p'],
            'q': 1.602e-19,
            'alpha': self.estimated_params['a'],
            'K': 1.380649e-23,
            'T': self.parameters.get('Tp', 298),
            'Is': self.estimated_params['I0']
        }

    def get_voltage_vars(self, model):
        self.vars['Vd'] = model.V_D
    


    def add_to_pyomo_model(self, model, index):
        p = self.export_parameters()

        self.vars['Vd'] = Var(initialize=0.5, bounds=(0, 500))
        self.vars['VBatt'] = Var(initialize=0.5, bounds=(0, 500))

        model.add_component(f"Vd_{index}", self.vars['Vd'])
        model.add_component(f"VBatt_{index}", self.vars['VBatt'])

        V_t = p['K'] * p['T'] / p['q']
        V_t_alpha = p['alpha'] * V_t * 72  # Assuming Ns=72

        def diode_kcl_rule(m):
            V_diff = self.vars['Vd'] - self.vars['VBatt']
            diode_current = p['Is'] * (exp(V_diff / (V_t_alpha + 1e-8)) - 1)
            Rp_current = V_diff / (p['R_P'] + 1e-6)
            Rs_current = (self.vars['Vd'] - self.vars['VBatt']) / p['R_S']
            return -p['I_PV'] + diode_current + Rp_current + Rs_current == 0

        c = Constraint(rule=diode_kcl_rule)
        model.add_component(f"KCL_diode_{index}", c)
        self.constraints.append(c)

    
