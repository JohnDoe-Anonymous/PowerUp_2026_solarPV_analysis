import numpy as np
from scipy.optimize import fsolve
from pyomo.environ import (
	ConcreteModel, Var, Param, RangeSet, Expression, Objective, Constraint,
	SolverFactory, minimize, value, exp
)

class PVCellArrayModel:
	def __init__(self, panel_data):
		"""
		Initialize parameters for multiple PV panels.
		:param panel_data: Dictionary of panel parameters with (m, n) as keys and values as dict of individual panel data.
						   Example: {(0, 0): {'I_mp': 9.5, 'I_sc': 10, 'V_mp': 42, 'V_oc': 50, 'P_max_e': 400}, ...}
		:param K_V: Temperature coefficient for voltage.
		:param K_I: Temperature coefficient for current.
		:param Ns: Number of series-connected cells.
		:param G: Irradiance (W/m^2).
		:param Tn: Nominal temperature (K).
		"""
		self.panel_data = panel_data
		self.Tn = 298 
		self.Gn = 1000

		# Constants
		self.q = 1.602e-19  # Elementary charge (C)
		self.k = 1.380649e-23  # Boltzmann constant (J/K)

	def _current_equation(self, I, V, I_PV, I0, R_s, V_t, a, R_p):
		"""Defines the current equation to be solved."""
		return I - (I_PV - I0 * (np.exp((V + I * R_s) / (V_t * a)) - 1) - (V + I * R_s) / R_p)
	
	def fit_panel_parameters(self, tolerance):
		"""
		Compute parameters for all panels using Pyomo + IPOPT.
		:param tolerance: Convergence tolerance for error (not directly used).
		:return: Dictionary with (m, n) as keys and computed parameters as values.
		"""
		results = {}
		k = 1.380649e-23  # Boltzmann constant [J/K]
		q = 1.602e-19     # Electron charge [C]
		Tn = self.Tn
		Gn = self.Gn

		for (m, n), panel in self.panel_data.items():
			# Extract panel data
			I_mp = panel['I_mp']
			I_sc = panel['I_sc']
			V_mp = panel['V_mp']
			V_oc = panel['V_oc']
			P_max_e = panel['P_max_e']
			T = panel['Tp']
			G = panel['Gp']
			K_V = panel['K_V'] / 100 * V_oc
			K_I = panel['K_I'] / 100 * I_sc
			Ns = panel['Ns']
			delT = T - Tn
			a = 1.02
			V_t = Ns * k * T / q

			# Build Pyomo model
			model = ConcreteModel()
			model.Rs = Var(bounds=(0.0, 2.0), initialize=0.5)
			model.Rp = Var(bounds=(1.0, 1000.0), initialize=10.0)
			model.I0 = Var(bounds=(1e-14, 1e-2), initialize=1e-6)

			# Voltage sampling
			V_array = np.linspace(0, V_oc, 100)
			model.V_pts = RangeSet(0, len(V_array) - 1)
			model.V = Param(model.V_pts, initialize={i: v for i, v in enumerate(V_array)})

			# Photo current adjusted for irradiance and temperature
			I_ph = (I_sc + K_I * delT) * (G / Gn)

			# Define current and power expressions
			def current_rule(m, i):
				V = m.V[i]
				Rs = m.Rs
				Rp = m.Rp
				I0 = m.I0
				exponent = (V + Rs * I_ph) / (a * V_t)
				return I_ph - I0 * (exp(exponent) - 1) - (V + Rs * I_ph) / Rp

			model.I = Expression(model.V_pts, rule=current_rule)
			model.P = Expression(model.V_pts, rule=lambda m, i: m.V[i] * m.I[i])

			# Objective: Minimize absolute power error
			model.error = Var(bounds=(0, None))
			model.obj = Objective(expr=model.error, sense=minimize)

			def power_constraint_rule(m):
				return m.error >= abs(P_max_e - max(value(m.P[i]) for i in m.V_pts))
			model.power_constraint = Constraint(rule=power_constraint_rule)

			# Solve
			solver = SolverFactory("ipopt")
			solver.options['tol'] = tolerance
			result = solver.solve(model, tee=False)

			# Store result in your original format
			results[(m, n)] = {
				"R_s": value(model.Rs),
				"R_p": value(model.Rp),
				"a": a,
				"I_PV": I_ph,
				"I0": value(model.I0),
				"error": value(model.error),
				"iterations": None,  # IPOPT doesn't provide it by default
				"Ns": Ns
			}

		return results
			

	def compute_parameters(self, tolerance):
		"""
		Compute parameters for all panels.
		:param tolerance: Convergence tolerance for error.
		:return: Dictionary with (m, n) as keys and computed parameters as values.
		"""
		results = {}

		for (m, n), panel in self.panel_data.items():
			# Extract panel parameters
			I_mp = panel['I_mp']
			I_sc = panel['I_sc']
			V_mp = panel['V_mp']
			V_oc = panel['V_oc']
			P_max_e = panel['P_max_e']
			T = panel['Tp']  # Use nominal temperature if not specified
			G = panel['Gp']  # Use default irradiance if not specified
			K_V = panel['K_V']/100*V_oc
			K_I = panel['K_I']/100*I_sc
			Ns = panel['Ns']

			delT = T - self.Tn
			V_t = Ns * self.k * T / self.q  # Thermal voltage
			V_t_n = Ns * self.k * self.Tn / self.q  # Thermal voltage at 25 C
			a = 1.02  # Ideality factor (adjustable)
			
			print ("==== ====== ====== debugging area ==== ====== ======")
			print ("I_mp: ", I_mp)
			print ("I_sc: ", I_sc)
			print ("V_mp: ", V_mp)
			print ("V_oc: ", V_oc)
			print ("V_t_n: ", V_t_n)
			print ('exponential term: ', np.exp((V_oc) / (a * V_t_n)))
			print ('I0_n: ', (I_sc) / (np.exp((V_oc) / (a * V_t_n)) - 1))
			
			# Initial guesses
			R_s = 0.0  # Series resistance CHANGE THIS TO A HIGHER VALUE TO REDUCE TIME TO RUN CODE
			R_p = (V_mp / (I_sc - I_mp)) - ((V_oc - V_mp) / I_mp)  # Parallel resistance (initial guess)
			I0_n = (I_sc) / (np.exp((V_oc) / (a * V_t_n)) - 1)
			I0 = (I_sc + K_I * delT) / (np.exp((V_oc + K_V * delT) / (a * V_t)) - 1)
			error = 1000  # Initial error
			optimal_R_s = R_s  # Optimal series resistance
			iteration_count = 0  # Counter for iterations

			while error > tolerance:
				iteration_count += 1

				# Calculate parameters
				I_PV_n = ((R_p + R_s) / R_p) * I_sc
				I_PV = (I_PV_n + K_I * delT) * G / self.Gn
				R_p = V_mp * (V_mp + I_mp * R_s) / (
					V_mp * I_PV_n - V_mp * I0_n * np.exp((V_mp + I_mp * R_s) * self.q / (Ns * a * self.k * self.Tn)) +
					V_mp * I0_n - P_max_e)

				# Compute error
				V = np.linspace(0, V_oc, 1000)
				I = np.zeros_like(V)

				for i, v in enumerate(V):
					I_guess = I_sc  # Initial guess for current
					I[i] = fsolve(self._current_equation, I_guess, args=(v, I_PV_n, I0_n, R_s, V_t_n, a, R_p))

				P = V * I
				P_max = np.max(P)
				error = abs(P_max_e - P_max)

				# Update optimal R_s
				optimal_R_s = R_s
				R_s += 0.001  # Increment R_s

				print(error)

			# Store results
			results[(m, n)] = {
				"R_s": optimal_R_s,
				"R_p": R_p,
				"a": a,
				"I_PV": I_PV,
				"I0": I0,
				"error": error,
				"iterations": iteration_count,
				"Ns": Ns
			}

		return results
	
	def estimate_pv_parameters_E(self, tolerance):
		results = {}
		k = 1.380649e-23  # Boltzmann constant [J/K]
		q = 1.602e-19     # Electron charge [C]
		Tn = self.Tn 	  # Nominal temperature in Kelvin
		Gn = self.Gn
		a = 1.0  		# Diode ideality factor keep the same as the alpha_val for initialize_solar_farm func
		for (m, n), panel in self.panel_data.items():
			
			i_mp = panel['I_mp']
			i_sc = panel['I_sc']
			v_mp = panel['V_mp']
			v_oc = panel['V_oc']
			P_max_e = panel['P_max_e']
			T = panel['Tp']  # Use nominal temperature if not specified
			G = panel['Gp']  # Use default irradiance if not specified
			K_V = panel['K_V']/100*v_oc
			K_I = panel['K_I']/100*i_sc
			cells_in_series = panel['Ns']
			
			# Photo current adjusted for irradiance and temperature
			I_ph = i_sc * (1+ K_I * (T - Tn)) * (G / Gn)
			
			k = 1.380649e-23  # Boltzmann constant [J/K]
			q = 1.602176634e-19  # Elementary charge [C]
			V_th = a * k * T / q * cells_in_series
			V_th_n = a * k * Tn / q * cells_in_series  # Thermal voltage at nominal temperature

			I_0_ref = i_sc / (np.exp(v_oc / V_th_n) - 1)
			Eg = 1.12  # Bandgap energy for silicon in e
			I_0 = I_0_ref * (T/ Tn) ** 3 * np.exp(q*Eg/k * (1/Tn - 1/T))  # Adjusted for temperature
			print ("==== ====== ====== debugging area ==== ====== ======")
			# print ("I_mp: ", i_mp)
			# print ("I_sc: ", i_sc)
			# print ("V_mp: ", v_mp)
			# print ("V_oc: ", v_oc)
			print ("V_th_n: ", V_th_n)
			print ('exponential term: ', np.exp((v_oc) / (V_th_n)))
			print ('I_0: ', I_0)
			# exit()
			# I_ph = i_sc
			R_s = (V_th_n * np.log((I_ph - i_mp) / I_0_ref) - v_mp) / i_mp
			exp_term = np.exp((v_mp + i_mp * R_s) / V_th_n)
			R_sh = (v_mp + i_mp * R_s) / (I_ph - i_mp - I_0_ref * exp_term) # you need to use the I_ph and I_0_ref to calculate R_sh
			
			print(f"R_s: {R_s:.4f} Ohm")
			print(f"R_sh: {R_sh:.4f} Ohm")
			print(f"I_0: {I_0:.4e} A")
			
			results[(m, n)] = {
				"R_s": R_s,
				"R_p": R_sh,
				"a": a,
				"I_PV": I_ph,
				"I0": I_0,
				"error": None,
				"iterations": None,
				"Ns": cells_in_series
			}
		# exit()
		return results
	

class PVPanelDataProcessor:
	def __init__(self, results, q_val, alpha_val, K_val, T_val, R_Load_val, total_row, total_col):
		"""
		Initialize with the computed panel results and additional constants.
		:param results: Dictionary with (m, n) as keys and computed parameters as values.
		:param q_val: Charge of an electron (Coulombs).
		:param alpha_val: Diode ideality factor.
		:param K_val: Boltzmann constant (J/K).
		:param T_val: Temperature (K).
		:param Is_val: Reverse saturation current (A).
		:param R_Load_val: Load resistance.
		:param total_row: Total number of rows of panels.
		:param total_col: Total number of columns of panels.
		"""
		self.results = results
		self.q_val = q_val
		self.alpha_val = alpha_val
		self.K_val = K_val
		self.T_val = T_val
		self.R_Load_val = R_Load_val
		self.total_row = total_row
		self.total_col = total_col

	def generate_panel_data_list(self):
		"""
		Generate the list of panel data in the required format.
		:return: List of lists, each representing a panel's data.
		"""
		panel_data_list = []

		for (m, n), params in self.results.items():
			print(f"Panel ({m}, {n}): {params} estimated parameters")
			I_PV_val = params['I_PV'] # this is just Iph
			# R_S_val = params['R_s']
			# R_P_val = params['R_p']
			I_0_val = params['I0']
			Ns = params['Ns']

			# print("==== ====== ====== PE deactivated except I_PV_Val, R_S_val using data at PE line 238 ==== ====== ======")
			# I_PV_val = 10.24  # Example value, replace with actual calculation if needed
			R_S_val = 0.3719447111281037
			R_P_val = 8.072833897087926e+2
			# I_0_val = 2.44162749982378e-11
			# Ns = 72

			# Construct the list for this panel
			panel_data = [
				I_PV_val,
				R_S_val,
				R_P_val,
				self.q_val,
				# self.alpha_val*Ns, # why is this multiplied by Ns?
				self.alpha_val,
				self.K_val,
				self.T_val,
				I_0_val,
				m,  # Row index
				n,  # Column index
				self.R_Load_val,
				self.total_row,
				self.total_col
			]

			panel_data_list.append(panel_data)

		return panel_data_list
	