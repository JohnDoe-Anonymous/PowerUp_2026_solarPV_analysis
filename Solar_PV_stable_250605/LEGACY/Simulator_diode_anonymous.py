from itertools import count
import random
from pyomo.environ import ConcreteModel, Var, Param, Constraint, Objective, RangeSet, SolverFactory, value, exp,maximize,minimize
from pyomo.core.expr.visitor import identify_variables

class SolarFarmDefine:
    _idPanels = count(1)
    instances = []

    def __init__(self, I_PV, R_S, R_P, q, alpha, K, T, Is, row, column, Z_Load, m, n):
        self.id = self._idPanels.__next__()
        self.I_PV = I_PV
        self.R_S = R_S
        self.R_P = R_P
        self.q = q
        self.alpha = alpha
        self.K = K
        self.T = T
        self.Is = Is # this is the reverse saturation current I0
        self.row = row
        self.column = column
        self.Z_Load = Z_Load
        self.coordinate = (row, column)
        self.m = m
        self.n = n


class SolarFarmSolverSim:

    def __init__(self, paneldata, total_rows,total_columns,initial_guess, find_MPP=False):
        self.paneldata = paneldata
        self.total_rows = total_rows
        self.total_columns = total_columns
        self.no_of_cells = len(self.paneldata)
        self.initial_guess = initial_guess
        self.find_MPP = find_MPP
        # Create a Pyomo model
        self.model = ConcreteModel()

        # Initialize the model
        self.initialize_model()

    def extract_all_data(self):
        I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col = [], [], [], [], [], [], [], []

        for cell in self.paneldata:
            I_PV_Col.append(cell.I_PV)
            R_S_Col.append(cell.R_S)
            R_P_Col.append(cell.R_P)
            q_Col.append(cell.q)
            alpha_Col.append(cell.alpha)
            K_Col.append(cell.K)
            T_Col.append(cell.T)
            Is_Col.append(cell.Is)
            Z_Load_Col = cell.Z_Load

        return I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col, Z_Load_Col

    def initialize_model(self):
        I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col, Z_Load_Col = self.extract_all_data()

        R_tot = Z_Load_Col

        self.model.indices = RangeSet(1, len(I_PV_Col))
        self.model.I_indices = RangeSet(1, self.total_columns)

        # Initialize parameters as indexed
        self.model.I_PV_model = Param(self.model.indices, initialize=dict(enumerate(I_PV_Col, start=1))) # I_ph
        self.model.Is_model = Param(self.model.indices, initialize=dict(enumerate(Is_Col, start=1)))
        self.model.q_model = Param(self.model.indices, initialize=dict(enumerate(q_Col, start=1)))
        self.model.R_P_model = Param(self.model.indices, initialize=dict(enumerate(R_P_Col, start=1)))
        self.model.R_S_model = Param(self.model.indices, initialize=dict(enumerate(R_S_Col, start=1)))
        self.model.alpha_model = Param(self.model.indices, initialize=dict(enumerate(alpha_Col, start=1)))
        self.model.K_model = Param(self.model.indices, initialize=dict(enumerate(K_Col, start=1)))
        self.model.T_model = Param(self.model.indices, initialize=dict(enumerate(T_Col, start=1)))
        self.model.Z_Load_model = Param(initialize=R_tot)

        # Define variables
        # Define variables
        
        # if self.find_MPP:
        #     std_Vd = 200
        #     std_VBatt = 180
        #else:
        std_Vd = 0.01
        std_VBatt = 0.02
        
        # TODO: How do I use the last step solution as initalization?
        if self.initial_guess is None:
            self.model.Vd = Var(self.model.indices, initialize=lambda model, i: (self.total_rows - ((i - 1) % self.total_rows)) * std_Vd,bounds=(0,500))
            self.model.VBatt = Var(self.model.indices, initialize=lambda model, i: (self.total_rows - ((i - 1) % self.total_rows)) * std_VBatt,bounds=(0,500))
            self.model.I = Var(self.model.I_indices, initialize=lambda model, i: random.uniform(0.1, 10.0))
            self.model.I_MPP = Var(initialize=random.uniform(0.1, 10.0))
        else:
            self.model.Vd = Var(self.model.indices, initialize=lambda model, i: self.initial_guess['Vd'][i-1],bounds=(-100,700))
            self.model.VBatt = Var(self.model.indices, initialize=lambda model, i: self.initial_guess['VBatt'][i-1],bounds=(-100,700))
            self.model.I = Var(self.model.I_indices, initialize=lambda model, i: self.initial_guess['I'][i-1],bounds=(-100,700))
            self.model.I_MPP = Var(initialize=self.initial_guess['I_MPP'])

        # Add constraints here (updated according to the requirements)
        self.model.constraint1 = Constraint(self.model.indices, rule=self.equation1)
        self.model.constraint2 = Constraint(self.model.indices, rule=self.equation2)
        self.model.constraint3 = Constraint(rule=self.equation3)
        if self.find_MPP:
            pass
        else:
            self.model.constraint4 = Constraint(rule=self.equation4)

        # Ensure that all VBatt values in the first row are equal
        first_row_indices = [i for i in self.model.indices if (i - 1) % self.total_rows + 1 ==1]
        self.model.equal_vbatt_constraint = Constraint(first_row_indices, rule=self.equal_vbatt)
        # Define THE OBJECTIVE FUNCTION
        if self.find_MPP:
            self.model.obj = Objective(expr=self.model.VBatt[1] * self.model.I_MPP, sense=maximize)
        else:
            self.model.obj = Objective(expr = 1)

    def equal_vbatt(self, model, i):
        """Enforce that all VBatt values in the first row are equal."""
        if i == self.model.indices.first():
            return Constraint.Skip  # Skip the first element as a reference
        return model.VBatt[i] == model.VBatt[self.model.indices.first()]

    from pyomo.environ import exp

    def equation1(self, model, i):
        """Safe Node KCL at panel diode (PV model)"""

        current_column = (i - 1) // self.total_rows + 1
        current_row = (i - 1) % self.total_rows + 1

        if current_row == self.total_rows:
            V_Batt_next = 0
        else:
            next_index = i + 1
            if next_index in model.indices:
                V_Batt_next = model.VBatt[next_index]

        model.V_t = model.K_model[i] * model.T_model[i] / model.q_model[i]
        V_t_alpha = model.alpha_model[i] * model.V_t * 72 # 72 is the number of cells in series, this should be a parameter


        V_diff = model.Vd[i] - V_Batt_next
        
        Rp_safe = model.R_P_model[i]

        print("V_diff", V_diff)
        print("V_t_alpha^1", 1/V_t_alpha)
        
        exp_argument = V_diff / (V_t_alpha)
        diode_current = model.Is_model[i] * (exp(exp_argument) - 1)
        Rp_current = V_diff / Rp_safe # Rp current
        Rs_current = (model.Vd[i] - model.VBatt[i]) / model.R_S_model[i]

        return (-model.I_PV_model[i] + diode_current + Rp_current + Rs_current) == 0

    
    def equation2(self, model, i):
        """Safe Node KCL at panel junction (VBatt node)"""

        col = (i - 1) // self.total_rows + 1
        row = (i - 1) % self.total_rows + 1

        if row == self.total_rows:
            V_next = 0
        else:
            V_next = model.VBatt[i + 1]

        R_S = model.R_S_model[i]

        I_series = (model.Vd[i] - model.VBatt[i]) / R_S
        
        # V_diff_block = (V_next - model.VBatt[i]) 
        model.V_t = model.K_model[i] * model.T_model[i] / model.q_model[i]
        V_t_alpha = model.alpha_model[i] * model.V_t * 72 # 72 is the number of cells in series, this should be a parameter
        # V_t_alpha = 0.09
        # V_t_alpha = 1e10
        print("V_t_alpha^-1", 1/V_t_alpha)
        exp_argument_block = (V_next - model.VBatt[i]) / (V_t_alpha) # V_next is V_Batt[i+1], if only one row, V_next = V_G = 0 
        # exp_argument_block = max(min(exp_argument_block, 100), -100)

        # I0_block = 8.8247e-15
        I0_block = 1.0e-3
        I_Block_Diode = I0_block * (exp(exp_argument_block) - 1)
        
        
        return -I_series - I_Block_Diode + model.I[col] == 0
        # return -I_series + model.I[col] == 0

    def equation3(self, model):
        """Constraint for the sum of all I values minus VBatt[1] / Z_Load_model equals zero."""
        # Calculate the sum of all I values
        I_sum = sum(model.I[i] for i in model.I_indices)
        # Return the constraint expression
        return I_sum - model.I_MPP == 0

    def equation4(self, model):
        """Constraint linking Z_Load to VBatt[1] and I_MPP."""
        # if not, Z_Load_model is a parameter and VBatt[1] is the voltage at the first panel
        # this is actually R not Z
        return model.VBatt[1] - model.Z_Load_model * model.I_MPP == 0


    def sum_values(self, value_dict):
        """Sum the values in a dictionary."""
        return sum(value_dict.values())
    
    def check_initial_constraints(self):
        """Evaluate all constraints at initial point."""
        print("\n--- Checking initial constraint residuals ---")
        max_violation = 0
        for c in self.model.component_objects(Constraint, active=True):
            cobject = getattr(self.model, c.name)
            for index in cobject:
                try:
                    body_val = value(cobject[index].body)
                    if cobject[index].equality:
                        residual = abs(body_val)
                    else:
                        residual = max(
                            0,
                            value(cobject[index].lower) - body_val if cobject[index].lower is not None else 0,
                            body_val - value(cobject[index].upper) if cobject[index].upper is not None else 0
                        )
                    print(f"Constraint {c.name}{index}: residual = {residual:.6e}")
                    if residual > max_violation:
                        max_violation = residual
                except Exception as e:
                    print(f"Constraint {c.name}{index}: Error evaluating - {e}")

        print(f"\nMaximum constraint violation at initial point = {max_violation:.6e}")
        print("--- Done checking initial constraints ---\n")

    def check_constraint_residuals(self):
        """Check constraint violations after solver crash."""
        print("\n--- Checking constraint violations ---")
        for c in self.model.component_objects(Constraint, active=True):
            cname = c.name
            cobj = getattr(self.model, cname)
            for index in cobj:
                try:
                    expr = cobj[index].expr
                    
                    body_val = value(cobj[index].body)
                    if cobj[index].equality:
                        residual = abs(body_val)
                    else:
                        residual = max(
                            0,
                            value(cobj[index].lower) - body_val if cobj[index].lower is not None else 0,
                            body_val - value(cobj[index].upper) if cobj[index].upper is not None else 0
                        )
                    print(f"Constraint {cname}[{index}]: Residual = {residual:.4e}")
                    print(f"{cname}[{index}]  -->  {expr}")
                    print(f"Constraint {cname}[{index}]: R = {residual:.4e}")
                except Exception as e:
                    print(f"Constraint {cname}[{index}]: Error evaluating constraint - {e}")
        print("--- End of constraint check ---\n")
    
    def debug_constraints_with_values(self):
        """Print constraint expression, residual, and involved variable values."""
        print("\n--- Constraint Debug Report ---")
        for c in self.model.component_objects(Constraint, active=True):
            cname = c.name
            cobj = getattr(self.model, cname)

            for index in cobj:
                try:
                    con = cobj[index]
                    expr = con.expr
                    residual = abs(value(con.body)) if con.equality else "?"

                    print(f"\n Constraint {cname}[{index}]:")
                    print(f"  Expression: {expr}")
                    print(f"  Residual: {residual}")

                    # --- Extract and print variable values in the expression ---
                    print("  Variables used:")
                    for var in identify_variables(expr):
                        vname = var.name
                        vval = value(var)
                        print(f"{vname} = {vval:.6f}")

                except Exception as e:
                    print(f" Constraint {cname}[{index}]: Error evaluating - {e}")

        print("\n --- End of Constraint Debug Report ---\n")

    def solve(self):
        # Create a solver
        solver = SolverFactory('ipopt')
        solver.options['print_level'] = 5  # Suppress output
        solver.options['max_iter'] = 1000
        solver.options['tol'] = 1e-4
        solver.options["halt_on_ampl_error"] = "yes"

        # Solve the model
        for v in self.model.component_data_objects(ctype=Var):
            print(f"{v.name} = {v.value}")
        
        self.check_initial_constraints()
        
        # result = solver.solve(self.model, tee=True)

        # # Extract and return the values of indexed variables
        # Vd_values = {i: value(self.model.Vd[i]) for i in self.model.indices}
        # VBatt_values = {i: value(self.model.VBatt[i]) for i in self.model.indices}
        # I_values = {i: value(self.model.I[i]) for i in self.model.I_indices}
        # I_MPP_value = value(self.model.I_MPP)

        # # Calculate sums
        # I_sum = self.sum_values(I_values)

        # return Vd_values, VBatt_values, I_values, I_sum, I_MPP_value
        try:
            result = solver.solve(self.model, tee=True)
            # Extract and return the values of indexed variables
            Vd_values = {i: value(self.model.Vd[i]) for i in self.model.indices}
            VBatt_values = {i: value(self.model.VBatt[i]) for i in self.model.indices}
            I_values = {i: value(self.model.I[i]) for i in self.model.I_indices}
            I_MPP_value = value(self.model.I_MPP)

            # Calculate sums
            I_sum = self.sum_values(I_values)
            print("Solver finished successfully.")
            # save solution for next initial guess
            initial_guess = {
                'Vd': [value(self.model.Vd[i]) for i in self.model.indices],
                'VBatt': [value(self.model.VBatt[i]) for i in self.model.indices],
                'I': [value(self.model.I[i]) for i in self.model.I_indices],
                'I_MPP': value(self.model.I_MPP)
            }
            return Vd_values, VBatt_values, I_values, I_sum, I_MPP_value, initial_guess
        except Exception as e:
            print("Solver failed with error:", e)
            print("\n Now checking constraint residuals at crash point...")
            self.debug_constraints_with_values()
            raise e  # re-raise error after checking (optional)