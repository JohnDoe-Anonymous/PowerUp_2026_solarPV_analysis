from itertools import count
import random
from pyomo.environ import ConcreteModel, Var, Param, Constraint, Objective, RangeSet, SolverFactory, value, exp,maximize,minimize

class SolarFarmDefine:
    _idPanels = count(1)
    instances = []

    def __init__(self, I_PV, R_S, R_P, q, alpha, K, T, Is, row, column, R_Load, m, n):
        self.id = self._idPanels.__next__()
        self.I_PV = I_PV
        self.R_S = R_S
        self.R_P = R_P
        self.q = q
        self.alpha = alpha
        self.K = K
        self.T = T
        self.Is = Is
        self.row = row
        self.column = column
        self.R_Load = R_Load
        self.coordinate = (row, column)
        self.m = m
        self.n = n


class SolarFarmSolver:

    def __init__(self, paneldata, total_rows,total_columns):
        self.paneldata = paneldata
        self.total_rows = total_rows
        self.total_columns = total_columns
        self.no_of_cells = len(self.paneldata)

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
            R_Load_Col = cell.R_Load

        return I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col, R_Load_Col

    def initialize_model(self):
        I_PV_Col, R_S_Col, R_P_Col, q_Col, alpha_Col, K_Col, T_Col, Is_Col, R_Load_Col = self.extract_all_data()

        R_tot = R_Load_Col

        self.model.indices = RangeSet(1, len(I_PV_Col))
        self.model.I_indices = RangeSet(1, self.total_columns)

        # Initialize parameters as indexed
        self.model.I_PV_model = Param(self.model.indices, initialize=dict(enumerate(I_PV_Col, start=1)))
        self.model.Is_model = Param(self.model.indices, initialize=dict(enumerate(Is_Col, start=1)))
        self.model.q_model = Param(self.model.indices, initialize=dict(enumerate(q_Col, start=1)))
        self.model.R_P_model = Param(self.model.indices, initialize=dict(enumerate(R_P_Col, start=1)))
        self.model.R_S_model = Param(self.model.indices, initialize=dict(enumerate(R_S_Col, start=1)))
        self.model.alpha_model = Param(self.model.indices, initialize=dict(enumerate(alpha_Col, start=1)))
        self.model.K_model = Param(self.model.indices, initialize=dict(enumerate(K_Col, start=1)))
        self.model.T_model = Param(self.model.indices, initialize=dict(enumerate(T_Col, start=1)))
        self.model.R_Load_model = Param(initialize=R_tot)

        # Define variables
        # Define variables
        self.model.Vd = Var(self.model.indices, initialize=lambda model, i: random.uniform(-0.5, 0.5))
        self.model.VBatt = Var(self.model.indices, initialize=lambda model, i: random.uniform(-0.5, 0.5))
        self.model.I = Var(self.model.I_indices, initialize=lambda model, i: random.uniform(-0.5, 0.5))

        self.model.I_MPP = Var(initialize=random.uniform(-0.5, 0.5)) 

        # Add constraints here (updated according to the requirements)
        self.model.constraint1 = Constraint(self.model.indices, rule=self.equation1)
        self.model.constraint2 = Constraint(self.model.indices, rule=self.equation2)
        self.model.constraint3 = Constraint(rule=self.equation3)

        # Ensure that all VBatt values in the first row are equal
        first_row_indices = [i for i in self.model.indices if (i - 1) % self.total_rows + 1 ==1]
        self.model.equal_vbatt_constraint = Constraint(first_row_indices, rule=self.equal_vbatt)

        # Define THE OBJECTIVE FUNCTION
        self.model.obj = Objective(expr=self.model.VBatt[1] * self.model.I_MPP, sense=maximize)
        # self.model.obj = Objective(expr = 1)

    def equal_vbatt(self, model, i):
        """Enforce that all VBatt values in the first row are equal."""
        if i == self.model.indices.first():
            return Constraint.Skip  # Skip the first element as a reference
        return model.VBatt[i] == model.VBatt[self.model.indices.first()]

    def equation1(self, model, i):
        """Handle the case for V_Batt_next based on the position in the grid."""
        # Calculate current row and column based on the index i
        current_column = (i - 1) // self.total_rows + 1
        current_row = (i - 1) % self.total_rows + 1

        if current_row == self.total_rows:  # Last row
            V_Batt_next = 0

        else:  # Other rows
            next_index = i + 1
            if next_index in model.indices:
                V_Batt_next = model.VBatt[next_index]
            
        # Return the equation
        return (-model.I_PV_model[i] 
                + model.Is_model[i] * (exp(model.q_model[i] * (model.Vd[i] - V_Batt_next) / (model.alpha_model[i] * model.K_model[i] * model.T_model[i])) - 1)
                + (model.Vd[i] - V_Batt_next) / model.R_P_model[i] 
                + model.I[current_column] == 0)
    
    def equation2(self, model, i):

        # Calculate current row and column based on the index i
        current_column = (i - 1) // self.total_rows + 1

        return (model.VBatt[i] - model.Vd[i]) / model.R_S_model[i] + model.I[current_column] == 0
    
    def equation3(self, model):
        """Constraint for the sum of all I values minus VBatt[1] / R_Load_model equals zero."""
        # Calculate the sum of all I values
        I_sum = sum(model.I[i] for i in model.I_indices)
        
        # Return the constraint expression
        return I_sum - model.I_MPP == 0
    


    def sum_values(self, value_dict):
        """Sum the values in a dictionary."""
        return sum(value_dict.values())
    
    def solve(self):
        # Create a solver
        solver = SolverFactory('ipopt')
        solver.options['print_level'] = 5  # Suppress output

        # Solve the model
        result = solver.solve(self.model, tee=True)

        # Extract and return the values of indexed variables
        Vd_values = {i: value(self.model.Vd[i]) for i in self.model.indices}
        VBatt_values = {i: value(self.model.VBatt[i]) for i in self.model.indices}
        I_values = {i: value(self.model.I[i]) for i in self.model.I_indices}
        I_MPP_value = value(self.model.I_MPP)

        # Calculate sums
        I_sum = self.sum_values(I_values)

        return Vd_values, VBatt_values, I_values, I_sum, I_MPP_value