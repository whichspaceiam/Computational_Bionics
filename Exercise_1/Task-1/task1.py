from scipy.integrate import solve_ivp
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define numerical values
l_O_val = 0.410  # length of the thigh [m]
l_U_val = 0.415  # length of the shank [m]

m_body = 100  # mass of the body [kg]

mO_val = 9.73  # mass of the thigh [kg]
mU_val = 5.07  # mass of the shank [kg]
mF_val = 0.44  # mass of the foot [kg]

g_val = 9.81  # gravity [m/s^2]

# Define symbolic variables
l_O, l_U, q1, q2, w1, w2 = sp.symbols('l_O l_U q1 q2 w_1 w_2')
mO, mU, mF, g = sp.symbols('m_O m_U m_F g')

# Position vectors
r_SO = sp.Matrix([0.5 * l_O * sp.cos(q1), 0.5 * l_O * sp.sin(q1), 0])
r_SU = sp.Matrix([
    l_O * sp.cos(q1) + 0.5 * l_U * sp.cos(q2),
    l_O * sp.sin(q1) + 0.5 * l_U * sp.sin(q2),
    0
])
r_S = sp.Matrix([
    l_O * sp.cos(q1) + l_U * sp.cos(q2),
    l_O * sp.sin(q1) + l_U * sp.sin(q2),
    0
])

# Velocity vectors
v_SO = 0.5 * l_O * w1 * sp.Matrix([-sp.sin(q1), sp.cos(q1), 0])
v_SU = (l_O * w1 * sp.Matrix([-sp.sin(q1), sp.cos(q1), 0]) +
        0.5 * l_U * w2 * sp.Matrix([-sp.sin(q2), sp.cos(q2), 0]))
v_S = (l_O * w1 * sp.Matrix([-sp.sin(q1), sp.cos(q1), 0]) +
       l_U * w2 * sp.Matrix([-sp.sin(q2), sp.cos(q2), 0]))

# Kinetic energy
T1 = 0.5 * mO * v_SO.dot(v_SO)
T2 = 0.5 * mU * v_SU.dot(v_SU)
T3 = 0.5 * mF * v_S.dot(v_S)
T = T1 + T2 + T3
T = T.simplify()

# Potential energy
V1 = mO * g * r_SO[1]
V2 = mU * g * r_SU[1]
V3 = mF * g * r_S[1]
V = V1 + V2 + V3
V = V.simplify()

# Lagrangian
L = T - V

# Derivatives
dLdw1 = sp.diff(L, w1).simplify()
dLdw2 = sp.diff(L, w2).simplify()
dLdq1 = sp.diff(L, q1).simplify()
dLdq2 = sp.diff(L, q2).simplify()

# Time derivatives of dL/dw_i
w1_dot, w2_dot = sp.symbols('w1_dot w2_dot')
ddt_dLdw1 = sp.diff(dLdw1, q1) * w1 + sp.diff(dLdw1, q2) * w2 + sp.diff(dLdw1, w1) * w1_dot + sp.diff(dLdw1, w2) * w2_dot
ddt_dLdw2 = sp.diff(dLdw2, q1) * w1 + sp.diff(dLdw2, q2) * w2 + sp.diff(dLdw2, w1) * w1_dot + sp.diff(dLdw2, w2) * w2_dot

# Equations of motion
eq1 = ddt_dLdw1 - dLdq1

eq2 = ddt_dLdw2 - dLdq2

# Substitute constants
eq1 = eq1.subs({l_O: l_O_val, l_U: l_U_val, mO: mO_val, mU: mU_val, mF: mF_val, g: g_val})
eq2 = eq2.subs({l_O: l_O_val, l_U: l_U_val, mO: mO_val, mU: mU_val, mF: mF_val, g: g_val})

# Solve for angular accelerations
w1_dot_sol = sp.solve(eq1, w1_dot)[0]
w2_dot_sol = sp.solve(eq2, w2_dot)[0]

# Convert to numerical functions
domega1 = sp.lambdify((q1, q2, w1, w2), w1_dot_sol, 'numpy')
domega2 = sp.lambdify((q1, q2, w1, w2), w2_dot_sol, 'numpy')

# Read gait data
filename = 'gait_data.xls'
gait_data = pd.read_excel(filename, engine='xlrd')

# Extract gait data
gait_step = np.array(gait_data["gait_%"])
GRFz = np.array(gait_data["GRFz[%BW]"]) * m_body * g_val / 100
GRFx = gait_data["GRFx[%BW]"] * m_body * g_val / 100
MX_H = np.array(gait_data["MX_H[Nm/kg]"])
MX_K = np.array(gait_data["MX_K[Nm/kg]"])
q1_gait = np.deg2rad(np.array(gait_data["Flex_Ext_H[deg]"])) + 3 / 2 * np.pi
q2_gait = q1_gait - np.deg2rad(np.array(gait_data["Flex_Ext_K[deg]"]))

# Interpolate the moment data
MX_H_interp = interp1d(gait_step, MX_H, kind='cubic')
MX_K_interp = interp1d(gait_step, MX_K, kind='cubic')

# Time step
dt = 0.01
t_eval = np.arange(0, len(gait_step) * dt, dt)

# Initial conditions
q1_0 = q1_gait[0]
q2_0 = q2_gait[0]
omega1_0 = 0
omega2_0 = 0

# ODE system
def ode_system(t, y):
    q1, q2, omega1, omega2 = y

    dq1 = omega1
    dq2 = omega2
    domega1 = domega1(q1, q2, omega1, omega2)
    domega2 = domega2(q1, q2, omega1, omega2)

    return [dq1, dq2, domega1, domega2]

# Solve ODEs
sol = solve_ivp(ode_system, [0, len(gait_step) * dt], [q1_0, q2_0, omega1_0, omega2_0], t_eval=t_eval)

print("Done")
