# Imports
from scipy.integrate import solve_ivp
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# define numerical values
l_O_val = 0.410 # length of the tigh [m]
l_U_val = 0.415 # length of the shank [m]

m_body = 100 # mass of the body [kg]

mO_val = 9.73 # mass of the thigh [kg]
mU_val = 5.07 # mass of the shank [kg]
mF_val = 0.44 # mass of the foot [kg]

g_val = 9.81 # gravity [m/s^2]

# define symbolic variables
l_O, l_U, q1, q2, t = sp.symbols('l_O l_U q1 q2 t')
omega1, omega2 = sp.symbols(r'\omega_1 \omega_2')
omega1_dot, omega2_dot = sp.symbols(r'\dot{\omega}_1 \dot{\omega}_2')
mO, mU, mF = sp.symbols('m_O m_U m_F')
g = sp.symbols('g')

# define angles q as a function of time
q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)

# define angular velocities w as the derivative of the angles
w1 = q1.diff(t)
w2 = q2.diff(t)

# define angular accelerations as the derivative of the angular velocities
dot_w1 = w1.diff(t)
dot_w2 = w2.diff(t)

# Position vectors
r_SO = sp.Matrix([0.5 * l_O * sp.cos(q1), 0.5 * l_O *sp.sin(q1), 0])
r_SU = sp.Matrix([l_O * sp.cos(q1) + 0.5 * l_U * sp.cos(q2), l_O * sp.sin(q1) + 0.5 * l_U * sp.sin(q2), 0])
r_S = sp.Matrix([l_O * sp.cos(q1) + l_U * sp.cos(q2), l_O * sp.sin(q1) + l_U * sp.sin(q2), 0])

# Define velocity vectors
v_SO = r_SO.diff(t)
v_SU = r_SU.diff(t)
v_S = r_S.diff(t)

# Kinetic energy
T1 = 0.5 * mO * v_SO.dot(v_SO)
T2 = 0.5 * mU * v_SU.dot(v_SU)
T3 = 0.5 * mF * v_S.dot(v_S)
T = T1 + T2 + T3

# Potential energy
V1 = mO * g * r_SO[1]
V2 = mU * g * r_SU[1]
V3 = mF * g * r_S[1]
V = V1 + V2 + V3

# Lagrangian
L = T - V

# Derivatives: dL / d(dot_q_i)
dL_domega_1 = sp.diff(L, q1.diff(t))
dL_domega_2 = sp.diff(L, q2.diff(t))

# Time derivatives of the dL/d(dot_q_i)
dL_domega_1_dt = dL_domega_1.diff(t)
dL_domega_2_dt = dL_domega_2.diff(t)

# Derivatives: dL / dq_i
dL_dq_1 = sp.diff(L, q1)
dL_dq_2 = sp.diff(L, q2)

# Substitute the values of the parameters
subsDict = {q1.diff(t): omega1,
            q2.diff(t): omega2,
            q1.diff(t, 2): omega1_dot,
            q2.diff(t, 2): omega2_dot}

dL_domega_1_dt = dL_domega_1_dt.subs(subsDict).simplify()
dL_domega_2_dt = dL_domega_2_dt.subs(subsDict).simplify()
dL_dq_1 = dL_dq_1.subs(subsDict).simplify()
dL_dq_2 = dL_dq_2.subs(subsDict).simplify()

M1, M2 = sp.symbols('M_1 M_2')

# Equations of motion
eq1 = dL_domega_1_dt - dL_dq_1 - M1
eq2 = dL_domega_2_dt - dL_dq_2 - M2

# Solve for omega1_dot and omega2_dot
sol = sp.solve([eq1, eq2], (omega1_dot, omega2_dot))

dot_omega1 = sol[omega1_dot].simplify()
dot_omega2 = sol[omega2_dot].simplify()

# Define the functions
func1 = sp.lambdify((q1, q2, omega1, omega2, M1, M2), dot_omega1, 'numpy')
func2 = sp.lambdify((q1, q2, omega1, omega2, M1, M2), dot_omega2, 'numpy')

# Read gait data
filename = 'gait_data.xls'
gait_data = pd.read_excel(filename, engine='xlrd')

# Extract gait data
gait_step = np.array(gait_data["gait_%"]) / 100
GRFz = np.array(gait_data["GRFz[%BW]"]) * m_body * g_val / 100
GRFx = gait_data["GRFx[%BW]"] * m_body * g_val / 100
MX_H = np.array(gait_data["MX_H[Nm/kg]"]) * m_body
MX_K = np.array(gait_data["MX_K[Nm/kg]"]) * m_body
q1_gait = np.deg2rad(np.array(gait_data["Flex_Ext_H[deg]"])) # + 3 / 2 * np.pi
q2_gait = np.deg2rad(np.array(gait_data["Flex_Ext_K[deg]"]))
# q2_gait = q1_gait - np.deg2rad(np.array(gait_data["Flex_Ext_K[deg]"]))

dt = 0.01
t_start = 0
t_end = 1
t_eval = np.arange(t_start, t_end, dt)

# initial conditions
q1_0 = 3/2 * np.pi #q1_gait[0] # initial angle of the thigh
q2_0 = q1_0 #q2_gait[0] # initial angle of the shank

omega1_0 = 0
omega2_0 = 0

M1_0 = 0
M2_0 = 0

y0 = [omega1_0, omega2_0, q1_0, q2_0, M1_0, M2_0]

# Define the function for the ODE solver
def leg_model(t, y):
    omega1, omega2, q1, q2, M1, M2 = y

    # Compute joint moments
    M1 = np.interp(t, gait_step, MX_H)
    M2 = np.interp(t, gait_step, MX_K)

    domega1 = func1(q1, q2, omega1, omega2, M1, M2)
    domega2 = func2(q1, q2, omega1, omega2, M1, M2)

    dq1 = omega1
    dq2 = omega2

    dM1 = -1 * omega1
    dM2 = -1 * omega2

    return [domega1, domega2, dq1, dq2, M1, M2]

solution = solve_ivp(leg_model, (t_start, t_end), y0, t_eval=t_eval, method='RK45')

t = solution.t
omega1 = solution.y[0]
omega2 = solution.y[1]
q1 = solution.y[2] 
q2 = solution.y[3] 
M1 = solution.y[4]
M2 = solution.y[5]

# use subplots to plot the results for q, omega, and M
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# add title
fig.suptitle('Leg model', fontsize=16, fontweight='bold', y=0.92)

axs[0].plot(t, q1, label='q1')
axs[0].plot(t, q2, label='q2')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Angle [rad]')
axs[0].grid()
axs[0].legend()

axs[1].plot(t, omega1, label='omega1')
axs[1].plot(t, omega2, label='omega2')
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Angular velocity [rad/s]')
axs[1].grid()
axs[1].legend()

axs[2].plot(t, M1, label='M1')
axs[2].plot(t, M2, label='M2')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Torque [Nm]')
axs[2].grid()
axs[2].legend()

fig.get_tight_layout()
plt.show()
