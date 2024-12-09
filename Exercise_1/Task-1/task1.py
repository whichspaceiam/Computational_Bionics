# Exercise 1 - Task 1
# Modeling the Leg as an ODE

import numpy as np
import sympy as sp
import pandas as pd

# define symbolic variables
l_O, l_U, q1, q2, w1, w2, t = sp.symbols('l_O l_U q1 q2 w_1  w_2 t')
mO, mU, mF = sp.symbols('m_O m_U m_F')
g = sp.symbols('g')

q1 = sp.Function('q1')(t)
q2 = sp.Function('q2')(t)

w1 = sp.Function('w_1')(t)
w2 = sp.Function('w_2')(t)

# Position vectors
r_SO = sp.Matrix([0.5 * l_O * sp.cos(q1), 0.5 * l_O *sp.sin(q1), 0])
r_SU = sp.Matrix([l_O * sp.cos(q1) + 0.5 * l_U * sp.cos(q2), l_O * sp.sin(q1) + 0.5 * l_U * sp.sin(q2), 0])
r_S = sp.Matrix([l_O * sp.cos(q1) + l_U * sp.cos(q2), l_O * sp.sin(q1) + l_U * sp.sin(q2), 0])

# Define velocity vectors
v_SO = 0.5 * l_O * w1 * sp.Matrix([[-sp.sin(q1)], [sp.cos(q1)], [0]])
v_SU = l_O * w1 * sp.Matrix([[-sp.sin(q1)], [sp.cos(q1)], [0]]) + 0.5 * l_U * w2 * sp.Matrix([[-sp.sin(q2)], [sp.cos(q2)], [0]])
v_S = l_O * w1 * sp.Matrix([[-sp.sin(q1)], [sp.cos(q1)], [0]]) + l_U * w2 * sp.Matrix([[-sp.sin(q2)], [sp.cos(q2)], [0]])

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

# Derivatives: dL / dw_i
dLdw1 = sp.diff(L, w1).simplify()
dLdw2 = sp.diff(L, w2).simplify()

# Derivatives: d/dt (dL / dw_i)
ddt_dLdw1 = sp.diff(dLdw1, t)
ddt_dLdw2 = sp.diff(dLdw2, t)

# Derivatives: dL / dq_i
dLdq1 = sp.diff(L, q1)
dLdq2 = sp.diff(L, q2)

M_H, M_K, w1_dot, w2_dot  = sp.symbols(r'M_H M_K \dot{w}_1 \dot{w}_2')

# Equations of motion for q1 and q2
eq1 = ddt_dLdw1 - dLdq1 - M_H
eq2 = ddt_dLdw2 - dLdq2 - M_K

eq1 = eq1.subs(w1.diff(t), w1_dot).subs(w2.diff(t), w2_dot)
eq2 = eq2.subs(w2.diff(t), w2_dot).subs(w1.diff(t), w1_dot)

eq1 = eq1.simplify()
eq2 = eq2.simplify()

w1_dot = sp.solve(eq1, w1_dot, dict=True, simplify=True, rational=True)
w2_dot = sp.solve(eq2, w2_dot, dict=True, simplify=True, rational=True)

print(w1_dot)