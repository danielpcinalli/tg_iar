from sympy import symbols, trigsimp
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.printing.latex import latex
from sympy.printing.pretty.pretty import pretty
from sympy import pi

t1, t2, t3, t4, t5 = symbols('theta1 theta2 theta3 theta4 theta5')
ti, al, a, d = symbols('theta_i alpha a d')

al1, al2, al3, al4, al5 = pi/2, 0, pi/2, pi/2, 0
d1, d2, d3, d4, d5 = .2305, 0, 0, .2225, 0
a1, a2, a3, a4, a5 = 0, .2211, 0, 0, .1488


T = Matrix([[cos(ti),         -sin(ti),        0,        a], 
            [sin(ti)*cos(al), cos(ti)*cos(al), -sin(al), -sin(al)*d],
            [sin(ti)*sin(al), cos(ti)*sin(al), cos(al),  cos(al)*d],
            [0,               0,               0,        1]])

print(pretty(T))

T_01 = T.subs(ti, t1).subs(al, al1).subs(d, d1).subs(a, a1)
T_12 = T.subs(ti, t2).subs(al, al2).subs(d, d2).subs(a, a2)
T_23 = T.subs(ti, t3).subs(al, al3).subs(d, d3).subs(a, a3)
T_34 = T.subs(ti, t4).subs(al, al4).subs(d, d4).subs(a, a4)
T_45 = T.subs(ti, t5).subs(al, al5).subs(d, d5).subs(a, a5)

print(pretty(T_01))
print(pretty(T_12))
print(pretty(T_23))
print(pretty(T_34))
print(pretty(T_45))

T_05 = trigsimp(T_01*T_12*T_23*T_34*T_45)

r11 = T_05[0,0]
r12 = T_05[0,1]
r13 = T_05[0,2]
r21 = T_05[1,0]
r22 = T_05[1,1]
r23 = T_05[1,2]
r31 = T_05[2,0]
r32 = T_05[2,1]
r33 = T_05[2,2]
px =  T_05[0,3]
py =  T_05[1,3]
pz =  T_05[2,3]

print("Results")
print(pretty(r11))
print(pretty(r12))
print(pretty(r13))
print(pretty(r21))
print(pretty(r22))
print(pretty(r23))
print(pretty(r31))
print(pretty(r32))
print(pretty(r33))
print(pretty(px))
print(pretty(py))
print(pretty(pz))

print("Latex")
print(latex(r11))
print(latex(r12))
print(latex(r13))
print(latex(r21))
print(latex(r22))
print(latex(r23))
print(latex(r31))
print(latex(r32))
print(latex(r33))
print(latex(px))
print(latex(py))
print(latex(pz))