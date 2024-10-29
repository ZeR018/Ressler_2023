from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import time

# def d(x1, y1, z1, x2, y2, z2):

def lorenz_system_maker(sigma = 10, b = 8/3, d = 0.1, rp = [166.1, 166.12]):
    
    r1 = rp[0]
    r2 = rp[1]
    def lorenz_system(t, r):
        # print(t)
        x1 = r[0]
        y1 = r[1]
        z1 = r[2]
        x2 = r[3]
        y2 = r[4]
        z2 = r[5]

        res_arr = []

        dx1 = sigma * (y1 - x1)
        dy1 = r1 * x1 - y1 - x1*z1 + d * (y2 - y1)
        dz1 = - b * z1 + x1 * y1

        dx2 = sigma * (y2 - x2)
        dy2 = r2 * x2 - y2 - x2*z2 + d * (y1 - y2)
        dz2 = - b * z2 + x2 * y2

        res_arr.append(dx1)
        res_arr.append(dy1)
        res_arr.append(dz1)
        res_arr.append(dx2)
        res_arr.append(dy2)
        res_arr.append(dz2)
        return res_arr

    return lorenz_system

def make_lorenz_experiment():
    IC = [5, 5., 5., 
          1., 1., 1.]
    
    lorenz_system = lorenz_system_maker(d = 0.1)
    
    sol = solve_ivp(lorenz_system, [0, 30], IC, 'RK45', rtol=1e-11, atol=1e-11)

    xs1, ys1, zs1, xs2, ys2, zs2 = sol.y
    ts = sol.t

    plt.figure(figsize=[20, 6])
    plt.plot(ts, xs1, label='x1')
    plt.plot(ts, xs2, label='x2')
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Two Lorenz systems, x(t)')
    plt.show()

    plt.plot(xs1, ys1)
    plt.plot(xs2, ys2)
    plt.show()



make_lorenz_experiment()