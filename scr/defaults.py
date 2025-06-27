from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def lorenz_default_maker(sigma = 10, ro = 28, beta = 8/3):
    def rhs(t, X):
        x, y, z = X
        dx = sigma * (y - x)
        dy = x * (ro - z) - y
        dz = x*y - beta*z
        return [dx, dy, dz]
    return rhs

def lorenz_new_maker(sigma = 10, b = 8/3, r = 166.1):
    def rhs(t, X):
        x, y, z = X
        dx = sigma * (y - x)
        dy = r * x - y - x*z
        dz = - b * z + x * y
        return [dx, dy, dz]
    return rhs

def ressler_default_maker(a=0.22, b=0.1, c=8.5):
    def rhs(t, X):
        x, y, z = X
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return [dx, dy, dz]
    return rhs

def make_xt_attractors(system = 3):
    if system == 1 or system == 4:
        ic = lorenz_default_IC = [5, 1, 25] # 50
        lim = lorenz_default_lim = 30
        rhs = lorenz_default_maker()
    elif system == 2:
        ic = lorenz_new_IC = [1, 1, 100] # 50
        lim = lorenz_new_lim = 40
        rhs = lorenz_new_maker()
    else:
        ic = rossler_default_IC = [1, 1, 0] # 350
        lim = rossler_default_lim = 350
        rhs = ressler_default_maker()
    

    sol = solve_ivp(rhs, [0, lim], ic, rtol=1e-12, atol=1e-12)

    xs, ys, zs = sol.y
    ts = sol.t

    plt.figure(figsize=[4, 4])
    plt.subplots_adjust(left=0.26, right=0.97, top=0.960, bottom=0.19)
    if system == 3:
        plt.plot(xs, ys)
        plt.xlabel('x', fontsize=25)
        plt.ylabel('y', fontsize=25)
    elif system == 1:
        plt.plot(xs, ys)
        plt.xlabel('x', fontsize=25)
        plt.ylabel('y', fontsize=25)
    if system == 4:
        u_arr = []
        for i in range(len(ts)):
            u_arr.append(np.sqrt(xs[i]**2+ys[i]**2))
        plt.plot(u_arr, zs)
        plt.xlabel('u', fontsize=25)
        plt.ylabel('z', fontsize=25)
    if system == 2:
        du_arr = []
        dz_arr = []
        for i in range(len(ts)):
            dx = 10 * (ys[i]-xs[i])
            dy = 166.1 * xs[i] - ys[i] - xs[i] * zs[i]
            chisl = xs[i] * dx + ys[i] * dy
            zn = (np.sqrt(xs[i]**2+ys[i]**2))
            du_arr.append(chisl/zn)

            dz_arr.append(xs[i]*ys[i] - 8/3 * zs[i])
        plt.plot(du_arr, dz_arr)
        plt.xlabel("u'", fontsize=25)
        plt.ylabel("'z'", fontsize=25)

    
    # plt.xlim(0, lim)
    # plt.grid()
    plt.axis('equal')
    plt.xticks([-10, 0, 10], fontsize=20)
    plt.yticks([-10, 0, 10], fontsize=20)

    plt.show()

    # plt.plot(xs, ys)
    # plt.grid()
    # plt.show()

make_xt_attractors()
