from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

def make_xt_attractors():
    system = 3
    if system == 1:
        ic = lorenz_default_IC = [0, 1, 0] # 50
        lim = lorenz_default_lim = 50
        rhs = lorenz_default_maker()
    elif system == 2:
        ic = lorenz_new_IC = [1, 1, 1] # 50
        lim = lorenz_new_lim = 50
        rhs = lorenz_new_maker()
    else:
        ic = rossler_default_IC = [1, 1, 0] # 350
        lim = rossler_default_lim = 350
        rhs = ressler_default_maker()
    

    sol = solve_ivp(rhs, [0, lim], ic, rtol=1e-12, atol=1e-12)

    xs, ys, zs = sol.y
    ts = sol.t

    plt.figure(figsize=[10, 2])
    plt.subplots_adjust(left=0.0825, right=0.975, top=0.960, bottom=0.275)
    plt.plot(ts, xs)
    plt.xlim(0, lim)
    # plt.grid()
    plt.xlabel('t', fontsize=17)
    plt.ylabel('x', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()

    # plt.plot(xs, ys)
    # plt.grid()
    # plt.show()

make_xt_attractors()
