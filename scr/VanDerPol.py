from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import memory_worker as mem
import colorama

colorama.init()

def phase_converter(phase):
    if phase > 0:
        if phase > np.pi:
            phase = phase - 2 * np.pi
            phase_converter(phase)
        return phase
    else:
        if phase < - np.pi:
            phase = phase + 2 * np.pi
            phase_converter(phase)
        return phase

def solver(IC = [1.9, 0.3, 2.1, -0.5], t_max = 5000):
    class vdp_params():
        def __init__(self, mu = 0.001, beta = 0, gamma = 0., delta = 0.1, l = 0.):
            self.mu = mu
            self.delta = delta
            self.beta = beta
            self.gamma = gamma
            self.l = l

    def func_vdp_2_maker(p : vdp_params):

        def dissipative_coup(beta, y_this, y_other):
            return beta * (y_other - y_this)
        
        def divide_coup(l, y_this, y_other):
            return l / (y_this - y_other)

        def func_vdp_2(t, r):
            print(f'\033[F\033[KCurrent integrate time: {round(t, 1)};')
            x1, y1, x2, y2 = r
            
            dx1 = y1
            dy1 = p.mu * (1. - x1**2) * y1 - x1 + p.mu * divide_coup(p.l, y2, y1)
            dx2 = y2
            dy2 = p.mu * (1 + p.gamma - x2**2) * y2 - (1 + p.mu * p.delta) * x2 + p.mu * divide_coup(p.l, y1, y2)

            return [dx1, dy1, dx2, dy2]
        return func_vdp_2
    
    params = vdp_params()
    rhs = func_vdp_2_maker(params)
    sol = solve_ivp(rhs, [0, t_max], IC, method='RK45', rtol=1e-5, atol=1e-5)
    print(sol.status)

    xs = [sol.y[0], sol.y[2]]
    ys = [sol.y[1], sol.y[3]]
    ts = sol.t
    
    size = len(ts)

    plt.plot(ts, xs[0], label='1')
    plt.plot(ts, xs[1], label='2')
    plt.grid()
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

    # plt.plot(ts, ys[0], label='1')
    # plt.plot(ts, ys[1], label='2')
    # plt.grid()
    # plt.xlabel('t')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()

    # Dists
    dists = []
    for i in range(size):
        dists.append(np.sqrt((xs[0][i]-xs[1][i])**2+(ys[0][i]-ys[1][i])**2))
    plt.figure(figsize=[12, 3])
    plt.plot(ts, dists)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('dist')
    plt.show()

    # phases diff
    phases_diff = []
    for i in range(size):
        phase1 = np.arctan2(ys[0][i], xs[0][i])
        phase2 = np.arctan2(ys[1][i], xs[1][i])
        phases_diff.append(phase_converter(phase1 - phase2))
    plt.figure(figsize=[12, 3])
    plt.plot(ts, phases_diff)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'$\phi_1 - \phi_2$')
    plt.show()

    # amplitudes diff
    As_diff = []
    As = [[], []]
    for i in range(size):
        A1 = np.sqrt(xs[0][i]**2 + ys[0][i]**2)
        A2 = np.sqrt(xs[1][i]**2 + ys[1][i]**2)
        As[0].append(A1)
        As[1].append(A2)
        As_diff.append(A1 - A2)
    plt.figure(figsize=[12, 3])
    plt.plot(ts, As_diff)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'A_1 - A_2')
    plt.show()

    plt.figure(figsize=[12, 3])
    plt.plot(ts, As[0], label=r'A_1')
    plt.plot(ts, As[1], label=r'A_2')
    plt.legend()
    plt.grid()
    plt.xlabel('t')
    plt.ylabel(r'A_1, A_2')
    plt.show()
    
solver()
