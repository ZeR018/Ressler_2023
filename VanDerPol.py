from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

def solver(IC = [1.9, 0.3, 2.1, 0.], t_max = 4000):
    class vdp_params():
        def __init__(self, mu = 0.001, beta = 0, gamma = 0., delta = 0.1, l = 0.1):
            self.mu = mu
            self.delta = delta
            self.beta = beta
            self.gamma = gamma
            self.l = l

    def func_vdp_2_maker(p : vdp_params):

        def dissipative_coup(beta, y_this, y_other):
            return beta * (y_other - y_this)
        
        def divide_coup(l, y_this, y_other):
            return l / (y_other - y_this)

        def func_vdp_2(t, r):
            x1, y1, x2, y2 = r

            dx1 = y1
            dy1 = p.mu * (1. - x1**2) * y1 - x1 + p.mu * dissipative_coup(p.beta, y1, y2) + p.mu * divide_coup(p.l, y2, y1)
            dx2 = y2
            dy2 = p.mu * (1 + p.gamma - x2**2) * y2 - (1 + p.mu * p.delta) * x2 + p.mu * dissipative_coup(p.beta, y2, y1) + p.mu * divide_coup(p.l, y1, y2)

            return [dx1, dy1, dx2, dy2]
        return func_vdp_2
    
    params = vdp_params()
    rhs = func_vdp_2_maker(params)
    sol = solve_ivp(rhs, [0, t_max], IC, method='RK45', rtol=1e-12, atol=1e-12)

    xs = [sol.y[0], sol.y[2]]
    ys = [sol.y[1], sol.y[3]]
    ts = sol.t

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
    for i in range(len(ts)):
        dists.append(np.sqrt((xs[0][i]-xs[1][i])**2+(ys[0][i]-ys[1][i])**2))
    plt.plot(ts, dists)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('dist')
    plt.show()


    
solver()
