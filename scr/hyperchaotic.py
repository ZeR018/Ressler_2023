import settings as s
import memory_worker as mem
from scipy.integrate import solve_ivp

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, p):
    return 0

def func_dx(i, r, _T=s.T, connect_f=default_f, k = 4, ):
    return - r[i*k + 1] - r[i*k + 2] + connect_f(i, r, _T, 'x')

def func_dy(i, r, a, _T=s.T, connect_f=default_f, k = 4):
    return r[i*k] + a * r[i*k + 1] + r[i*k + 3] + connect_f(i, r, _T, 'x')

def func_dz(i, r, b, _T=s.T, connect_f=default_f, k = 4):
    return b + r[i*k] * r[i*k + 2]

def func_dw(i, r, c, d, _T=s.T, connect_f=default_f, k = 4):
    return - c * r[i*k + 2] + d * r[i*k + 3] + connect_f(i, r, _T, 'x')

def func_rossler_4d_params_maker(a, b, c, d, T = 0.3, k_elements = 1, tau = 1):

    def func_rossler_4d_params(t, r):
        res_arr = []

        for i in range(k_elements):
            # x_i = r[i*k]
            # y_i = r[i*k + 1]
            # z_i = r[i*k + 2]
            # w_i = r[i*k + 3]

            dx = tau * func_dx(i, r, T)
            dy = tau * func_dy(i, r, a, T)
            dz = tau * func_dz(i, r, b, T)
            dw = tau * func_dw(i, r, c, d, T)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)
            res_arr.append(dw)

        return res_arr
    return func_rossler_4d_params

#################################################

def solo_experiment_4d_rossler():
    # params
    a = 0.22
    b = 3
    c = 0.5
    d = 0.05

    # IC
    IC = mem.generate_random_IC_ressler(5., 5., 1., 1)
    IC.append(0.5)

    # integrate
    func_rossler_4d_params = func_rossler_4d_params_maker(a, b, c, d)
    sol = solve_ivp(func_rossler_4d_params, [0, 100], IC, rtol=s.toch[0], atol=s.toch[1], method=s.method)

    xs = sol.y[0]
    ys = sol.y[1]
    zs = sol.y[2]
    ts = sol.t

    print(ts)

    # graphs

solo_experiment_4d_rossler()