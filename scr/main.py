from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time

w = 0.98        #
a = 0.22        # Параметры
b = 0.1         # системы
c = 8.5         #

k_elements = 5  # Число агентов
k = 3           # Число уравнений для одного агента (всегда 3)

radius = 3      # Радиус связи
T = 0.5         # Сила связи


# Функции синхронизации

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r):
    return 0


# Функция связи по y. Последовательное движение цепочки агентов
def func_connect_y(index, r):
    summ = 0
    for i in range(k_elements):
        summ += d(T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1]) * (r[i*k + 1] - r[index*k + 1])
    return summ


# Функция связи по x. Параллельное движение цепочки агентов
def func_connect_x(index, r):
    summ1, summ2 = 0, 0
    for j in range(k_elements):   
        if j != index:
            summ1 += d(T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) * (r[j*k] - r[index*k])
            summ2 += d(T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) / (r[index*k] - r[j*k])
            
    return summ1 + summ2


# Функция включения связи между двумя агентами
def d(_T, _radius, x_i, x_j, y_i, y_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 < _radius**2:
        return _T
    else:
        return 0


# Функции правой части
# По x
def func_dx(index, r, connect_f=default_f):
    return - w * r[index*k + 1] - r[index*k + 2] + connect_f(index, r)


# По y.
def func_dy(index, r, connect_f=default_f):
    return w * r[index*k] + a * r[index*k + 1] + connect_f(index, r)


# По z
def func_dz(index, r, connect_f=default_f):
    return b + r[index*k + 2] * (r[index*k] - c) + connect_f(index, r)


def func_rossler_2_dim(t, r):
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = func_dx(i, r) + func_connect_x(i, r)
        dy = func_dy(i, r)
        dz = func_dz(i, r)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr


initial_conditions = [
    .0, .0, .0,
    .5, .5, .5,
    -.5, -.5, -.5,
    .3, .3, .3,
    -.3, -.3, -.3,
]

sol = solve_ivp(func_rossler_2_dim, [0, 200], initial_conditions, rtol=1e-11, atol=1e-11)

xs, ys, zs = [], [], []
for i in range(k_elements):
    xs.append(sol.y[i*k])
    ys.append(sol.y[i*k+1])
    zs.append(sol.y[i*k+2])
ts = sol.t
print(ts)

plt.figure(1, figsize=(10, 5))
for i in range(k_elements):
    plt.plot(ts, xs[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid()
    plt.title('X(t)')
    plt.legend()
plt.show()

for i in range(k_elements):
    plt.figure(2, figsize=(10, 5))
    plt.plot(ts, ys[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.xlabel('t')
    plt.ylabel('y')
    plt.grid()
    plt.title('Y(t)')
    plt.legend()
plt.show()

for i in range(k_elements):
    plt.figure(3, figsize=(10, 5))
    plt.plot(ts, zs[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.xlabel('t')
    plt.ylabel('z')
    plt.grid()
    plt.title('Z(t)')
    plt.legend()
plt.show()

for i in range(k_elements):
    plt.plot(xs[i], ys[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('Y(x)')
    plt.scatter(xs[i][-1], ys[i][-1], marker='o')
    plt.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
for i in range(k_elements):
    ax.plot3D(xs[i], ys[i], zs[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('f(x,y,z)')
    ax.scatter(xs[i][-1], ys[i][-1], zs[i][-1], marker='o')
    plt.legend()
plt.show()

xs_last_100, ys_last_100, zs_last_100 = [], [], []
for i in range(k_elements):
    xs_last_100.append(xs[i][-100:])
    ys_last_100.append(ys[i][-100:])
    zs_last_100.append(zs[i][-100:])


for i in range(k_elements):
    plt.plot(xs_last_100[i], ys_last_100[i], alpha=0.5, label=('eq' + str(i + 1)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title('Y(x)')
    plt.scatter(xs[i][-1], ys[i][-1], marker='o')
    plt.legend()
plt.show()

# ax2 = plt.figure().add_subplot(projection='3d')
# for i in range(k_elements):
#     ax2.plot3D(xs_last_100[i], ys_last_100[i], zs_last_100[i], alpha=0.5, label=('eq' + str(i + 1)))
#     plt.grid()
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('Y')
#     ax2.set_zlabel('Z')
#     plt.title('f(x,y,z)')
#     ax2.scatter(xs_last_100[i][-1], ys_last_100[i][-1], zs_last_100[i][-1], marker='o')
#     plt.legend()
# plt.show()
