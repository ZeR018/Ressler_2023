import numpy as np
from matplotlib import *
from scipy import *
from pylab import figure, show, setp

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import math

def rossler_eq(x_n,y_n,z_n,h,a,b,c,w):
    dx=((-w) * y_n - z_n)
    dy=(w * x_n + a * y_n)
    dz=(b + z_n * (x_n - c))

    x_n1 = x_n+h*dx
    y_n1 = y_n+h*dy
    z_n1 = z_n+h*dz

    phi = np.arctan(y_n1/x_n1)
    dphi_1 = (w * dx * (z_n - w * y_n) + dy * (a * z_n - w * a + (w**2) * x_n + w * a * y_n) + dz * (w * x_n + a * y_n)) / ((w**2) * (y_n**2) + 2 * w * y_n * z_n + (z_n**2))

    dphi_f_1 = 1 / (1 + (dy/dx)**2)
    dphi_g_1 = dphi_1
    #  -->
    res_dphi1 = dphi_f_1 * dphi_g_1

    return x_n1,y_n1,z_n1,phi,res_dphi1

def main():
    a = 0.22
    b = 0.1
    c = 8.5
    w = 0.98

    t_ini=0
    t_fin=200
    h=0.001
    numsteps=int((t_fin - t_ini) / h)

    t = np.linspace(t_ini,t_fin,numsteps)

    x = np.zeros(numsteps)
    y = np.zeros(numsteps)
    z = np.zeros(numsteps)
    phi = np.zeros(numsteps)
    dphi = np.zeros(numsteps)

    x[0] = 0
    y[0] = 0
    z[0] = 0
    phi[0] = 0
    dphi[0] = 0

    for i in range(numsteps - 1):
        x[i + 1], y[i + 1], z[i + 1], phi[i + 1], dphi[i + 1] = rossler_eq(x[i], y[i], z[i], t[i + 1] - t[i], a, b, c, w)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_axes([0.1, 0.7, 0.4, 0.2])
    ax2 = fig.add_axes([0.1, 0.4, 0.4, 0.2])
    ax3 = fig.add_axes([0.1, 0.1, 0.4, 0.2])
    ax4 = fig.add_axes([0.55, 0.25, 0.35, 0.5],projection='3d')

    # ax1.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax1.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax1.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))

    # ax2.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax2.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax2.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))

    # ax3.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax3.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax3.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))

    # ax4.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax4.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    # ax4.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))

    ax1.plot(t, x,color='red',lw=1,label='x(t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x(t)')
    ax1.legend()
    ax1.axis((t_ini,t_fin,min(x),max(x)))

    ax2.plot(t, y,color='green',lw=1,label='y(t)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t)')
    ax2.legend()
    ax2.axis((t_ini,t_fin,min(y),max(y)))

    ax3.plot(t, z,color='blue',lw=1,label='z(t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('z(t)')
    ax3.legend()
    ax3.axis((t_ini,t_fin,min(z),max(z)))

    ax4.plot(x, y,z,color='darkcyan',lw=1,label='Evolution(t)')
    ax4.set_xlabel('x(t)')
    ax4.set_ylabel('y(t)')
    ax4.set_zlabel('z(t)')
    show()

    fig_phase = plt.figure(figsize=(5, 3))
    ax_phase = fig_phase.add_subplot()
    ax_phase.plot(t, phi,color='red',lw=1,label='phi(t)')
    ax_phase.axis((t_ini,t_fin,min(phi) * 2,max(phi) * 2))
    show()

    fig_freq = plt.figure(figsize=(5, 3))
    ax_freq = fig_freq.add_subplot()
    ax_freq.plot(t, dphi,color='red',lw=1,label='phi(t)')
    ax_freq.axis((t_ini,t_fin,min(dphi) * 2,max(dphi) * 2))
    show()

if __name__ == "__main__":
    main()
