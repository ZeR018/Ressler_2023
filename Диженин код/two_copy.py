import numpy as np
from matplotlib import *
from scipy import *
from pylab import figure, show, setp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.signal import hilbert
from matplotlib.animation import FuncAnimation

from sympy import *

def num_rossler(x1_n, y1_n, z1_n, x2_n, y2_n, z2_n, h, a, b, c, w1, w2, d_it):
    def d(x1_n, x2_n, y1_n, y2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (x1_n - x2_n)**2 < 3**2):
            # d_val = 0.2
            d_val = d_it

        return d_val

    # dx_1=(-w1*y1_n-z1_n)
    # dy_1=(w1*x1_n+a*y1_n + d(x1_n, x2_n, y1_n, y2_n) * (y2_n - y1_n))
    # dz_1=(b+z1_n*(x1_n-c))

    # dx_2=(-w2*y2_n-z2_n)
    # dy_2=(w2*x2_n+a*y2_n + d(x1_n, x2_n, y1_n, y2_n) * (y2_n - y1_n))
    # dz_2=(b+z2_n*(x2_n-c))

    dx_1=(-w1*y1_n-z1_n)
    dy_1=(w1*x1_n+a*y1_n + d_it * (y2_n - y1_n))
    dz_1=(b+z1_n*(x1_n-c))

    dx_2=(-w2*y2_n-z2_n)
    dy_2=(w2*x2_n+a*y2_n + d_it * (y1_n - y2_n))
    dz_2=(b+z2_n*(x2_n-c))

    x1_n1=x1_n+h*dx_1
    y1_n1=y1_n+h*dy_1
    z1_n1=z1_n+h*dz_1

    x2_n1=x2_n+h*dx_2
    y2_n1=y2_n+h*dy_2
    z2_n1=z2_n+h*dz_2

    phi_1 = np.arctan(dy_1/dx_1)
    phi_1 = np.arctan(y1_n/x1_n)
    # dphi_1 = (w1 * dx_1 * (z1_n - w1 * y1_n) + dy_1 * (a * z1_n - w1 * a + w1 * w1 * x1_n + w1 * a * y1_n) + dz_1 * (w1 * x1_n + a * y1_n)) / (w1 * w1 * y1_n * y1_n + 2 * w1 * y1_n * z1_n + z1_n * z1_n)

    # phi_2 = np.arctan(dy_2/dx_2)
    # dphi_2 = (w2 * dx_2 * (z2_n - w2 * y2_n) + dy_2 * (a * z2_n - w2 * a + w2 * w2 * x2_n + w2 * a * y2_n) + dz_2 * (w2 * x2_n + a * y2_n)) / (w2 * w2 * y2_n * y2_n + 2 * w2 * y2_n * z2_n + z2_n * z2_n)

    # dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    # dphi_g_1 = dphi_1
    # res_dphi1 = dphi_f_1 * dphi_g_1

    # dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    # dphi_g_2 = dphi_2
    # res_dphi2 = dphi_f_2 * dphi_g_2

    dphi_1 = ((w1 + a) * dx_1 + (-w1 - 1) * dy_1) / (dx_1**2)

    phi_2 = np.arctan(dy_2/dx_2)
    phi_2 = np.arctan(y2_n/x2_n)
    dphi_2 = ((w2 + a) * dx_2 + (-w2 - 1) * dy_2) / (dx_2**2)

    dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    dphi_g_1 = dphi_1
    res_dphi1 = dphi_f_1 * dphi_g_1

    dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    dphi_g_2 = dphi_2
    res_dphi2 = dphi_f_2 * dphi_g_2

    d_dx = (-1)*w1 - 1
    d_dy = w1 + a

    first = dy_1 * d_dx - dx_1 * d_dy
    sec = dx_1**2 + dy_1**2

    res_dphi1 = (first/sec)

    d_dx = (-1)*w2 - 1
    d_dy = w2 + a

    first = dy_2 * d_dx - dx_2 * d_dy
    sec = dx_2**2 + dy_2**2

    res_dphi2 = first/sec

    phase_diff_1 = phi_1

    # return x1_n1, y1_n1, z1_n1, x2_n1, y2_n1, z2_n1, res_dphi1, res_dphi2, phase_diff_1
    return x1_n1, y1_n1, z1_n1, x2_n1, y2_n1, z2_n1

def num_rossler_freq(x1_n, y1_n, z1_n, x2_n, y2_n, z2_n, h, a, b, c, w1, w2, d_it):
    def d(x1_n, x2_n, y1_n, y2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (x1_n - x2_n)**2 < 3**2):
            # d_val = 0.2
            d_val = d_it

        return d_val

    # dx_1=(-w1*y1_n-z1_n)
    # dy_1=(w1*x1_n+a*y1_n + d(x1_n, x2_n, y1_n, y2_n) * (y2_n - y1_n))
    # dz_1=(b+z1_n*(x1_n-c))

    # dx_2=(-w2*y2_n-z2_n)
    # dy_2=(w2*x2_n+a*y2_n + d(x1_n, x2_n, y1_n, y2_n) * (y2_n - y1_n))
    # dz_2=(b+z2_n*(x2_n-c))

    dx_1=(-w1*y1_n-z1_n)
    dy_1=(w1*x1_n+a*y1_n + d_it * (y2_n - y1_n))
    dz_1=(b+z1_n*(x1_n-c))

    dx_2=(-w2*y2_n-z2_n)
    dy_2=(w2*x2_n+a*y2_n + d_it * (y1_n - y2_n))
    dz_2=(b+z2_n*(x2_n-c))

    # y_1_diff = Symbol('y1')
    # y_2_diff = Symbol('y2')
    # x_1_diff = Symbol('x1')
    # x_2_diff = Symbol('x2')
    # z_1_diff = Symbol('z1')
    # z_2_diff = Symbol('z2')

    # y_for_diff = w1*x_1_diff+a*y_1_diff + d_it * (y_2_diff - y_1_diff)
    # x_for_diff= -w1*y_1_diff-z_1_diff

    # x_diff = x_for_diff.diff('y1')
    # print(x_diff)

    x1_n1=x1_n+h*dx_1
    y1_n1=y1_n+h*dy_1
    z1_n1=z1_n+h*dz_1

    x2_n1=x2_n+h*dx_2
    y2_n1=y2_n+h*dy_2
    z2_n1=z2_n+h*dz_2

    phi_1 = np.arctan(dy_1/dx_1)
    phi_1 = np.arctan(y1_n/x1_n)
    # dphi_1 = (w1 * dx_1 * (z1_n - w1 * y1_n) + dy_1 * (a * z1_n - w1 * a + w1 * w1 * x1_n + w1 * a * y1_n) + dz_1 * (w1 * x1_n + a * y1_n)) / (w1 * w1 * y1_n * y1_n + 2 * w1 * y1_n * z1_n + z1_n * z1_n)

    # phi_2 = np.arctan(dy_2/dx_2)
    # dphi_2 = (w2 * dx_2 * (z2_n - w2 * y2_n) + dy_2 * (a * z2_n - w2 * a + w2 * w2 * x2_n + w2 * a * y2_n) + dz_2 * (w2 * x2_n + a * y2_n)) / (w2 * w2 * y2_n * y2_n + 2 * w2 * y2_n * z2_n + z2_n * z2_n)

    # dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    # dphi_g_1 = dphi_1
    # res_dphi1 = dphi_f_1 * dphi_g_1

    # dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    # dphi_g_2 = dphi_2
    # res_dphi2 = dphi_f_2 * dphi_g_2

    dphi_1 = ((w1 + a) * dx_1 + (-w1 - 1) * dy_1) / (dx_1**2)

    phi_2 = np.arctan(dy_2/dx_2)
    phi_2 = np.arctan(y2_n/x2_n)
    dphi_2 = ((w2 + a) * dx_2 + (-w2 - 1) * dy_2) / (dx_2**2)

    dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    dphi_g_1 = dphi_1
    res_dphi1 = dphi_f_1 * dphi_g_1

    dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    dphi_g_2 = dphi_2
    res_dphi2 = dphi_f_2 * dphi_g_2

    d_dx = (-1)*w1 - 1
    d_dy = w1 + a

    first = dy_1 * d_dx - dx_1 * d_dy
    sec = dx_1**2 + dy_1**2

    res_dphi1 = (first/sec)

    d_dx = (-1)*w2 - 1
    d_dy = w2 + a

    first = dy_2 * d_dx - dx_2 * d_dy
    sec = dx_2**2 + dy_2**2

    res_dphi2 = first/sec

    phase_diff_1 = phi_1

    return res_dphi1, res_dphi2, phase_diff_1


a = 0.22
b = 0.1
c = 8.5

w1=1.02
w2=0.98

t_ini=0
t_fin=1
h=0.00001
numsteps=int((t_fin-t_ini)/h)
t=np.linspace(t_ini,t_fin,numsteps)

d_ini=0
d_fin=0.16
d_h=0.0000016
d_numsteps=int((d_fin-d_ini)/d_h)
d=np.linspace(d_ini,d_fin,d_numsteps)

x1=np.zeros(numsteps)
y1=np.zeros(numsteps)
z1=np.zeros(numsteps)

x2=np.zeros(numsteps)
y2=np.zeros(numsteps)
z2=np.zeros(numsteps)

freq_1=np.zeros(numsteps)
freq_2=np.zeros(numsteps)

phase_diff_1=np.zeros(numsteps)
phase_diff_2=np.zeros(numsteps)

x1[0]=0.1
y1[0]=0
z1[0]=0

x2[0]=0
y2[0]=0
z2[0]=0

# for k in range(numsteps-1):
#     [x1[k+1],y1[k+1],z1[k+1],x2[k+1],y2[k+1],z2[k+1]]=num_rossler(x1[k],y1[k],z1[k],x2[k],y2[k],z2[k],t[k+1]-t[k],a,b,c,w1,w2,d[k])

# for k in range(d_numsteps-1):
#     [freq_1[k+1],freq_2[k+1],phase_diff_1[k+1]]=num_rossler_freq(x1[k],y1[k],z1[k],x2[k],y2[k],z2[k],t[k+1]-t[k],a,b,c,w1,w2,d[k])

# h1 = hilbert(x1)
# h2 = hilbert(x2)
# phi1 = np.unwrap(np.angle(h1))
# phi2 = np.unwrap(np.angle(h2))

# fig = figure(figsize=(10, 6))
# ax1 = fig.add_axes([0.30, 0.7, 0.4, 0.2])
# ax3 = fig.add_axes([0.05, 0.1, 0.4, 0.5],projection='3d')
# ax4 = fig.add_axes([0.55, 0.1, 0.4, 0.5],projection='3d')

# ax1.plot(t, x1,color='red',lw=1,label='x1(t)')
# ax1.set_xlabel('t')
# ax1.set_ylabel('x(t)')
# ax1.axis((t_ini,t_fin,min(x1),max(x1)))

# ax1.plot(t, x2,color='blue',lw=1,label='x2(t)')
# ax1.axis((t_ini,t_fin,min(x2),max(x2)))

# ax4.plot(x2, y2,z2,color='blue',lw=1)
# ax4.set_xlabel('x2(t)')
# ax4.set_ylabel('y2(t)')
# ax4.set_zlabel('z2(t)')

# ax3.plot(x1, y1,z1,color='red',lw=1)
# ax3.set_xlabel('x1(t)')
# ax3.set_ylabel('y1(t)')
# ax3.set_zlabel('z1(t)')
# show()

# instantaneous_frequency_1 = (np.diff(phi1) / (2.0*np.pi) * 200)
# instantaneous_frequency_2 = (np.diff(phi2) / (2.0*np.pi) * 200)

ph_for_gr_1=np.zeros(100)
ph_for_gr_2=np.zeros(100)
d_val=np.zeros(100)

freq_1_c = np.zeros(100)
freq_2_c = np.zeros(100)

for d_i in range(d_numsteps-1):
    if (d_i < 100):
        d_val[d_i] = d[d_i]
        for k in range(d_numsteps-1):
            if (k < 100):
                [freq_1[k+1],freq_2[k+1],phase_diff_1[k+1]]=num_rossler_freq(x1[k],y1[k],z1[k],x2[k],y2[k],z2[k],t[k+1]-t[k],a,b,c,w1,w2,d[k])
            else:
                break
        ph_for_gr_1[d_i] = np.mean(freq_1_c)
        ph_for_gr_2[d_i] = np.mean(freq_2_c)
    else:
        break

fig_freq = figure(figsize=(10, 6))
ax_freq_1 = fig_freq.add_subplot()
ax_freq_1.plot(d_val, ph_for_gr_1,color='red',lw=1,label='')
ax_freq_1.axis((min(d_val),max(d_val),min(ph_for_gr_1),max(ph_for_gr_1)))
ax_freq_1.set_xlabel('d')

# ax_freq_1.plot(t, freq_1,color='red',lw=1,label='')
# ax_freq_1.axis((t_ini,t_fin,min(freq_1)*2,max(freq_1)*2))
# ax_freq_1.set_xlabel('t')

ax_freq_1.set_ylabel('frequency')
ax_freq_1.plot(d_val, ph_for_gr_2,color='blue',lw=1,label='')

# fig_phase_diff = figure(figsize=(10, 6))
# ax_phase_diff_1 = fig_phase_diff.add_subplot()
# ax_phase_diff_1.plot(d, phi1,color='red',lw=1,label='')
# ax_phase_diff_1.axis((d_ini,d_fin,min(phi1),max(phi1)))
# ax_phase_diff_1.set_xlabel('d')
# ax_phase_diff_1.set_ylabel('phase')


# anim = FuncAnimation(fig, update_all, frames=np.size(x), interval=0, blit=True)
# anim_phi = FuncAnimation(fig, update_phi, frames=np.size(x), interval=0, blit=True)

# plt.savefig('rosslerAttractor.png')
# plt.show()
show()
