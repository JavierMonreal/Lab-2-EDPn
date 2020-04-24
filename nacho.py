import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import scipy as sp
from scipy.sparse import csr_matrix, linalg
from scipy.stats import linregress
from scipy import linalg, optimize

# Escriba dos funciones que calculen A_h y b_h. La entrada debe ser N,f,g


def calcula_A(N):
    h = 1/ N
    down = np.ones(N-2)
    center = np.ones(N-1)
    upper = np.ones(N-2)
    d1 = -1*down
    d2 = 4*center
    d3 = -1*upper
    d = np.array([d1, d2, d3])
    offset = [-1, 0, 1]
    L4 = sp.sparse.diags(d, offset)

    dd1 = -down
    dd2 = np.zeros(N-1)
    dd3 = -upper
    dd = np.array([dd1, dd2, dd3])
    A1 = sp.sparse.diags(dd, offset)

    I = np.identity(N-1)

    L = sp.sparse.kron(A1, I)
    R = sp.sparse.kron(I, L4)

    A = (L + R) / (h**2)
    return A


def g(x,y):
    if y==1 and x<1 and x>0:
        return np.sin(2*np.pi*x)
    else:
        return 0


f = lambda x,y: 8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


def calcula_b(N, f, g):
    f_h = np.zeros((N+1)**2)
    g_h = np.zeros((N+1)**2)
    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)

    for j in range(N+1):
        for k in range(N+1):
            f_h[(k)*(N+1)+j] = f(x[j], y[k])

    g_h[1] = N**2*(g(x[1],0)+g(0,y[1]))
    g_h[N-1] = N**2*(g(x[N-1],0)+g(1,y[1]))
    g_h[(N-1)**2-(N-2)] = N**2*(g(x[1],1)+g(0,y[N-1]))
    g_h[(N-1)**2] = N**2*(g(x[N-1],1)+g(1,y[N-1]))
    for j in range(2,N-1):#esto es para j in {2,...,N-2}
        g_h[j] = N**2*g(x[j],0)
        g_h[(N-1)*(N-2)+j] = N**2*g(x[j],1)
        g_h[j*(N-1)+1] = N**2*(g(0,y[j]))
        g_h[j*(N-1)] = N**2*g(1,y[j])

    b_h = f_h + g_h      # matriz de NxN que no tiene el largo de A_h

    # hacer matriz de (N-1)x(N-1)
    bb_h = np.zeros((N-1)**2)

    for i in range(1, N):  # parte en 1 y termina en N-1 (interior de [0,1])
        for j in range(1,N): # parte en 1 y termina en N-1 (interior de [0,1])
            bb_h[(i-1) + (j-1)*(N-1)] = b_h[i + j*(N+1)]

    return bb_h

## Parte 2
# Para N en {4,16}, grafique la solución numérica y la solución única de
# la ecuación.

N = [4, 16]

for i in range(2):
    u = sp.sparse.linalg.spsolve(calcula_A(N[i]), calcula_b(N[i], f, g))
    U = np.zeros((N[i]+1, N[i]+1))
    x = np.linspace(0,1, N[i]+1)

    for j in range(N[i]+1):
        for k in range(N[i]+1):
            if j == N[i]:
                U[k][j] = g(x[k], 1)
            if 0 < k < N[i] and 0 < j < N[i]:
                U[k][j] = u[(k-1) + (j-1)*(N[i]-1)]

    X, Y = np.meshgrid(x, x)
    fig = plt.figure(i)
    fig.clf()
    ax = fig.add_subplot(111, projection='3d', elev=15, azim=10)
    ax.plot_surface(X, Y, U) #, rstride=2, cstride=2, cmap=cm.plasma
    #ax.dist = 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    fig.show()
    

# Parte 3
#
# para N en {4,8,16,32,64} , calcule el error en norma L2.

arreglo_N = [4, 8, 16, 32, 64]
errores_norm_2 = []

u_analitico = lambda x,y: np.sin(2*np.pi*x)*(np.sin(2*np.pi*y) + (np.sinh(2*np.pi*y) / np.sinh(2*np.pi)))

for i in range(len(arreglo_N)):
    N = arreglo_N[i]
    h = 1 / N

    u = sp.sparse.linalg.spsolve(calcula_A(N), calcula_b(N, f, g))
    U = np.zeros((N+1, N+1))
    x = np.linspace(0, 1, N+1)
    
    for j in range(N+1):
        for k in range(N+1):
            if j == N:
                U[k][-1] = g(x[k], 1)
            if 0 < k < N and 0 < j < N:
                U[k][j] = u[(k-1) + (j-1)*(N-1)]
    # for j in range(N):
    #     for k in range(N):
    #         U[k+1][j+1] = u[k + j*(N)]

    x = np.linspace(0,1,N+1)
    y = np.linspace(0,1,N+1)
    U_analitico = np.zeros((N+1, N+1))

    for j in range(N+1):
        for k in range(N+1):    
            U_analitico[j][k] = u_analitico(x[j], y[k])

    if N == 16:
        x = np.linspace(0,1, N+1)
        
        X, Y = np.meshgrid(x, x)
        fig = plt.figure(3)
        fig.clf()
        ax = fig.add_subplot(111, projection='3d', elev=15, azim=10)
        ax.plot_surface(X, Y, U_analitico) #, rstride=2, cstride=2, cmap=cm.plasma
        #ax.dist = 1
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        fig.show()
    
    err = h * np.linalg.norm(U_analitico - U, 2)
    errores_norm_2.append(err)

# Grafique los respectivos valores en función de h, en escala logarítmica
# usando log log ¿qué puede observar?

     
h = 1 / np.array(arreglo_N)

plt.figure(2)
plt.clf()
plt.loglog(h, errores_norm_2, label='Errores $\\| \\cdot \\|_{2}$')
plt.xlabel('paso')
plt.ylabel('error')
plt.legend()
plt.show()


lr = linregress(np.log(h), np.log(errores_norm_2))

print('Estimacion de p = {}'.format(lr[0]))
print('')


""" arreglo_N = [ 2**(i) for i in range(2,7)]
arreglo_condicion_A_h = []

for N in arreglo_N:
    # cambiar por matriz A_h
    A_h = np.identity( (N-1)**2 )
    condicion_A_h = np.linalg.cond(A_h, p = 2)
    arreglo_condicion_A_h.append(condicion_A_h)

arreglo_h   = np.divide(1,arreglo_N)
arreglo_N_2 = np.multiply(arreglo_N,arreglo_N) 

condicion_fig,condicion_ax = plt.subplots(2)
condicion_h  = condicion_ax[0]
condicion_n2 = condicion_ax[1]

condicion_h.loglog(arreglo_h,arreglo_condicion_A_h)
condicion_n2.loglog(arreglo_N_2,arreglo_condicion_A_h)

plt.show() """


# Calcule el orden de error experimental, es decir, estime mediante
# regresión lineal el valor de p tal que e_h sea de orden O(h^p)

