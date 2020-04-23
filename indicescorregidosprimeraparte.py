"""
Created on Thu Apr 23 16:18:20 2020

@author: javie
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import scipy as sp
from scipy.sparse import csr_matrix, linalg
from scipy.stats import linregress
from scipy import linalg, optimize

################### Problema 1.4
## Parte 1
# Escriba dos funciones que calculen A_h y b_h. La entrada debe ser N,f,g
#falta hacer otra, podria ser otra que no use kron
def calcula_A(N):
    h = 1/N
    #Primero calculamos el L_4
    down = np.ones(N-2)
    center = np.ones(N-1)
    upper = np.ones(N-2)
    d1 = -down
    d2 = 4*center
    d3 = -1*upper
    d = np.array([d1, d2, d3])
    offset = [-1, 0, 1]
    L4 = sp.sparse.diags(d, offset)
    #Ahora la A_h
    dd1 = -down
    dd2 = np.zeros(N-1)
    dd3 = -upper
    dd = np.array([dd1, dd2, dd3])
    A1 = sp.sparse.diags(dd, offset)

    I = np.identity(N-1)

    L = sp.sparse.kron(A1, I)
    R = sp.sparse.kron(I, L4)

    A = (L + R) / h**2
    return A

def g(x,y):
    if y==1 and x<1 and x>0:
        return np.sin(2*np.pi*x)
    else:
        return 0

f = lambda x,y: 8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

def calcula_b(N, f, g):
    #h = 1/N
    f_h=np.zeros((N-1)**2)
    g_h=np.zeros((N-1)**2)
    x=np.linspace(0,1,num=N)
    y=np.linspace(0,1,num=N)

    for j in range(1,N):
        for k in range(1,N):    
            f_h[(k-1)*(N-1)+j-1]=f(x[j], y[k])

    g_h[0]=N**2*(g(x[0],0)+g(0,y[0]))
    g_h[N-1-1]=N**2*(g(x[N-1-1],0)+g(1,y[1-1]))
    g_h[(N-1)**2-(N-2)-1]=N**2*(g(x[1-1],1)+g(0,y[N-1-1]))
    g_h[(N-1)**2-1]=N**2*(g(x[N-1-1],1)+g(1,y[N-1-1]))
    for j in range(2,N-1):#esto es para j in {2,...,N-2}
        g_h[j-1]=N**2*g(x[j-1],0)
        g_h[(N-1)*(N-2)+j-1]=N**2*g(x[j-1],1)
        g_h[j*(N-1)+1-1]=N**2*(g(0,y[j-1]))
        g_h[j*(N-1)-1]=N**2*g(1,y[j-1])

    b_h=f_h+g_h
    return b_h

## Parte 2
# Para N en {4,16}, grafique la solución numérica y la solución única de
# la ecuación.

N = [4, 16]

for i in range(0,len(N)):
    u = sp.sparse.linalg.spsolve(calcula_A(N[i]), calcula_b(N[i], f, g))
    U = np.zeros((N[i]+1, N[i]+1))
    h = 1/N[i]
    for j in range(1,N[i]-1):
        U[j][N[i]]=np.sin(2*np.pi*j/h)
        for k in range(1,N[i]-1):
            U[k][j] = u[j + (k-1)*(N[i]-1)]
    #falta agregar los bordes a U
    
    x = np.linspace(0,1, N[i]+1)

    X, Y = np.meshgrid(x, x)
    fig = plt.figure(i)
    fig.clf()
    ax = fig.add_subplot(111, projection='3d', elev=30, azim=10)
    ax.plot_surface(X, Y, U) #, rstride=2, cstride=2, cmap=cm.plasma
    #ax.dist = 1
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    fig.show()
print("funciona")