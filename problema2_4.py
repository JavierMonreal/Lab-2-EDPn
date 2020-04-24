import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import matplotlib.cm
import scipy.sparse

############################  problema 2.4

############### 2.4 Parte 1


'''
d_dt u - d2_dt2 u = 0 ;   (t,x) en (0,T)x(0,1)
u(t,x+1) = u(t,x)     ;
u(0,x)   = u_(x)      ;    x en (0,1)  

N_T entero, un paso tempora dt > 0,
N   entero, paso espacial   dx = 1/N                        

x_j = j dx, j en {0,...,N}
t_n = n dt, n en {0,...,N_T}

obs T = N_t*dt
'''

u_00 = lambda x: np.cos(0*x)

def calcula_bloque_C(dt,N_T,N,u_0):
    dx = 1/N

    # |{0,...,N}| = N+1 valores
    diagonal_interior  = np.full(N+1, 2*( 1/(dx**2) + 1/dt ) )
    diagonal_exterior = np.full(N,   -1/(dx**2) )
    
    diagonales = [diagonal_exterior,diagonal_interior,diagonal_exterior]
    offset = [-1,0,1]

    # las dimensiones del bloque son  (0 hasta N) dos veces
    C = sp.sparse.diags(diagonales,offset)
    return C

def calcula_b(dt,N_T,N,u_0):
    dx = 1/N
    b = np.zeros((N+1)*(N_T+1))

    for j in range(0,N+1):
        b[j] = u_0(j*dx)

    return b

def calcula_A(dt,N_T,N,u_0):
    C = calcula_bloque_C(dt,N_T,N,u_0)

    filas = []
    fila_vacía = np.full(N_T+1,None)

    fila_identidad = fila_vacía.copy()
    fila_identidad[0] = np.identity(N+1)

    filas.append(fila_identidad)

    for i in range(0,N_T):
        fila_iter = fila_vacía.copy()
        fila_iter[i]   =  C
        fila_iter[i+1] = -C

        filas.append(fila_iter)

    A = sp.sparse.bmat(filas)

    return A


def des_empaca_u(u,N,N_T):
    u_nuevo = np.zeros([N_T+1,N+1])
    for n in range(0,N_T+1):
        for j in range(0,N+1):
            u_nuevo[n][j] = u[n*(N+1)+j]
    return u_nuevo


def make_grid(N,N_T,dt):
    tt = np.linspace(0,N_T*dt,num=N_T+1,endpoint=True)
    xx = np.linspace(0,1,num=N+1,endpoint=True)
    
    assert (len(xx)==N+1) and (len(tt)==N_T+1)
    
    X,T = np.meshgrid(xx,tt)

    return T,X

############### 2.4 Parte 2

'''
Considere ∆x = 0.05 y ∆t ∈ {0.00625,0.025,0.1}.
Calcule la soluci´on num´erica para cada caso y comp´arelas con la
soluci´on real con una gr´aﬁca en T = 0.5. 
'''

# dx = 0.05 <-> 1/N = 0.05 <-> N = int(1/0.05)
N = int(1/0.05)
# T = 0.5 <-> N_T*dt = 0.5 <-> N_T = int(0.5/dt)
arr_dt = [0.00625,0.025,0.1]
arr_N_T  = np.divide(0.5*np.ones(len(arr_dt)),arr_dt)

arr_A  = []
arr_b  = []
arr_u  = []
arr_tx = []
arr_U  = []

L = len(arr_dt)

for i in range(0,L):
    dt = arr_dt[i]
    N_T = int(arr_N_T[i])

    A = calcula_A(dt,N_T,N,u_00).tocsr()
    b = calcula_b(dt,N_T,N,u_00)
    u = spsolve(A,b)

    T,X = make_grid(N,N_T,dt)
    U   = des_empaca_u(u,N,N_T)

    arr_A.append(A)
    arr_b.append(b)
    arr_u.append(u)

    arr_tx.append([T,X])
    arr_U.append(U)

gráficos = plt.figure()
ejes = np.full(L,None)

for i in range(0,L):
    ejes[i] = gráficos.add_subplot()

    CS = ejes[i].contourf(arr_tx[i][1] , arr_tx[i][0] , arr_U[i], cmap=cm.coolwarm)

    ejes[i].set_xlabel("variable x")
    ejes[i].set_ylabel("variable t")

gráficos.colorbar(CS)