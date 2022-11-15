import numpy as np
import sympy as sym
import scipy as scipy
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import bessel, lsim
import control as control


def kalman(A, B):
    # budowanie macierzy kalmana
    kalmanMatrix = B
    rankA = matrix_rank(A)

    for x in range(rankA-1):
        col = A ** (x+1) @ B
        kalmanMatrix = np.append(kalmanMatrix, col, axis = 1)

    # Sprawdzenie czy rząd macierzy A jest równy rzadowi macierzy kalmana, gdy jest macierz jest sterowalna
    if rankA == matrix_rank(kalmanMatrix):
        print('Układ sterowalny')
    else:
        print('Układ niesterowalny')
    return kalmanMatrix

def zadanie1_3(A, B, C, D):
    print('zad 1.3')
    sys = signal.StateSpace(A, B, C, D)
    t1, y1 = signal.step(sys)

    # odpowiedź skokowa
    plt.figure(0)
    plt.plot(t1, y1)
    plt.xlabel('Czas')
    plt.ylabel('y(t)')
    plt.title('1.3 Odpowiedź skokowa')
    plt.grid()
    plt.show()

    # odpowiedź na wymuszenie sinusoidalne
    u_down = np.arange(1, -50, -1)
    t = np.arange(0, 51, 1)
    plt.figure(1)
    tout, y2, x2 = signal.lsim(sys, u_down, t)
    plt.plot(t, y2)
    plt.title('1.3 wymuszenie sinusoidalne')
    plt.grid()
    plt.show()


def zadanie2(A,C,D, K):
    print('zad 2')
    #macierz obliczona ręcznie współczynniki i macierz sterowalnosci
    poly = np.poly(A)
    #nowa macierz
    alpha = np.matrix([[0, 1, 0],[0, 0, 1],[-poly[3], -poly[2], -poly[1]]])
    beta = np.matrix([[0],[0],[1]])
    K2 = kalman(alpha, beta)
    Sal = K2
    print(K2)

    #macierz podobieństwa
    P = Sal @ K**(-1)
    print(C,P)
    gamma = C @ np.linalg.inv(P)
    delta = D
    newMatrix = signal.StateSpace(alpha,beta,gamma,delta)
    G2 = signal.StateSpace(alpha, beta, gamma, delta)
    ti, yi = signal.step(G2)
    plt.plot(ti, yi, 'g')
    plt.show()
    return(poly, S, alpha, beta, K2, Sal, P, gamma, delta, newMatrix)


def zadanie3_1(A):
    print('zad 3.1')
    if np.linalg.matrix_rank(A) == 3:
        x = sym.Symbol('x')
        Lam = np.matrix([[x, 0, 0], [0, x, 0], [0, 0, x]])
        X1 = Lam-A
        charakt = X1[0, 0]*X1[1, 1]*X1[2, 2]
        print(sym.expand(charakt))
    else:
        print("macierz nie sterowalna")


def zadanie3_2(A, B):
    print('zad 3.2')
    # Deklaracja zmiennych symbolicznych
    k1 = sym.Symbol('k1')
    k2 = sym.Symbol('k2')
    k3 = sym.Symbol('k3')
    x = sym.Symbol('x')

    #stworzenie macierzy lmbda oraz K
    Lam = np.matrix([[x, 0, 0], [0, x, 0], [0, 0, x]])
    K = np.matrix([k1, k2, k3])

    X = A-(B@K)
    Z = (Lam-X)
    M = sym.Matrix(Z)
    Wyz = (M.det())
    k =np.matrix( scipy.optimize.fsolve(zadanie3_2_1, np.array([0, 0, 0])))
    # k = [-9.9, -13.8, 68.9]
    print(k)
    print(f'Wyznacznik {Wyz}')
    u = -k@X
    print(u)


# def zadanie3_3(A,B,C,D,K):
#     An = A - B*K
#     Bn = np.matrix([[0], [0], [1]])
#     Cn = C - D*K
#     model = signal.StateSpace(An, Bn, Cn, 0)


# Rozwiązanie układu nieliniowego na podstawie wyliczonego wyznacznika z poprzedniego zadania
def zadanie3_2_1(w):
    k1 = w[0]
    k2 = w[1]
    k3 = w[2]

    # Rozwiązania
    R = [0, 0, 0]
    R[0] = 1.83333333333333+k3-8
    R[1] = k2 +1-17
    R[2] = k1 + 0.166666666666667-10
    return R


if __name__ == '__main__':
    # Parametry
     A = np.matrix([[-1, 0, -0], [0, -1/2, 0], [0, 0, -1/2]])
     B= np.matrix([[1], [1/3], [1/3]])
     C = np.matrix([[-1, 1/2, -1/3]])
     D = 11/6
     Aprim = np.matrix([[0, 1, 0], [0 , 0 , 1], [-1/6, -1, -11/6]])
     Bprim = np.matrix([[0], [0], [1]])
    # sprawdzanie czy układ jest sterowalny
     K=kalman(A, B)
     #zadanie1_3(A, B, C, D)
     #zadanie2(A,C,D,K)
     zadanie3_1(Aprim)
    # Zadanie 3.2.1
     dane = scipy.optimize.fsolve(zadanie3_2_1, np.array([0, 0, 0]))
     print('zad 3.2',dane)
     t = np.linspace(0, 10, num=50)
     u = np.zeros_like(t)
     k = [9.83333333]
     AA = np.matrix([[0, 1, 0],[0, 0, 1],[-1/6-dane[0], -1-dane[1], -11/6-dane[2]]])
     system = signal.lti(AA, np.matrix([[0],[0],[0]]), np.matrix([1, 0, 0]), 0)
     x0 = np.matrix([1, 0 ,0])
     tout,y ,x = signal.lsim2(system,u,t,x0)
     plt.plot(t,x)
     plt.show()
     zadanie3_2(Aprim,Bprim)
