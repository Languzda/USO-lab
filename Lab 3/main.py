import numpy as np
import sympy as sym
import scipy as scipy
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import bessel, lsim


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


def zadanie1_3(A, B, C, D):
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
    tout, y2, x2 = lsim(sys, u_down, t)
    plt.plot(t, y2)
    plt.grid()
    plt.show()


def zadanie3_1(A):
    if np.linalg.matrix_rank(A) == 3:
        x = sym.Symbol('x')
        Lam = np.matrix([[x, 0, 0], [0, x, 0], [0, 0, x]])
        X1 = Lam-A
        charakt = X1[0, 0]*X1[1, 1]*X1[2, 2]
        print(sym.expand(charakt))
    else:
        print("macierz nie sterowalna")


def zadanie3_2(A, B):
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
    print(Wyz)


def zadanie2(A, K):
    #macierz obliczona ręcznie współczynniki i macierz sterowalnosci
    poly = np.poly(A)
    S = K[0]

    #nowa macierz
    alpha = np.matrix([[0, 1, 0],[0, 0, 1],[-poly[3], -poly[2], -poly[1]]])
    beta = np.matrix([[0],[0],[1]])
    K2 = kalman(alpha, beta)
    Sal = K2[0]

    #macierz podobieństwa
    P = Sal @ S**(-1)
    gamma = C*P**(-1)
    delta = D
    newMatrix = signal.StateSpace(alpha,beta,gamma,delta)
    return(poly, S, alpha, beta, K2, Sal, P, gamma, delta, newMatrix)


# Rozwiązanie układu nieliniowego na podstawie wyliczonego wyznacznika z poprzedniego zadania
def zadanie3_2_1(w):
    k1 = w[0]
    k2 = w[1]
    k3 = w[2]

    # Rozwiązania
    R = [0, 0, 0]
    R[0] = k1+0.5*k2+0.333333333333333*k3 + 1.83333333333333-8
    R[1] = 0.833333333333333*k1 + 0.666666666666667*k2 + 0.5*k3-17
    R[2] = 5.55111512312578e-17*k1*k2*k3 + 0.166666666666667*k1 + 0.166666666666667*k3 + 0.166666666666667-10
    return R


if __name__ == '__main__':
    # Parametry
    A = np.matrix([[-1, 2, -1], [1, -1, -3], [1, 4, 0]])
    B = np.matrix([[2], [3], [2]])
    C = np.matrix([[1, 1, 0]])
    D = 0
    # sprawdzanie czy układ jest sterowalny
    kalman(A, B)
    zadanie1_3(A, B, C, D)
    # Zadanie 3.2.1
    print(scipy.optimize.fsolve(zadanie3_2_1, np.array([0, 0, 0])))

