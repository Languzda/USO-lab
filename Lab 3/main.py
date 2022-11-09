import numpy as np
import sympy as sym
import scipy as scipy


def zadanie1_2():
    A1 = np.matrix([[-1/2, 0], [0, -1/2]])
    B1 = np.matrix([[1/2], [1/2]])
    #K1 = np.matrix([[B1], [A1*B1]])
    #print(K1)


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
    k1 = sym.Symbol('k1')
    k2 = sym.Symbol('k2')
    k3 = sym.Symbol('k3')
    x = sym.Symbol('x')
    Lam = np.matrix([[x, 0, 0], [0, x, 0], [0, 0, x]])
    K = np.matrix([k1, k2, k3])
    X = A-(B@K)
    Z = (Lam-X)
    M = sym.Matrix(Z)
    Wyz = (M.det())
    print(Wyz)


def zadanie3_2_1(w):
    k1 = w[0]
    k2 = w[1]
    k3 = w[2]
    R = [0, 0, 0]
    R[0] = k1+0.5*k2+0.333333333333333*k3 + 1.83333333333333-8
    R[1] = 0.833333333333333*k1 + 0.666666666666667*k2 + 0.5*k3-17
    R[2] = 5.55111512312578e-17*k1*k2*k3 + 0.166666666666667*k1 + 0.166666666666667*k3 + 0.166666666666667-10
    return R


if __name__ == '__main__':
    # Zadanie 3.2.1
    print(scipy.optimize.fsolve(zadanie3_2_1, np.array([0, 0, 0])))
    zadanie1_2()
