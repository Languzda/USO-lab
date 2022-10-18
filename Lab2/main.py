import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import scipy.signal as signal
from scipy.signal import tf2ss
from scipy.signal import ss2tf
import scipy.integrate as odeint
from scipy.signal import bessel, lsim2

def zad_2():
    kp = 3
    T = 2
    A = -(1/T)
    B = kp/T
    C = 1
    D = 0

    g1 = signal.lti([kp],[T,1])
    t, y = signal.step(g1)
    plt.plot(t, y, 'bs', label='czasowa')

    g2 = signal.StateSpace(A, B, C, D)
    t, y = signal.step(g2)
    plt.plot(t, y, 'g^', label='zmienne stanu')

    #2.6
    def model(y, t):
        # kp = 3
        # T = 2
        u = 1
        dydt = (-y + kp*u)/T
        return dydt

    y0 = 0
    t = np.arange(0, 16,.1)
    y = scipy.integrate.odeint(model, y0, t)

    plt.plot(t, y, 'r--', label='całka')
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend()
    plt.show()


def zad_3():
    # 3.1
    R = 12
    L = 1
    C = 0.0001

    num = [1, 0]
    den = [L, R, 1 / C]
    obj1 = signal.TransferFunction(num, den)

    #odpowiedź skokowa
    t, y = signal.step(obj1)

    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Skokowa')
    plt.grid()
    plt.show()

    # odpowiedź impulsowa
    t, y = signal.impulse(obj1)
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('impulsowa')
    plt.grid()
    plt.show()

    # 3.2
    A = np.matrix([[0, 1], [-1 / (L * C), -R / L]])
    B = np.matrix([[0], [1 / L]])
    C = np.matrix([[0, 1]])
    D = 0

    #obiekt z zmiennych stanów
    obj2 = signal.StateSpace(A, B, C, D)

    #skokowa
    t, y = signal.step(obj2)
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Skokowa zmienne stanu')
    plt.grid()
    plt.show()

    #impulsowa
    t, y = signal.impulse(obj2)
    plt.plot(t, y)
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Impulsowa zmienne stanu')
    plt.grid()
    plt.show()
    # Wykresy się pokrywają

    # 3.3
    A1, B1, C1, D1 = tf2ss(num, den)
    #zmienne stanu wyliczone i wyznaczone róźnią się od siebie
    print(f'wyznaczone {A,B,C,D}')
    print(A1, B1, C1, D1)

    num1, den1 = ss2tf(A, B, C, D)
    #transmitancje sa prawie takie same
    print(f'wyznaczone {num, den}')
    print(num1, den1)

    #3.4


def zad_4():
    m = 1
    L = 0.5
    d = 0.1
    J = (m * L ** 2) / 3

    def model(x,t):
        tn=0
        dydt =(tn-d*x)/J
        return dydt

    y0 = 0
    t = np.linspace(0, 5, num=50)
    y = scipy.integrate.odeint(model,y0,t)
    #plt.plot(t,y,'r--')

    G1=signal.TransferFunction([1],[J,d,0])
    print(G1)
    t, y=signal.step(G1)
    plt.plot(t, y,'b--')


    A=np.array([[0,1],[0,-d/J]])
    B=np.array([[0],[1/J]])
    C=np.array([1,0])
    D=0

    G3=signal.StateSpace(A,B,C,D)
    t, y = signal.step(G3)
    plt.plot(t, y,'r')

    t = np.linspace(0, 5, num=50)
    u = np.ones_like(t)
    u = np.arange(0,len(t))

    system = signal.lti(A, B, C, D)
    tout, y, x = signal.lsim2(system, u, t)
    plt.plot(t, y,'g^')
    plt.show()

    #4.4
    w, mag, phase =signal.bode(G1)
    plt.figure()
    plt.semilogx(w, mag)  # Bode magnitude plot
    plt.figure()
    plt.semilogx(w, phase)  # Bode phase plot
    plt.show()


if __name__ == '__main__':
    #zad_2()
    #zad_3()
    zad_4()

