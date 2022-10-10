import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt


def zad2_1():
    print('Zad 2.1')
    x = pow(3, 12) - 5
    print(f'x = {x}')

    a = np.array([2, .5])
    b = np.array([[1, 4], [-1, 3]])
    c = np.array([[-1], [-3]])
    print(f'x = {a @ b @ c}')

    x = lin.matrix_rank(np.array([[1, -2, 0], [-2, 4, 0], [2, -1, 7]]))
    print(f'x = {x}')

    a = np.array([[-1], [2]])
    b = lin.inv(np.array([[1, 2], [-1, 0]]))
    x = b @ a
    print(f'x = {x}')


def zad2_2():
    print('Zad 2.2')
    x = -46
    table = np.array([1, 1, -129, 171, 1620])
    wsp = np.array([pow(x, 4), pow(x, 3), pow(x, 2), pow(x, 1), pow(x, 0)])
    wielomian = table * wsp;
    final = wielomian @ np.array([[1], [1], [1], [1], [1]])
    print(f'Dla x1=-46: {final}')

    x = 14
    wsp = np.array([pow(x, 4), pow(x, 3), pow(x, 2), pow(x, 1), pow(x, 0)])
    wielomian = table * wsp;
    final = wielomian @ np.array([[1], [1], [1], [1], [1]])
    print(f'Dla x2=14: {final}')


def zad3_1():
    print('Zad 3.1')
    table = np.array([1, 1, -129, 171, 1620])
    max = False
    min = False

    for x in np.arange(-46, 14, 1):
        wsp = np.array([pow(x, 4), pow(x, 3), pow(x, 2), pow(x, 1), pow(x, 0)])
        wielomian = table * wsp
        temp = wielomian @ np.array([[1], [1], [1], [1], [1]])
        if max:
            if max < temp:
                max = temp
            elif min > temp:
                min = temp
        else:
            min = temp
            max = temp

    print(f'max: {max}')
    print(f'min: {min}')


def zad3_2(p):
    print('Zad 3.2')
    table = np.array([1, 1, -129, 171, 1620])
    max = False
    min = False

    for x in np.arange(-46, 14, p):
        wsp = np.array([pow(x, 4), pow(x, 3), pow(x, 2), pow(x, 1), pow(x, 0)])
        wielomian = table * wsp
        temp = wielomian @ np.array([[1], [1], [1], [1], [1]])
        if max:
            if max < temp:
                max = temp
            elif min > temp:
                min = temp
        else:
            min = temp
            max = temp

    print(f'max: {max}')
    print(f'min: {min}')


def zad4(wsp, down, high, p):
    print('Zad 4')
    max = False
    min = False
    leng = len(wsp)
    Yzad5 = np.empty(high-down+p, dtype=int)
    iterator = 0
    for x in np.arange(down, high+p, p):

        wielomian = 0
        i = leng

        while i > 0:
            wielomian += wsp[leng-i] * pow(x, i-1)
            i -= 1

        Yzad5[iterator] = wielomian
        iterator += 1

        if max or max == 0:
            if max < wielomian:
                max = wielomian
            elif min > wielomian:
                min = wielomian
        else:
            min = wielomian
            max = wielomian

    minMax = np.array([max, min])
    print(f'Wynik: {minMax}')

    plt.xlabel('arguments')
    plt.ylabel('value')
    plt.title('Polynomial')
    plt.grid('both')
    plt.plot(np.arange(down, high+p, p), Yzad5)
    plt.show()
    print(Yzad5)
    plt.savefig('wykres.pdf', format='pdf')


if __name__ == '__main__':
    zad2_1()
    zad2_2()
    zad3_1()
    zad3_2(1)
    zad4(np.array([1, 1, -129, 171, 1620]), -46, 14, 1)



