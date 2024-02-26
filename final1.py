import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.integrate import quad
from scipy.special import jv
import math
import datetime
from numba import jit

start = datetime.datetime.now()     # Record time start

# Initialize plot
style.use('fivethirtyeight')

fig, axes = plt.subplots(3, 2)
((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axes

plt.autoscale(enable=True)

# Initialize variables
gamma = np.array([np.nan, 0.00004, 0.001, 0.0001, 0.0001, 0.7480169, 0.0001])    # Bromwich contour parameter for each equation
iterations = 100000    # Number of times to iterate the series
num = 1     # Number of equations to be processed in total
n = 0   # Repetition constant for equation 3, set to either zero or 1 (diverges at 1)
max_count = 1000   # Number of data points in x and y
epsilon = 10 ** -40     # Minimum number to divide by in Aitken's iteration

u, step = np.linspace(0.000001, 10, max_count, retstep=True)    # Initializing x values

y = np.empty([6, 4, max_count])     # Initializing y values -- form is y[equation number][iteration in Aitken's][x value]

y_denom = np.empty([6, max_count])    # Initializing array to store Aitken's denominator values for each equation
x_iter = np.empty([6, max_count])    # Initializing array to store Aitken's iteration as starting point for next iteration

def printer(num, k):    # Prints the equation number, iteration and y value at 10
    if num == 1:
        print(str(num) + "    " + str(k) + "    " + str(y[0][3][max_count - 1]) + "\n")
    elif num == 2:
        print(str(num) + "    " + str(k) + "    " + str(y[1][3][max_count - 1]) + "\n")
    elif num == 3:
        print(str(num) + "    " + str(k) + "    " + str(y[2][3][max_count - 1]) + "\n")
    elif num == 4:
        print(str(num) + "    " + str(k) + "    " + str(y[3][3][max_count - 1]) + "\n")
    elif num == 5:
        print(str(num) + "    " + str(k) + "    " + str(y[4][3][max_count - 1]) + "\n")
    elif num == 6:
        print(str(num) + "    " + str(k) + "    " + str(y[5][3][max_count - 1]) + "\n")

def plot1():    # Plots estimation 1 (blue) and actual equation 1 (red)
    ax1.clear()
    line1, = ax1.plot(u, y[0][3][:])
    ax1.plot(u, np.sin(u), 'tab:red')
    ax1.set_xlabel('L_inv(sin(x))', fontsize=12)
    return line1,

def plot2():    # Plots estimation 2 (blue) and actual equation 2 (red)
    ax2.clear()
    line2, = ax2.plot(u, y[1][3][:])
    ax2.plot(u, np.heaviside(u, 1), 'tab:red')
    ax2.set_xlabel('L_inv(H(u))', fontsize=12)
    return line2,

def plot3():    # Plots estimation 3 (blue) and actual equation 3 (red)
    ax3.clear()
    line3, = ax3.plot(u, y[2][3][:])
    ax3.plot(u, np.cos(2 * np.sqrt(u)) / np.sqrt(np.pi * u), 'tab:red')
    ax3.set_xlabel('L_inv(cos(2*sqrt(u))/(sqrt(pi)*u))', fontsize=12)
    return line3,

def plot4():    # Plots estimation 4 (blue) and actual equation 4 (red)
    ax4.clear()
    line4, = ax4.plot(u, y[3][3][:])
    ax4.plot(u, np.log(u), 'tab:red')
    ax4.set_xlabel('L_inv(ln(u))', fontsize=12)
    return line4,

def plot5():    # Plots estimation 5 (blue) and actual equation 5 (red)
    ax5.clear()
    line5, = ax5.plot(u, y[4][3][:])
    ax5.plot(u, np.heaviside(u - 1, 1) - np.heaviside(u - 2, 1), 'tab:red')
    ax5.set_xlabel('L_inv(H(u-1)-H(u-2))', fontsize=12)
    return line5,

def plot6():    # Plots estimation 6 (blue) and actual equation 6 (red)
    ax6.clear()
    line6, = ax6.plot(u, y[5][3][:])
    ax6.plot(u, jv(0, u), 'tab:red')
    ax6.set_xlabel('L_inv(J_0(u))', fontsize=12)
    return line6,

@jit(nopython=True, cache=True)
def f_p1(omega, u, k):      # Equation 1, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[1]

    f1 = (1 + a ** 2 - b ** 2) / (
        1 + a ** 4 + b ** 4 + 2 * a ** 2 - 2 * b ** 2 + 2 * a ** 2 * b ** 2) * np.cos(omega)
    return f1

@jit(nopython=True, cache=True)
def f_p2(omega, u, k):      # Equation 2, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[2]

    f2 = a / (a ** 2 + b ** 2) * np.cos(omega)
    return f2

@jit(nopython=True, cache=True)
def f_p3(omega, u, k):      # Equation 3, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[3]

    f3 = np.e ** (-a / (a ** 2 + b ** 2)) * (a ** 2 + b ** 2) ** (1 / 4) * np.cos(
        (math.atan2(b, a) + 2 * n * np.pi) / 2) * np.cos(b / (a ** 2 + b ** 2)) / (
            np.sqrt(a ** 2 + b ** 2) * (np.cos((math.atan2(b, a) + 2 * n * np.pi) / 2) ** 2 + np.sin(
                (math.atan2(b, a) + 2 * n * np.pi) / 2))) * np.cos(omega)
    return f3

@jit(nopython=True, cache=True)
def f_p4(omega, u, k):      # Equation 4, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[4]

    f4 = (-a * np.log(np.sqrt(a ** 2 + b ** 2)) - a * np.e + a * math.atan2(
        b, a) - b * math.atan2(b, a)) / (a ** 2 + b ** 2) * np.cos(omega)
    return f4

@jit(nopython=True, cache=True)
def f_p5(omega, u, k):      # Equation 5, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[5]

    f5 = a * (np.e ** (-a) * np.cos(b) - np.e ** (-2 * a) * np.cos(2 * b)) / (a ** 2 + b ** 2) + b * (
        np.e ** (-a) * np.sin(b) - np.e ** (-2 * a) * np.sin(2 * b)) / (a ** 2 + b ** 2) * np.cos(omega)
    return f5

@jit(nopython=True, cache=True)
def f_p6(omega, u, k):      # Equation 6, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma[6]

    f6 = np.sqrt(np.sqrt((a ** 2 - b ** 2 + 1) ** 2 + (2 * a * b) ** 2)) * np.cos(
        (math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) / (
            np.sqrt((a ** 2 - b ** 2 + 1) ** 2 + (2 * a * b) ** 2) * (
                np.cos((math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) ** 2 + np.sin(
                    (math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) ** 2)) * np.cos(omega)
    return f6

def closing(num, k):    # Checks for convergence. If so, adds each term to f_0 and ends process

    print("Equation " + str(num) + " converging with " + str(k) + " iterations." + "\n")

    if num == 1:
        for i in range(max_count):
            y[0][3][i] += f_0(num, i)
        return
    elif num == 2:
        for i in range(max_count):
            y[1][3][i] += f_0(num, i)
        return
    elif num == 3:
        for i in range(max_count):
            y[2][3][i] += f_0(num, i)
        return
    elif num == 4:
        for i in range(max_count):
            y[3][3][i] += f_0(num, i)
        return
    elif num == 5:
        for i in range(max_count):
            y[4][3][i] += f_0(num, i)
        return
    elif num == 6:
        for i in range(max_count):
            y[5][3][i] += f_0(num, i)
        return

def f_0(num, i):     # Initial integral in the series. Added at end after closing() convergence test
    if num == 1:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p1, 0, np.pi / 2, args=(u[i], 0))[0]
    elif num == 2:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p2, 0, np.pi / 2, args=(u[i], 0))[0]
    elif num == 3:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p3, 0, np.pi / 2, args=(u[i], 0))[0]
    elif num == 4:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p4, 0, np.pi / 2, args=(u[i], 0))[0]
    elif num == 5:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p5, 0, np.pi / 2, args=(u[i], 0))[0]
    elif num == 6:
        x_0 = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * quad(
            f_p6, 0, np.pi / 2, args=(u[i], 0))[0]

    return x_0

def f_u(u, num):    # Main algorithm. Here, the summation integrals iterate
    k = 1   # Number of iterations

    for i in range(max_count):  # First overall series iteration n - 1 (initial 1st)
        if num == 1:
            y[0][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p1, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
        elif num == 2:
            y[1][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p2, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
        elif num == 3:
            y[2][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p3, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
        elif num == 4:
            y[3][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p4, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
        elif num == 5:
            y[4][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p5, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
        elif num == 6:
            y[5][0][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p6, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

    for j in range(iterations):   # Iteration loop. Goes until closing() or k = ~10000 for each equation
        if j > 0:
            for i in range(max_count):  # Sets the first iteration n - 1 (out of 3 for Aitken's) equal to the previous Aitken's or the first iteration if j = 0 (1st)
                if num == 1:
                    y[0][0][i] = x_iter[0][i]
                elif num == 2:
                    y[1][0][i] = x_iter[1][i]
                elif num == 3:
                    y[2][0][i] = x_iter[2][i]
                elif num == 4:
                    y[3][0][i] = x_iter[3][i]
                elif num == 5:
                    y[4][0][i] = x_iter[4][i]
                elif num == 6:
                    y[5][0][i] = x_iter[5][i]

        k += 1

        for i in range(max_count):  # Second iteration n for Aitken's (2nd)
            if num == 1:
                y[0][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p1, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 2:
                y[1][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p2, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 3:
                y[2][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p3, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 4:
                y[3][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p4, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 5:
                y[4][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p5, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 6:
                y[5][1][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p6, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k += 1

        for i in range(max_count):  # Third iteration n + 1 for Aitken's (3rd)
            if num == 1:
                y[0][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p1, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 2:
                y[1][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p2, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 3:
                y[2][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p3, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 4:
                y[3][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p4, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 5:
                y[4][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p5, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]
            elif num == 6:
                y[5][2][i] = 2 * np.e ** (gamma[num] * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                    f_p6, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k -= 1

        for i in range(max_count):  # Aitken's delta-squared method iteration n' using the previous 3 (4th)
            if num == 1:
                y_denom[0][i] = (y[0][2][i] - y[0][1][i]) - (y[0][1][i] - y[0][0][i])

                if abs(y_denom[0][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[0][3][i] = y[0][2][i] - ((y[0][2][i] - y[0][1][i]) ** 2) / y_denom[0][i]
            elif num == 2:
                y_denom[1][i] = (y[1][2][i] - y[1][1][i]) - (y[1][1][i] - y[1][0][i])

                if abs(y_denom[1][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[1][3][i] = y[1][2][i] - ((y[1][2][i] - y[1][1][i]) ** 2) / y_denom[1][i]
            elif num == 3:
                y_denom[2][i] = (y[2][2][i] - y[2][1][i]) - (y[2][1][i] - y[2][0][i])

                if abs(y_denom[2][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[2][3][i] = y[2][2][i] - ((y[2][2][i] - y[2][1][i]) ** 2) / y_denom[2][i]
            elif num == 4:
                y_denom[3][i] = (y[3][2][i] - y[3][1][i]) - (y[3][1][i] - y[3][0][i])

                if abs(y_denom[3][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[3][3][i] = y[3][2][i] - ((y[3][2][i] - y[3][1][i]) ** 2) / y_denom[3][i]
            elif num == 5:
                y_denom[4][i] = (y[4][2][i] - y[4][1][i]) - (y[4][1][i] - y[4][0][i])

                if abs(y_denom[4][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[4][3][i] = y[4][2][i] - ((y[4][2][i] - y[4][1][i]) ** 2) / y_denom[4][i]
            elif num == 6:
                y_denom[5][i] = (y[5][2][i] - y[5][1][i]) - (y[5][1][i] - y[5][0][i])

                if abs(y_denom[5][i]) < epsilon:
                    print("Denominator too small to calculate. Exiting...\n")
                    return

                y[5][3][i] = y[5][2][i] - ((y[5][2][i] - y[5][1][i]) ** 2) / y_denom[5][i]

        printer(num, k)

        for i in range(max_count):  # Setting current Aitken's iteration to first iteration
            if num == 1:
                x_iter[0][i] = y[0][3][i]
            elif num == 2:
                x_iter[1][i] = y[1][3][i]
            elif num == 3:
                x_iter[2][i] = y[2][3][i]
            elif num == 4:
                x_iter[3][i] = y[3][3][i]
            elif num == 5:
                x_iter[4][i] = y[4][3][i]
            elif num == 6:
                x_iter[5][i] = y[5][3][i]
    closing(num, k)
    return

if __name__ == '__main__':
    # Run the algorithm for each equation and plot sequentially
    while num <= 6:
        f_u(u, num)

        if num == 1:
            ln1, = plot1()
        elif num == 2:
            ln2, = plot2()
        elif num == 3:
            ln3, = plot3()
        elif num == 4:
            ln4, = plot4()
        elif num == 5:
            ln5, = plot5()
        elif num == 6:
            ln6, = plot6()

        num += 1

    end = datetime.datetime.now()   # Record time end

    print("Runtime: " + str(end - start))

    # Plots final result
    fig.tight_layout()
    plt.show()

    exit(0)





