import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from scipy.integrate import quad
from scipy.special import jv
import datetime
import math
from numba import jit

start = datetime.datetime.now()     # Record time start

# Initialize variables
gamma = 0.0001    # Bromwich contour parameter
iterations = 1000    # Number of times to iterate the series
max_count = 1000   # Number of data points in x and y
epsilon = 10 ** -40     # Minimum number to divide by in Aitken's iteration
n = 0

u, step = np.linspace(0.000001, 10, max_count, retstep=True)    # Initializing x values

y = np.empty([4, max_count])     # Initializing y values -- form is y[iteration in Aitken's][x value]

y_denom = np.empty([max_count])    # Initializing array to store Aitken's denominator values
x_iter = np.empty([max_count])    # Initializing array to store Aitken's iteration as starting point for next iteration

# Initialize plot
style.use('fivethirtyeight')

fig = plt.figure()
ax = fig.add_subplot(111)

plt.autoscale(enable=True)

def printer(k):    # Prints the iteration and y values
    print(str(k) + "\n")

def plot():    # Plots estimation 6 (blue) and actual equation 6 (red)
    ax.clear()
    line6, = ax.plot(u, y[3][:])
    ax.plot(u, jv(0, u), 'tab:red')
    return line6,

@jit(nopython=True, cache=True)
def f_p(omega, u, k):      # Equation 6, Laplace form
    b = (omega + k * np.pi) / u
    a = gamma

    f6 = np.sqrt(np.sqrt((a ** 2 - b ** 2 + 1) ** 2 + (2 * a * b) ** 2)) * np.cos(
        (math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) / (
            np.sqrt((a ** 2 - b ** 2 + 1) ** 2 + (2 * a * b) ** 2) * (
                np.cos((math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) ** 2 + np.sin(
                    (math.atan2(2 * a * b, a ** 2 - b ** 2 + 1) + 2 * n * np.pi) / 2) ** 2)) * np.cos(omega)
    return f6

def closing(k):    # Checks for convergence. If so, adds each term to f_0 and ends process
    print("Equation convergence with " + str(k) + " iterations." + "\n")

    for i in range(max_count):
        y[3][i] += f_0(i)
    return

def f_0(i):     # Initial integral in the series. Added at end after closing() convergence test
    x_0 = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * quad(
        f_p, 0, np.pi / 2, args=(u[i], 0))[0]
    return x_0

def f_u():    # Main algorithm. Here, the summation integrals iterate
    k = 1   # Number of iterations

    for i in range(max_count):  # First overall series iteration n - 1 (initial 1st)
        y[0][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
            f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

    for j in range(iterations):   # Iteration loop. Goes until closing() or k = ~2000 (~8000 total iterations)
        if j > 0:
            for i in range(max_count):  # Sets the first iteration n - 1 (out of 3 for Aitken's) equal to the previous Aitken's or the first iteration if j = 0 (1st)
                y[0][i] = x_iter[i]

        k += 1

        for i in range(max_count):  # Second iteration n for Aitken's (2nd)
            y[1][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k += 1

        for i in range(max_count):  # Third iteration n + 1 for Aitken's (3rd)
            y[2][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k -= 1

        for i in range(max_count):  # Aitken's delta-squared method iteration n' using the previous 3 (4th)
            y_denom[i] = (y[2][i] - y[1][i]) - (y[1][i] - y[0][i])

            if abs(y_denom[i]) < epsilon:
                print("Denominator too small to calculate. Exiting...\n")
                return

            y[3][i] = y[2][i] - ((y[2][i] - y[1][i]) ** 2) / y_denom[i]

        printer(k)

        for i in range(max_count):  # Setting current Aitken's iteration to first iteration
            x_iter[i] = y[3][i]
    closing(k)
    return

if __name__ == '__main__':
    # Run the algorithm and then plot
    f_u()

    plot()

    end = datetime.datetime.now()   # Record time end

    print("Runtime: " + str(end - start))

    # Plots final result
    plt.show()

    exit(0)