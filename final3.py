import matplotlib.pyplot as plt    # For plotting the graphs
from matplotlib import style    # For stylizing the graph area
import numpy as np    # For useful / efficient array operations and functions (e, pi, sin(), etc.)
from scipy.integrate import quad    # Gaussian quadrature integration function, used for all numerical integrations
from scipy.signal import unit_impulse    # Used to emulate discrete equivalent of Dirac delta function
import datetime    # Used to record runtime of algorithm, including initialization of variables
from numba import jit    # Used to compile numerically intensive functions (those with @jit decoration) for efficiency

start = datetime.datetime.now()    # Record time start

# Initialize variables
gamma = 1.000000001    # Bromwich contour parameter
max_count = 1000    # Number of data points in x (u) and y (F(u))
iterations = max_count - 1    # Number of times to iterate the series, excluding first term in series ----V
epsilon = 10 ** -40    # Minimum number to divide by in Aitken's iteration #                  # (total = iterations + 1)

A = 56    # Atomic number of Fe (Iron)
alpha = ((A - 1) / (A + 1)) ** 2    # The scattering parameter of the medium (Iron in this case)
q = np.log(1 / alpha)    # Maximum change in lethargy (u) per collision

u, step = np.linspace(0.000001, 1, max_count, retstep=True)    # Initializing x-axis (u) values
y = np.empty([4, max_count])    # Initializing y (F(u)) values -- form is y[iteration in Aitken's][u value]

y_denom = np.empty([max_count])    # Initializing array to store Aitken's denominator values
x_iter = np.empty([max_count])    # Initializing array to store Aitken's iteration as starting point for next iteration

# The first three functions in the Laplace transform series, representing one, two and three collisions, respectively
f0 = (1 / (1 - alpha)) * unit_impulse(max_count)
f1 = np.e ** (-u) * (1 / (1 - alpha)) * (np.heaviside(u, 1) - np.heaviside(u - q, 1))
f2 = np.e ** (-u) * (1 / (1 - alpha) ** 2) * (u - np.heaviside(u - q, 1) * 2 * (u - q) + np.heaviside(
    u - 2 * q, 1) * (u - 2 * q))

# Initialize plot
style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

plt.autoscale(enable=True)

def printer(k):    # Prints the iteration and y (F(u)) values
    print(str(k) + "    " + str(y[3][max_count - 1]) + "\n")

def plot1():    # Plots each separate collision interval, including uncollided (F(u))
    ax1.clear()
    ax1.set_xlabel("Lethargy (u)")
    ax1.set_ylabel("F(u)")
    line1, = ax1.plot(u * 13.874821, f0, 'tab:red')
    line2, = ax1.plot(u * 13.874821, f1, 'tab:orange')
    line3, = ax1.plot(u * 13.874821, f2, 'y-')
    line4, = ax1.plot(u * 13.874821, f3, 'tab:purple')
    return line1, line2, line3, line4,

def plot2():    # Plots continuous graph of collision intervals 1 - 3+ (represents F_c(u))
    ax2.clear()
    ax2.set_xlabel("Lethargy (u)")
    ax2.set_ylabel("F(u)")
    line5, = ax2.plot(u * 13.874821, fc)
    return line5

@jit(nopython=True, cache=True)
def f_p(omega, u, k):    # Neutron lethargy image equation (f) for the third+ collision intervals, Laplace form
    b = (omega + k * np.pi) / u    # Imaginary component of I.V. Im(p)
    a = gamma    # Real component of I.V. (Bromwich contour) Re(p)

    f = (((a ** 2 - b ** 2) * (a * (alpha - 1) - np.e ** (-q * a) * np.cos(q * b) + 1) - (2 * a * b) * (np.e ** (
        -q * a) * np.sin(q * b) + b * (a - 1))) * (1 - 3 * np.e ** (-q * a) * np.cos(q * b) + 3 * np.e ** (
            -2 * q * a) * np.cos(2 * q * b) - np.e ** (-3 * q * a) * np.cos(3 * q * b)) + ((a ** 2 - b ** 2) * (
                np.e ** (-q * a) * np.sin(q * b) + b * (a - 1)) + (2 * a * b) * (a * (alpha - 1) - np.e ** (
                    -q * a) * np.cos(q * b) + 1)) * (3 * np.e ** (-q * a) * np.sin(q * b) - 3 * np.e ** (
                        -2 * q * a) * np.sin(2 * q * b) + np.e ** (-3 * q * a) * np.sin(3 * q * b))) / (((
                            a ** 2 - b ** 2) * (a * (alpha - 1) - np.e ** (-q * a) * np.cos(q * b) + 1) - (
                                2 * a * b) * (np.e ** (-q * a) * np.sin(q * b) + b * (a - 1))) ** 2 + ((
                                    a ** 2 - b ** 2) * (np.e ** (-q * a) * np.sin(q * b) + b * (a - 1)) + (
                                        2 * a * b) * (a * (alpha - 1) - np.e ** (-q * a) * np.cos(
                                            q * b) + 1)) ** 2) * np.cos(omega)

    return f

def closing(k):    # Signifies the end of the series iterations, -----------v
    print("Equation convergence with " + str(k) + " iterations." + "\n")    # adds the first integral and multiplies -
                                                                            # by the non-inverted constant
    for i in range(max_count):
        y[3][i] += f_0(i)
        y[3][i] *= (1 / (1 - alpha) ** 2)
    return

def f_0(i):    # Initial integral in the series. Added at end at closing()
    x_0 = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * quad(
        f_p, 0, np.pi / 2, args=(u[i], 0))[0]
    return x_0

def f_u():    # Main algorithm. Here, the summation integrals iterate
    k = 1    # Number of iterations

    for i in range(iterations):    # First overall series iteration n - 1 (initial 1st)
        y[0][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
            f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

    for j in range(iterations):    # Iteration loop. Goes until closing() or k = iterations
        if j > 0:
            for i in range(max_count):    # Sets the first iteration n - 1 (out of 3 for Aitken's) equal -
                y[0][i] = x_iter[i]       # to the previous Aitken's or the first iteration if j = 0 (1st)

        k += 1

        for i in range(max_count):    # Second iteration n loop for Aitken's (2nd)
            y[1][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k += 1

        for i in range(max_count):    # Third iteration (n + 1) loop for Aitken's (3rd)
            y[2][i] = 2 * np.e ** (gamma * u[i]) / (np.pi * u[i]) * (-1) ** k * quad(
                f_p, -np.pi / 2, np.pi / 2, args=(u[i], k))[0]

        k -= 1

        for i in range(max_count):    # Aitken's delta-squared method iteration (n') loop using the previous 3 (4th)
            y_denom[i] = (y[2][i] - y[1][i]) - (y[1][i] - y[0][i])

            if abs(y_denom[i]) < epsilon:
                print("Denominator too small to calculate. Exiting...\n")
                return

            y[3][i] = y[2][i] - ((y[2][i] - y[1][i]) ** 2) / y_denom[i]

        printer(k)

        for i in range(max_count):    # Sets current Aitken's iteration equal to first iteration
            x_iter[i] = y[3][i]
    closing(k)   # Multiplies by constants and adds non-iterated initial integral
    return

if __name__ == '__main__':
    # Run the main algorithm, populates array for 3+ collision numerical solution
    f_u()

    # Map final 3+ collision result to exponential decay, then define collided neutron function (F_c(u))
    f3 = np.e ** (-u) * (y[3][:])
    fc = f1 + f2 + f3

    # Populate the F(u) and F_c(u) graph data
    plot1()
    plot2()

    end = datetime.datetime.now()    # Record time end

    print("Runtime: " + str(end - start))

    # Plots final result
    plt.show()

    exit(0)
