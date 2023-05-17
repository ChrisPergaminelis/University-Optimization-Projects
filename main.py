from forecast import Forecaster
from optimization import GradientBasedAlgorithms
import numpy as np
import random
import matplotlib.pyplot as plt


def fobj(weights):
    cost = []
    for t in range(N-m+1, N):
        y_hat_vector = np.array([
            fc.simple_moving_average(xi=5, t=t),
            fc.linear_exponential_smoothing(alpha=0.5, t=t),
            fc.simple_exponential_smoothing(alpha=0.3, beta=0.5, t=t)
        ])
        cost.append((np.dot(y_hat_vector, np.transpose(weights)) - y_vector[t]) ** 2)
    return (1/m) * sum(cost)

def gradfobj(weights):
    sas = []
    les = []
    ses = []
    cost = []
    for t in range(N-m+1, N):
        sas.append(fc.simple_moving_average(xi=5, t=t))
        les.append(fc.linear_exponential_smoothing(alpha=0.5, t=t))
        ses.append(fc.simple_exponential_smoothing(alpha=0.3, beta=0.5, t=t))
        y_hat_vector = np.array([
            fc.simple_moving_average(xi=5, t=t),
            fc.linear_exponential_smoothing(alpha=0.5, t=t),
            fc.simple_exponential_smoothing(alpha=0.3, beta=0.5, t=t)
        ])
        cost.append(np.dot(y_hat_vector, np.transpose(weights)) - y_vector[t])
    sas = sum(sas)
    les = sum(les)
    ses = sum(ses)
    cost = sum(cost)
    return np.array([(2/m) * cost * sas,
            (2/m) * cost * les,
            (2/m) * cost * ses])


def grad2fobj(weights):
    sas = []
    les = []
    ses = []
    for t in range(N-m+1, N):
        sas.append(fc.simple_moving_average(xi=5, t=t))
        les.append(fc.linear_exponential_smoothing(alpha=0.5, t=t))
        ses.append(fc.simple_exponential_smoothing(alpha=0.3, beta=0.5, t=t))
    sas = sum(sas)
    les = sum(les)
    ses = sum(ses)
    return np.array([np.array([(2/m) * sas *sas, 0, 0]),
            np.array([0, (2/m) * les *les, 0]),
            np.array([0, 0, (2/m) * ses *ses])])


global y_vector, N, m
N = 100
m = 5
f = open('GBPUSD.dat', 'r')
y_vector = np.array(list(map(eval, f.read().splitlines())))
fc = Forecaster(y_vector)

random.seed(10)
weights = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
print("starting weights")
print(weights)
gba = GradientBasedAlgorithms(fobj, gradfobj)
Hk = grad2fobj(weights)

method = int(input('''1) bfgs\n2) newton\n3) bfgs-trust_region\n'''))
res = []
title = ""
if method == 1:
    res = gba.bfgs(weights, Hk, 10 ** (-7), 1400)
    title = "bfgs"
elif method == 2:
    res = gba.newton(weights, Hk, 10 ** (-7), 1400)
    title = "newton"
else:
    res = gba.bfgs_trust_region(weights, Hk, 10 ** (-7), 1400)
    title = "bfgs-trust_region"
weights = res[1]
print("final weights")
print(weights)

y_hat_w = []
for t in range(1, 120) :
    y_hat_vector = np.array([
        fc.simple_moving_average(xi=5, t=t),
        fc.linear_exponential_smoothing(alpha=0.5, t=t),
        fc.simple_exponential_smoothing(alpha=0.3, beta=0.5, t=t)
    ])
    y_hat_w.append(np.dot(y_hat_vector, np.transpose(weights)))

plt.plot(np.linspace(1, 120, 119), y_hat_w, label='y_hat')
plt.plot(np.linspace(1, 120, 119), y_vector[1:], label='y')
plt.legend()
plt.title(title)
plt.show()

plt.plot(np.linspace(1, 120, 119), np.array(y_hat_w)-np.array(y_vector[1:]), label='y_hat-y')
plt.legend()
plt.title("error")
plt.show()
