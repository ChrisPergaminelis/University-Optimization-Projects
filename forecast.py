import numpy as np


class Forecaster:

    def __init__(self, y_array):
        self.y_array = np.array(y_array)
        self.y_hat_s = y_array[0]
        self.y_hat_l = y_array[0]
        self.Lt = 0
        self.Rt = 0
        return

    def simple_moving_average(self, xi, t):
        return 1/len(self.y_array[max([0, t-xi]):t]) * sum(self.y_array[max([0, t-xi]):t])

    def linear_exponential_smoothing(self, alpha, t):
        self.y_hat_l = alpha * self.y_array[t] + (1 - alpha) * self.y_hat_l
        return self.y_hat_l

    def simple_exponential_smoothing(self, alpha, beta, t):
        lt = self.Lt
        self.Lt = alpha * self.y_array[t] + (1 - alpha) * (self.Lt + self.Rt)
        self.Rt = beta * (self.Lt - lt) + (1 - beta) * self.Rt
        self.y_hat_s = self.Lt + self.Rt
        return self.y_hat_s
