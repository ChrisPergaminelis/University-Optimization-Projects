import numpy as np


class LineSearch:

    def __init__(self, fobj, c1, c2):
        self.fobj = fobj
        self.c1 = c1
        self.c2 = c2

    def phyobj(self, xk, a, pk):
        return self.fobj(xk + np.multiply(a, pk))

    def dphyobj(self, xk, a, pk):
        h = 10 ** (-16)
        return (self.phyobj(xk, a + h, pk) - self.phyobj(xk, a, pk)) / h

    def zoom(self, a_low, a_high, xk, pk):
        phy_0 = self.phyobj(xk, 0, pk)
        dphy_0 = self.dphyobj(xk, 0, pk)
        aj = 0
        k = 1
        while k <= 50:
            aj = (a_low + a_high) / 2
            phy_aj = self.phyobj(xk, aj, pk)

            if phy_aj > phy_0 + (self.c1 * aj * dphy_0) or phy_aj >= self.phyobj(xk, a_low, pk):
                a_high = aj
            else:
                dphy_aj = self.dphyobj(xk, aj, pk)

                if np.abs(dphy_aj) <= -self.c2 * dphy_0:
                    return aj

                if dphy_aj * (a_high - a_low) >= 0:
                    a_high = a_low

                a_low = aj
            k = k+1
        return aj

    def line_search(self, xk, a1, pk, a_max):
        a0 = 0
        i = 1
        while True:
            phy_a1 = self.phyobj(xk, a1, pk)
            phy_0 = self.phyobj(xk, 0, pk)
            dphy_0 = self.dphyobj(xk, 0, pk)

            if phy_a1 > phy_0 + self.c1 * a1 * dphy_0 or (phy_a1 >= self.phyobj(xk, a0, pk) and i > 1):
                return self.zoom(a0, a1, xk, pk)

            d_phy_a1 = self.dphyobj(xk, a1, pk)
            if np.abs(d_phy_a1) <= -self.c2 * dphy_0:
                return a1

            if d_phy_a1 >= 0:
                return self.zoom(a1, a0, xk, pk)

            a0 = a1
            a1 = (a1 + a_max) / 2
            i = i + 1


class GradientBasedAlgorithms:

    def __init__(self, fobj, gradofbj):
        self.fobj = fobj
        self.gradfobj = gradofbj
        return

    def steepest_descent(self, xk, epsilon, nstep):
        k = 1
        ln = LineSearch(self.fobj, 10**(-4), 0.9)
        while np.linalg.norm(self.gradfobj(xk)) >= epsilon and k <= nstep:
            pk = -np.array(self.gradfobj(xk))

            a_step = ln.line_search(xk, 0.5, pk, 1)

            xk = xk + np.multiply(a_step, pk)

            k = k + 1
        return [k, xk, self.fobj(xk)]

    def bfgs(self, xk, Hk, epsilon, nstep):
        k = 1
        ln = LineSearch(self.fobj, 10 ** (-4), 0.9)
        Hk = np.linalg.inv(Hk)
        while np.linalg.norm(self.gradfobj(xk)) >= epsilon and k <= nstep:
            pk = -np.matmul(Hk, self.gradfobj(xk))
            a_step = ln.line_search(xk, 0.5, pk, 1)
            xk_old = xk
            xk = xk + np.multiply(a_step, pk)

            sk = xk - xk_old
            if sk.all() == 0:
                return [k, xk, self.fobj(xk)]

            yk = np.array(self.gradfobj(xk)) - np.array(self.gradfobj(xk_old))
            rk = 1/np.matmul(np.transpose(yk), sk)
            temp = np.multiply( (1 - (rk * np.matmul(sk, np.transpose(yk)))), Hk )
            temp = np.multiply(temp, (1 - (rk * np.matmul(yk, np.transpose(sk)))) )
            Hk = temp + np.multiply(rk, np.matmul(sk, np.transpose(sk)))

            k = k+1
        return [k, xk, self.fobj(xk)]

    def newton(self, xk, Hk, epsilon, nstep):
        k = 1
        ln = LineSearch(self.fobj, 10 ** (-4), 0.9)
        Hk = np.linalg.inv(Hk)
        while np.linalg.norm(self.gradfobj(xk)) >= epsilon and k <= nstep:
            pk = -np.matmul(Hk, self.gradfobj(xk))
            a_step = ln.line_search(xk, 0.5, pk, 1)
            xk = xk + np.multiply(a_step, pk)
            k = k+1
        return [k, xk, self.fobj(xk)]

    def dogleg(self, xk, Dk, Hk):
        gk = self.gradfobj(xk)
        pB = -np.matmul(Hk, gk)

        if np.linalg.norm(pB) <= Dk:
            return pB

        pU = - np.dot((np.dot(np.transpose(gk), gk) / np.dot(np.transpose(gk), np.dot(Hk, gk))), gk)
        if np.linalg.norm(pU) >= Dk:
            return -np.dot(Dk / np.linalg.norm(gk), gk)

        tau = np.divide((Dk - pU), (pB - pU))
        return pU + np.matmul((tau - 1), (pB - pU))

    def trust_region(self, xk, Hk, epsilon, nstep):
        k = 1
        eta = 0.05
        Dk = 0.5
        Hk2 = Hk
        Hk = np.linalg.inv(Hk)
        while np.linalg.norm(self.gradfobj(xk)) >= epsilon and k <= nstep:
            pk = self.dogleg(xk, Dk, Hk)
            gk = np.array(self.gradfobj(xk))

            r_top = self.fobj(xk) - self.fobj(xk + pk)
            r_bot = -(np.dot(np.transpose(pk), gk) + 0.5 * np.dot(np.transpose(pk), np.dot(Hk2, pk)))
            rk = r_top/r_bot

            if rk < 1/4:
                Dk = (1/4)*Dk
            else:
                if rk > 3/4 and np.linalg.norm(pk) == Dk:
                    Dk = 2*Dk
                else:
                    Dk = Dk

            if rk > eta:
                xk = xk + pk
            else:
                xk = xk

            k = k+1

        return [k, xk, self.fobj(xk)]

    def bfgs_trust_region(self, xk, Hk, epsilon, nstep):
        k = 1
        Hk2 = Hk
        Hk = np.linalg.inv(Hk)
        ln = LineSearch(self.fobj, 10 ** (-4), 0.9)
        while np.linalg.norm(self.gradfobj(xk)) >= epsilon and k <= nstep:
            xk_old = xk
            xk = self.trust_region(xk, Hk2, epsilon, nstep)[1]

            sk = xk - xk_old
            if sk.all() == 0:
                return [k, xk, self.fobj(xk)]

            yk = np.array(self.gradfobj(xk)) - np.array(self.gradfobj(xk_old))
            rk = 1/np.matmul(np.transpose(yk), sk)
            temp = np.multiply( (1 - (rk * np.matmul(sk, np.transpose(yk)))), Hk )
            temp = np.multiply(temp, (1 - (rk * np.matmul(yk, np.transpose(sk)))) )
            Hk = temp + np.multiply(rk, np.matmul(sk, np.transpose(sk)))
            k = k+1
        return [k, xk, self.fobj(xk)]