import numpy as np
import time
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.io import loadmat


class RayTracing:

    def __init__(self, l, n_static, G_rx=1, G_tx=1, us=16, static_rx=False):
        self.l = l
        vmin = 0.5  # if RX moves with a speed below 0.5 m/s it is considered static
        if static_rx:
            self.v_rx = 0
        self.vrx = None  # speed vector
        self.n_static = n_static
        self.positions = np.zeros(
            (self.n_static + 3, 2)
        )  # positions coordinates [rx,tx,t,s1,...,sn_static]
        self.paths = {
            "delay": np.zeros(n_static + 2),
            "phase": np.zeros(n_static + 2),
            "AoA": np.zeros(n_static + 2),
            "gain": np.zeros(n_static + 2, dtype=complex),
        }
        self.beta = np.zeros(n_static + 1)
        self.G_rx = G_rx
        self.G_tx = G_tx
        self.us = us

        # select single carrier modulation for 60 GHz carrier frequency
        if l == 0.005:
            self.B = 1.76e9
            self.vmax = 5
            self.x_max = 20
            self.y_max = 20
            self.tx_signal = self.load_trn_field()
        # select OFDM with BPSK modulation for 28 GHz carrier frequency
        if l == 0.0107:
            # 5G-NR parameters
            self.delta_f = 120e3  # subcarrier spacing [Hz]
            self.n_sc = 3332  # number of subcarriers
            self.B = self.n_sc * self.delta_f  # bandwidth [Hz] (almost 400 MHz)
            self.vmax = 10
            self.x_max = 50
            self.y_max = 50
            self.tx_signal = self.generate_bpsk(self.n_sc)
        if l == 0.06:
            # 802.11ax parameters
            self.delta_f = 78.125e3  # subcarrier spacing [Hz]
            self.n_sc = 2048  # number of subcarriers
            self.B = self.n_sc * self.delta_f  # bandwidth [Hz] (160 MHz)
            self.vmax = 20
            self.x_max = 100
            self.y_max = 100
            self.tx_signal = self.generate_bpsk(self.n_sc)

        self.v_rx = np.random.uniform(vmin, self.vmax)  # speed modulus
        self.fd_max = self.vmax / l  # max achievable Doppler frequency
        self.fd_min = 100
        self.fd = np.random.uniform(self.fd_min, self.fd_max)
        self.T = 1 / (6 * self.fd_max)  # interpacket time (cir samples period)
        self.cir = None

    def dist(self, p1, p2):
        """
        Returns distance between two points in 2D.
        p1,p2: [x,y] Cartesian coordinates.
        """
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (0.5)

    def compute_AoA(self, ind):
        """
        Computes angles of arrival for each path.
        ind: path index whose AoA has to be computed.
        LoS AoA=0.
        """
        m = (self.positions[0, 1] - self.positions[1, 1]) / (
            self.positions[0, 0] - self.positions[1, 0]
        )  # LoS: y = mx+q
        q = self.positions[0, 1] - m * self.positions[0, 0]
        m1 = -1 / m

        x, y = self.positions[ind + 1, :]
        q1 = y - m1 * x
        x_p = (q - q1) / (m1 - m)
        y_p = m * x_p + q  # (x_p,y_p) = target/static obj. projection on LoS
        c = self.dist([x_p, y_p], self.positions[0, :])
        ip = self.dist([x, y], self.positions[0, :])
        assert int(ip**2) == int(c**2 + ((x - x_p) ** 2 + (y - y_p) ** 2))
        self.paths["AoA"][ind] = np.arccos(c / ip)

    def plot_pos(self):
        """
        Plots the environmental disposition.
        """
        print("AoA : LoS, t, s1, s2, ... \n" + str(np.rad2deg(self.paths["AoA"])))
        plt.plot(self.positions[0, 0], self.positions[0, 1], "ro", label="rx")
        plt.plot(self.positions[1, 0], self.positions[1, 1], "go", label="tx")
        plt.plot(self.positions[2, 0], self.positions[2, 1], "r+", label="target")
        plt.plot(self.positions[0:2, 0], self.positions[0:2, 1], label="LoS")
        for i in range(2, self.n_static + 3):
            plt.plot(
                [self.positions[1, 0], self.positions[i, 0], self.positions[0, 0]],
                [self.positions[1, 1], self.positions[i, 1], self.positions[0, 1]],
            )
            if i != 2:
                plt.plot(
                    self.positions[i, 0],
                    self.positions[i, 1],
                    "o",
                    label="static object %s" % (i - 2),
                )
        plt.plot(
            [self.positions[0, 0], self.positions[0, 0] + self.vrx[0]],
            [self.positions[0, 1], self.positions[0, 1] + self.vrx[1]],
            label="v_rx",
        )
        plt.legend()
        print("beta angles: t, s1, s2, ... \n" + str(np.rad2deg(self.beta)))
        print("eta: " + str(np.rad2deg(self.eta)))
        plt.show()

    def get_positions(self, x_max, y_max, res_min=1, dist_min=1, plot=False):
        """
        Generates random positions for rx, tx, n_static objects in 2D.
        x_max: maximum x coordinate [m],
        y_max: maximum y coordinate [m],
        res_min: minimum distance resolution [m],
        dist_min: minimum distance between two reflector/scatterer [m].

        returns an array of positions coordinates [rx,tx,t,s1,s2]
        """
        if self.l == 0.06 or self.l == 0.0107:
            res_min = 5
        #     x_max = 4*x_max
        #     y_max = 4*y_max
        alpha = np.random.uniform(
            0, 2 * np.pi
        )  # speed direction w.r.t. positive x-axis
        self.vrx = [
            self.v_rx * np.cos(alpha),
            self.v_rx * np.sin(alpha),
        ]  # speed vector
        th = np.arccos(
            3e8 / (2 * self.B * res_min)
        )  # threshold to check the system minimum distance resolution
        assert 2 * th < np.pi and 2 * (2 * np.pi - th) > 3 * np.pi
        self.positions[0, :] = [x_max, y_max]
        self.positions[1, :] = [0, 0]

        self.paths["delay"][0] = (
            self.dist(self.positions[0, :], self.positions[1, :]) / 3e8
        )  # LoS delay
        for i in range(2, self.n_static + 3):
            x = np.random.uniform(0, x_max)
            y = np.random.uniform(0, y_max)
            beta = np.pi
            start = time.time()
            while True:
                if time.time() - start > 1:
                    self.get_positions(x_max, y_max)
                    break
                check = []
                for j in range(i):
                    check.append(self.dist(self.positions[j, :], [x, y]) < dist_min)
                if any(check):
                    x = np.random.uniform(0, x_max)
                    y = np.random.uniform(0, y_max)
                    continue
                m_1 = (y - self.positions[1, 1]) / (x - self.positions[1, 0])
                q_1 = y - m_1 * x
                m_2 = -1 / m_1
                q_2 = self.positions[0, 1] - m_2 * self.positions[0, 0]
                x_p = (q_1 - q_2) / (m_2 - m_1)
                y_p = m_1 * x_p + q_1
                d = self.dist([x, y], [x_p, y_p])
                ip = self.dist(self.positions[0, :], [x, y])
                if self.dist(self.positions[1, :], [x_p, y_p]) > self.dist(
                    self.positions[1, :], [x, y]
                ):
                    beta = np.pi - np.arccos(d / ip)
                else:
                    beta = np.arccos(d / ip)
                assert int(ip**2) == int(
                    d**2
                    + (
                        (self.positions[0, 0] - x_p) ** 2
                        + (self.positions[0, 1] - y_p) ** 2
                    )
                )
                if (
                    beta < 2 * th
                    or (beta > np.pi and beta < 3 * np.pi)
                    or beta > 2 * (2 * np.pi - th)
                ):
                    self.beta[i - 2] = beta
                    self.positions[i, :] = x, y
                    # path delay for non-LoS
                    self.paths["delay"][i - 1] = (
                        self.dist(self.positions[1, :], self.positions[i, :])
                        + self.dist(self.positions[i, :], self.positions[0, :])
                    ) / 3e8
                    check = []
                    for j in range(0, i):
                        for k in range(0, i):
                            if k != j:
                                check.append(
                                    abs(self.paths["delay"][j] - self.paths["delay"][k])
                                    > 1 / self.B
                                )  # check all path are separable
                    if all(check):
                        check = []
                        self.compute_AoA(ind=i - 1)
                        for k, j in combinations(range(i), 2):
                            check.append(
                                abs(self.paths["AoA"][k] - self.paths["AoA"][j]) > 0.05
                            )  # AoAs must be different between them
                        if all(check):
                            break
                x = np.random.uniform(0, x_max)
                y = np.random.uniform(0, y_max)
        if self.vrx[0] < 0:
            alpha = alpha + np.pi
        beta = np.arctan(
            (self.positions[0, 1] - self.positions[1, 1])
            / (self.positions[0, 0] - self.positions[1, 0])
        )  # LoS path direction w.r.t. positive x-axis
        if (self.positions[0, 0] - self.positions[1, 0]) < 0:
            beta = beta + np.pi
        alpha = alpha % (2 * np.pi)
        beta = beta % (2 * np.pi)
        if alpha > beta:
            self.eta = alpha - beta
        else:
            self.eta = 2 * np.pi - beta + alpha
        if plot:
            self.plot_pos()

    def generate_bpsk(self, n_sc):
        txsym = np.random.randint(0, 2, n_sc)
        txsym[txsym == 0] = -1
        return txsym

    def load_trn_field(self):
        """
        Loads TRN field adding upsampling with rate self.us
        ts = t / us
        """
        trn_unit = loadmat("cir_estimation_sim/TRN_unit.mat")["TRN_SUBFIELD"].squeeze()
        trn_field = np.zeros(len(trn_unit) * self.us)
        trn_field[:: self.us] = trn_unit
        return trn_field

    def compute_phases(self):
        """
        Computes phases for each path.
        LoS phase initial offset=0.
        """
        self.paths["phase"][0] = self.v_rx / self.l * np.cos(self.eta)  # LoS
        self.paths["phase"][1] = self.fd + self.v_rx / self.l * np.cos(
            self.paths["AoA"][1] - self.eta
        )  # target
        for i in range(2, len(self.paths["phase"])):
            self.paths["phase"][i] = (
                self.v_rx / self.l * np.cos(self.paths["AoA"][i] - self.eta)
            )  # static

    def get_delays(self):
        """
        Computes delays for each path.
        """
        paths = np.zeros(len(self.paths["delay"]))
        paths[0] = self.dist(self.positions[0, :], self.positions[1, :]) / 3e8
        for i in range(1, len(paths)):
            paths[i] = (
                self.dist(self.positions[1, :], self.positions[i + 1, :])
                + self.dist(self.positions[i + 1, :], self.positions[0, :])
            ) / 3e8
        return paths

    def path_loss(self):
        """
        Returns the attenuation due to path loss (LoS).
        """
        return (
            self.G_tx
            * self.G_rx
            * self.l
            / ((4 * np.pi * self.dist(self.positions[0, :], self.positions[1, :])) ** 2)
        ) ** (0.5)

    def radar_eq(self, pos, rcs=1, scatter=False):
        """
        Returns the attenuation due to a reflector at position pos.
        pos: [x,y] Cartesian coordinates.
        """
        G_tx = 1  # 10**(20/10) ## assume tx gain for reflectors
        if scatter:
            a = (
                G_tx
                * self.G_rx
                * self.l**2
                * rcs
                / (
                    (4 * np.pi) ** 3
                    * (
                        self.dist(pos, self.positions[0, :])
                        * self.dist(pos, self.positions[1, :])
                    )
                    ** 2
                )
            ) ** (0.5)
        else:
            a = (
                G_tx
                * self.G_rx
                * self.l**2
                * rcs
                / (
                    (4 * np.pi) ** 3
                    * (
                        self.dist(pos, self.positions[0, :])
                        + self.dist(pos, self.positions[1, :])
                    )
                    ** 2
                )
            ) ** (0.5)
        return a

    def compute_attenuations(self):
        """
        Computes attenuations for each path,
        all paths besides the LoS have also a random phase offset due to reflections.
        """
        self.paths["gain"][0] = self.path_loss()
        for i in range(1, len(self.paths["gain"])):
            self.paths["gain"][i] = self.radar_eq(self.positions[i + 1]) * np.exp(
                1j * np.random.uniform(0, 2 * np.pi)
            )

    def compute_cir(self, init, k, plot=False):
        """
        Computes channel impulse response.
        init: if it is the first cir sample compute phases and attenuations.
        """
        self.get_positions(self.x_max, self.y_max)
        if self.l == 0.005:
            up_cir = np.zeros(256 * self.us).astype(complex)
            cir = np.zeros(256).astype(complex)
        else:
            cir = np.zeros(len(self.tx_signal)).astype(complex)
        if init:
            assert all(self.paths["delay"] == self.get_delays())
            self.compute_phases()
            self.compute_attenuations()
        delays = np.floor(self.paths["delay"] * self.B)
        for i, d in enumerate(delays.astype(int)):
            if self.l == 0.005:
                up_cir[d * self.us] = up_cir[d * self.us] + self.paths["gain"][
                    i
                ] * np.exp(1j * 2 * np.pi * self.T * k * self.paths["phase"][i])
            cir[d] = cir[d] + self.paths["gain"][i] * np.exp(
                1j * 2 * np.pi * self.T * k * self.paths["phase"][i]
            )
        assert np.count_nonzero(cir) == len(
            self.paths["delay"]
        )  # check that all paths are separable
        if plot:
            if self.l == 0.005:
                plt.title("upsampled cir")
                plt.grid()
                plt.stem(abs(up_cir), markerfmt="D")
                plt.show()
            else:
                plt.title("cir")
                plt.grid()
                plt.stem(abs(cir), markerfmt="D")
                plt.show()
        # assuming signal amplitude=1
        if self.l == 0.005:
            self.up_cir = up_cir / abs(self.paths["gain"][0])
        self.cir = cir / abs(self.paths["gain"][0])


if __name__ == "__main__":
    rt = RayTracing(l=0.06, n_static=2)
    rt.compute_cir(init=True, k=1, plot=True)
