from pathlib import Path
import numpy as np
import scipy as sp
from scipy.io import loadmat
#import h5py
import matplotlib.pyplot as plt
#from utils import *
import pickle
import tqdm
from scipy.signal import correlate, correlation_lags
from argparse import ArgumentParser


def normalizeNdB(s, dB=False):
    snorm = s / np.max(s)
    if dB:
        sout = 10 * np.log10(snorm + 1e-20)
    else:
        sout = snorm
    return sout


def load_trn_unit(path):
    trn = loadmat(path / "TRN_unit.mat")["TRN_FIELD"].squeeze()
    return trn


def load_Golay_seqs(path):
    Ga = loadmat(path / "Ga128_rot_2sps.mat")["Ga128_rot_2sps"].squeeze()
    Gb = loadmat(path / "Gb128_rot_2sps.mat")["Gb128_rot_2sps"].squeeze()
    return Ga, Gb


def oversamp(indata, sample):
    nsymb = np.size(indata)
    out = np.zeros(nsymb * sample, dtype=complex)
    for i in range(0, nsymb, 1):
        out[i * sample] = indata[i]
    return out


def pass_through_channel(signal, h):
    y = np.convolve(signal, h, mode="full")
    return y


def simulate_channel(params, ncir=256, ncfr=256, add_to=True, add_cfo=True):
    delays = 2 * params["ranges"] / 3e8
    amps = params["amplitudes"]
    taumax = params["dtau"] * ncir
    df = params["B"] / ncfr

    H_clean = np.zeros(ncfr, dtype=complex)
    fgrid = np.arange(ncfr) * df
    alpha = 0
    for i in range(len(delays)):
        if i == 0:  # LOS path
            path = amps[i] * np.exp(-1j * 2 * np.pi * fgrid * delays[i])
        else:
            path = amps[i] * np.exp(-1j * 2 * np.pi * fgrid * delays[i])
        H_clean += path

    if add_to:
        H_to = H_clean * np.exp(1j * 2 * np.pi * fgrid * params["TO"])

    if add_cfo:
        Htot = H_to * np.exp(1j * 2 * np.pi * params["CFO"])

    cir_clean = np.fft.ifft(H_clean * np.hanning(ncfr))
    cir_tot = np.fft.ifft(Htot * np.hanning(ncfr))

    return (
        cir_clean / np.abs(amps[0]),
        cir_tot / np.abs(amps[0]),
        df,
    )


def get_delta(tau, n, dt):
    grid = np.arange(n) * dt
    approx = np.argmin(np.abs(grid - tau))
    delta = np.zeros(n)
    delta[approx] = 1
    return delta


def estimate_CIR(signal, Ga, Gb):
    Gacor = correlate(signal, Ga)
    Gbcor = correlate(signal, Gb)
    # get index of the 0 lag correlation
    lags = correlation_lags(len(signal), len(Ga))
    start = np.argwhere(lags == 0)[0][0]
    # cut starting at 0 lag
    Gacor2 = Gacor[start:]
    Gbcor2 = Gbcor[start:]
    # align the a and b sequences
    Gacor3 = Gacor2[: -2 * len(Ga)]
    Gbcor3 = Gbcor2[len(Ga) : -len(Ga)]
    # extract the 3 Golay subsequences
    Gacor4 = np.stack(
        [
            Gacor3[: len(Ga) * 2],
            Gacor3[len(Ga) * 2 : len(Ga) * 4],
            Gacor3[len(Ga) * 4 : len(Ga) * 6],
        ],
        axis=1,
    )
    Gbcor4 = np.stack(
        [
            Gbcor3[: len(Gb) * 2],
            Gbcor3[len(Gb) * 2 : len(Gb) * 4],
            Gbcor3[len(Gb) * 4 : len(Gb) * 6],
        ],
        axis=1,
    )

    # pair complementary sequences
    # +Ga128, -Gb128, +Ga128. +Gb128, +Ga128, -Gb128
    a_part = Gacor4 * np.array([[1, 1, 1]])
    b_part = Gbcor4 * np.array([[-1, 1, -1]])
    ind_h = a_part + b_part

    # add individual results
    h128 = ind_h.sum(axis=1)
    return h128


def gen_noise(n, params):
    snr_lin = 10 ** (params["SNR"] / 10)
    noise_std = np.sqrt(params["A_signal"] ** 2 / snr_lin)
    noise = np.random.normal(0, noise_std / np.sqrt(2), size=n) + 1j * np.random.normal(
        0, noise_std / np.sqrt(2), size=n
    )
    return noise


def get_snr(h, h_n):
    """
        compute the SNR after the channel [dB].
        h: real channel impulse response.
        h_n:  noisy (estimated) channel impulse response.
    """
    h = h[:100]
    h_n = h_n[:100]
    h = h / np.max(np.abs(h))
    h_n = h_n / np.max(np.abs(h_n))
    ind = np.argmax(np.abs(h))
    signal_p = np.var(h)
    noise_p = np.mean(np.abs(h_n[ind]-h[ind])**2)
    snr = signal_p/noise_p
    return 10*np.log10(snr)


if __name__ == "__main__":
    params = {
        "A_signal": 1,
        "c": 3e8,
        "B": 1.76e9,
        "dtau": 1 / 1.76e9,
        "SNR": -5,
        "cir_bins": 256,
        "pulse_shaping": None,
        "amplitudes": None,
        "ranges": np.array([2, 3.22, 5.449, 5.85]),
        "AoAs": np.array([0.0, 0.0, 0.0]),
        "us_factor": 1,
        "TO": 1.6 * 1 / 1.76e9,
        "CFO": np.pi,
        "TOstd": 20,
        "simlen": 10000,
        "plot": False,
    }

    path = Path("cir_estimation_sim")

    trn = load_trn_unit(path)
    signal = params["A_signal"] * np.pad(trn, params["cir_bins"])[params["cir_bins"] :]
    Ga, Gb = load_Golay_seqs(path)

    snrlist = [0, 5, 10, 15]
    g = np.zeros(len(snrlist))
    for k, s in enumerate(snrlist):
        print("####################################")
        print(f"Simulating SNR = {s}")
        temp_g = []
        for i in range(params["simlen"]):
            print("iteration: ", i, end="\r")

            npaths = np.random.choice(np.arange(2, 10))
            params["ranges"] = np.zeros(npaths)
            params["ranges"][0] = np.random.uniform(1, 2)
            params["ranges"][1:] = np.sort(
                np.random.uniform(params["ranges"][0], 5, npaths - 1)
            )

            params["RCSs"] = np.random.uniform(
                10 ** (-10 / 10), 10 ** (10 / 10), size=npaths
            )
            # params["RCSs"] = 1
            wavelength = params["c"] / 60e9
            params["amplitudes"] = (
                (wavelength / 4 * np.pi)
                * np.sqrt(params["RCSs"])
                / ((params["ranges"] / 2) ** 2)
            )
            params["amplitudes"][0] = (wavelength / 4 * np.pi) / (params["ranges"][0])

            params["TO"] = (
                0#-np.random.uniform(0, params["TOstd"]) * params["dtau"]
            )  # off-grid

            params["CFO"] = 0#np.random.uniform(0, 0.5)

            params["SNR"] = s

            (
                true_h_clean,
                true_h_off,
                df,
            ) = simulate_channel(params)

            y_clean = pass_through_channel(signal, true_h_clean)
            y_off = pass_through_channel(signal, true_h_off)

            noise1 = gen_noise(len(y_clean), params)
            noise2 = gen_noise(len(y_off), params)

            y_clean += noise1
            y_off += noise2

            h_est_clean = estimate_CIR(y_clean, Ga, Gb)
            h_est_off = estimate_CIR(y_off, Ga, Gb)
            
            after_channel_snr = get_snr(true_h_off, h_est_off)
            temp_g.append(10**((after_channel_snr-s)/10)) # linear gain

            # example plots
            if params["plot"]:
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(
                    np.abs(true_h_clean) / np.max(np.abs(true_h_clean)), label="True"
                )
                ax[0].plot(
                    np.abs(h_est_clean) / np.max(np.abs(h_est_clean)), label="Estimated"
                )
                ax[0].set_title("Clean")
                ax[0].set_xlabel("Delay bins")
                ax[0].set_ylabel("CIR magnitude")
                ax[0].legend()

                ax[1].plot(np.abs(true_h_off) / np.max(np.abs(true_h_off)), label="True")
                ax[1].plot(np.abs(h_est_off) / np.max(np.abs(h_est_off)), label="Estimated")
                ax[1].set_title("With offset")
                ax[1].set_xlabel("Delay bins")
                ax[1].set_ylabel("CIR magnitude")
                ax[1].legend()

                plt.tight_layout()
                plt.show()
        g[k] = np.mean(temp_g)
        print('Average linear gain after the channel : ' + str(g[k]))
