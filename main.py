import numpy as np
from utils import *
import matplotlib.pyplot as plt

CHANNEL_PARAMS = {
    "scatter_amplitudes": [10, 7],
    "velocities": [0.0, 3.0],  # [m/s]
    "delays": [0.0, 20e-9],  # [s]
    "subbands_carriers": [60.48e9, 60.88e9],  # 62.64e9],  # [Hz]
    "subbands_bandwidth": [400e6, 400e6],  # [Hz]
    "fast_time_oversampling": 256,
    "slow_time_oversampling": 256,
    "sc_spacing": 240e3,  # [Hz]
    "t0": 0.019,  # [s]
    "Tslow": 0.9e-3,  # [s]
    "slow_time_samples": 64,
    "nominal_CFO": 1e-5,  # [ppm]
    "CFO": True,
    "RPO": True,
    "TO": False,
    "noise_var": 0.0,
}


class ChannelFrequencyResponse:
    def __init__(self, params):
        self.n_paths = len(params["scatter_amplitudes"])
        self.n_subbands = len(params["subbands_carriers"])

        scatter_amplitudes = np.array(params["scatter_amplitudes"])
        scatter_phases = np.array(
            [0.0] + [np.random.uniform(-1, 1) * np.pi for _ in range(self.n_paths - 1)]
        )
        # scatter_phases = np.zeros_like(scatter_amplitudes)
        self.scatter_coeff = scatter_amplitudes * np.exp(1j * scatter_phases)

        self.velocities = np.array(params["velocities"])
        self.delays = np.array(params["delays"])
        self.subbands_carriers = np.array(params["subbands_carriers"])
        self.subbands_bandwidth = np.array(params["subbands_bandwidth"])
        self.sc_spacing = params["sc_spacing"]
        self.fast_time_samples = int(self.subbands_bandwidth[0] / self.sc_spacing)
        self.slow_time_samples = params["slow_time_samples"]
        self.Tslow = params["Tslow"]
        self.Tfast = 1 / self.subbands_bandwidth[0]
        self.t0 = params["t0"]
        self.carrier_phase_vectors = {i: None for i in range(self.n_subbands)}
        self.noise_var = params["noise_var"]

        self.fast_time_oversampling = params["fast_time_oversampling"]
        self.slow_time_oversampling = params["slow_time_oversampling"]

        self.subbands_CFR = {i: None for i in range(self.n_subbands)}
        self.subbands_CIR = {i: None for i in range(self.n_subbands)}

        if params["CFO"]:
            self.CFOs = np.array(
                [
                    np.random.normal(0, 1, size=(self.slow_time_samples))
                    * params["nominal_CFO"]
                    for _ in range(self.n_subbands)
                ]
            )

        if params["RPO"]:
            self.RPOs = np.array(
                [
                    np.random.uniform(0, 1, size=(self.slow_time_samples)) * 2 * np.pi
                    for _ in range(self.n_subbands)
                ]
            )

        if params["TO"]:
            # self.TOs = np.array(
            #     [
            #         np.random.choice(np.arange(10), size=(1, self.slow_time_samples))
            #         * self.Tfast
            #         for _ in range(self.n_subbands)
            #     ]
            # )
            self.TOs = np.array(
                [
                    np.random.uniform(0, 1, size=(1, self.slow_time_samples))
                    * 10
                    * self.Tfast
                    for _ in range(self.n_subbands)
                ]
            )

    def generate_subbands_CFR(self):

        for i in range(self.n_subbands):
            self.subbands_CFR[i] = np.zeros(
                (self.fast_time_samples, self.slow_time_samples), dtype=complex
            )

            # carrier_freq_mod = (
            #     self.subbands_carriers[i] if i == 1 else self.subbands_carriers[i]
            # )

            # TODO: make this more efficient
            for j in range(self.n_paths):
                for n in range(self.fast_time_samples):
                    for k in range(self.slow_time_samples):
                        # generate complex noise sample
                        noise = (1 / np.sqrt(2)) * np.random.normal(
                            0, np.sqrt(self.noise_var)
                        ) + 1j * np.random.normal(0, np.sqrt(self.noise_var))
                        # generate CFR sample
                        self.subbands_CFR[i][n, k] += (
                            self.scatter_coeff[j]
                            * np.exp(  # delay term
                                -2j
                                * np.pi
                                * (n * self.sc_spacing + self.subbands_carriers[i])
                                * self.delays[j]
                            )
                            * np.exp(  # doppler term
                                2j
                                * np.pi
                                * (self.velocities[j] / 3e8)
                                * self.subbands_carriers[i]
                                * (self.t0 + k * self.Tslow)
                            )
                        ) + noise
                # print()
            # check if class has parameter TO

            if hasattr(self, "CFOs"):
                cfo_component = np.exp(2j * np.pi * self.CFOs[i])
                self.subbands_CFR[i] *= cfo_component.reshape(1, -1)

            if hasattr(self, "RPOs"):
                rpo_component = np.exp(1j * self.RPOs[i])
                self.subbands_CFR[i] *= rpo_component.reshape(1, -1)

            if hasattr(self, "TOs"):
                fgrid = np.arange(self.fast_time_samples) * self.sc_spacing
                to_component = np.exp(-2j * np.pi * self.TOs[i] * fgrid.reshape(-1, 1))
                self.subbands_CFR[i] *= to_component

    def compute_CIR(self):
        for i in range(self.n_subbands):
            self.subbands_CIR[i] = np.fft.ifft(self.subbands_CFR[i], axis=0)
            # print()

            # plt.imshow(np.abs(self.subbands_CIR[i]), aspect="auto")
            # plt.show()

    def get_carrier_phase_vector(self):
        for i in range(self.n_subbands):
            fast_time_grid = np.arange(self.fast_time_samples) * self.Tfast
            self.carrier_phase_vectors[i] = (
                -2 * np.pi * self.subbands_carriers[i] * fast_time_grid
            ) % (2 * np.pi)

            # plt.plot(np.exp(1j * self.carrier_phase_vectors[i]).real)
            # plt.show()


class SignalProcessor:
    def __init__(self, subbands_channel):
        self.subb_chn = subbands_channel

    def TO_compensation(self):
        for i in range(self.subb_chn.n_subbands):
            for k in range(self.subb_chn.slow_time_samples):
                cir = np.fft.ifft(
                    self.subb_chn.subbands_CFR[i][:, k],
                    n=self.subb_chn.fast_time_samples
                    * self.subb_chn.fast_time_oversampling,
                )
                pdp = np.abs(cir) ** 2
                peaks, _ = find_peaks_mod(pdp, height=0.2 * np.max(pdp))
                # rolled_cir = np.roll(cir, -peaks[0])
                tau_o = peaks[0] * (
                    self.subb_chn.Tfast / self.subb_chn.fast_time_oversampling
                )
                fgrid = (
                    np.arange(self.subb_chn.fast_time_samples)
                    * self.subb_chn.sc_spacing
                )
                to_component = np.exp(2j * np.pi * tau_o * fgrid)
                clean_cfr = self.subb_chn.subbands_CFR[i][:, k] * to_component
                # self.subb_chn.subbands_CIR[i][:, k] = rolled_cir
                # self.subb_chn.subbands_CFR[i][:, k] = np.fft.fft(
                #     rolled_cir, axis=0, n=self.subb_chn.fast_time_samples
                # )
                self.subb_chn.subbands_CFR[i][:, k] = clean_cfr
                self.subb_chn.subbands_CIR[i][:, k] = np.fft.ifft(
                    clean_cfr, axis=0, n=self.subb_chn.fast_time_samples
                )
                # plt.imshow(np.abs(self.subb_chn.subbands_CIR[i])[:50], aspect="auto")
                # plt.show()

    def PO_compensation(self):
        for i in range(self.subb_chn.n_subbands):
            anchor_phase = np.angle(self.subb_chn.subbands_CIR[i][0, :])
            compensation_term = np.exp(-1j * anchor_phase)

            self.subb_chn.subbands_CIR[i] *= compensation_term.reshape(1, -1)
            self.subb_chn.subbands_CFR[i] *= compensation_term.reshape(1, -1)

            # plt.imshow(np.abs(self.subb_chn.subbands_CIR[i]), aspect="auto")
            # plt.show()

    def Doppler_compensation(self):
        # slow-time DFT
        for i in range(self.subb_chn.n_subbands):
            st_cir_dft = np.fft.fft(
                self.subb_chn.subbands_CIR[i],
                axis=1,
                n=self.subb_chn.slow_time_samples
                * self.subb_chn.slow_time_oversampling,
            )  # the oversampling is essential for the phase correction!

            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(np.abs(st_cir_dft)[:50], aspect="auto")
            # ax[1].imshow(np.angle(st_cir_dft)[:50], aspect="auto")
            # plt.show()

            approx_range_prof = np.abs(self.subb_chn.subbands_CIR[i][:, 0])
            peaks, values = sp.signal.find_peaks(
                approx_range_prof, height=0.05 * np.max(approx_range_prof)
            )
            # plot the aprox range profile and the peaks
            # plt.close()
            # fig, ax = plt.subplots()
            # ax.plot(approx_range_prof[:50])
            # ax.plot(peaks, approx_range_prof[peaks], "x")
            # plt.show()
            # get max doppler peak for each fast-time bin
            max_doppler_idx = np.argmax(np.abs(st_cir_dft), axis=1)
            # # get corresponding phases
            # # initial_doppler_phases = np.angle(
            # #     st_cir_dft[np.arange(st_cir_dft.shape[0]), max_doppler_idx]
            # # )
            initial_doppler_phases = np.zeros((len(max_doppler_idx),))
            initial_doppler_phases[peaks[0]] = np.angle(
                st_cir_dft[peaks[0], max_doppler_idx[peaks[0]]]
            )
            # print(np.angle(st_cir_dft[peaks[0], max_doppler_idx[peaks[0]]]))
            # remove carrier phase part
            # plt.plot(self.subb_chn.carrier_phase_vectors[i])
            # plt.plot(initial_doppler_phases)
            # plt.show()
            # initial_doppler_phases -= self.subb_chn.carrier_phase_vectors[i]
            # apply phase correction
            # st_cir_dft *= np.exp(-1j * initial_doppler_phases[:, np.newaxis])
            # plt.imshow(np.abs(st_cir_dft)[:30], aspect="auto")
            # plt.show()
            # test = np.fft.ifft(st_cir_dft, axis=1)
            # plt.imshow(np.abs(test), aspect="auto")
            # plt.show()
            carrier_phase_vec = np.zeros((len(max_doppler_idx),))
            carrier_phase_vec[peaks[0]] = self.subb_chn.carrier_phase_vectors[i][
                peaks[0]
            ]
            correction_term = np.exp(-1j * initial_doppler_phases) * np.exp(
                -1j * carrier_phase_vec
            )
            corrected_cir = (
                self.subb_chn.subbands_CIR[i] * correction_term[:, np.newaxis]
            )

            # corrected_cir = np.fft.ifft(st_cir_dft, axis=1)
            self.subb_chn.subbands_CIR[i] = corrected_cir
            self.subb_chn.subbands_CFR[i] = np.fft.fft(
                self.subb_chn.subbands_CIR[i], axis=0
            )
            # print()
            # plt.imshow(np.abs(st_cir_dft)[:30], aspect="auto")
            # plt.show()
        cfr1 = cfr.subbands_CFR[0][:, 0]
        cfr2 = cfr.subbands_CFR[1][:, 0]
        phdiff = np.angle(cfr1[-1] * cfr2[0])
        phdiff = phdiff if np.abs(phdiff) < np.pi else 2 * np.pi - np.abs(phdiff)
        # if phdiff > 0.2:
        #     print()


if __name__ == "__main__":
    diffs = []
    for kk in range(100):

        cfr = ChannelFrequencyResponse(CHANNEL_PARAMS)
        cfr.generate_subbands_CFR()
        cfr.compute_CIR()
        cfr.get_carrier_phase_vector()

        proc = SignalProcessor(cfr)
        proc.TO_compensation()
        proc.PO_compensation()

        cfr1 = cfr.subbands_CFR[0][:, 0]
        grid1 = (
            np.arange(cfr.fast_time_samples) * cfr.sc_spacing + cfr.subbands_carriers[0]
        )
        cfr2 = cfr.subbands_CFR[1][:, 0]
        grid2 = (
            np.arange(cfr.fast_time_samples) * cfr.sc_spacing + cfr.subbands_carriers[1]
        )
        cfr2 = cfr.subbands_CFR[1][:, 0]
        plt.plot(grid1, np.angle(cfr1), "r")
        plt.plot(grid2, np.angle(cfr2), "b")

        proc.Doppler_compensation()

        cfr1 = cfr.subbands_CFR[0][:, 0]
        cfr2 = cfr.subbands_CFR[1][:, 0]

        phdiff = np.angle(cfr1[-1]) - np.angle(cfr2[0])
        phdiff = phdiff if np.abs(phdiff) < np.pi else 2 * np.pi - np.abs(phdiff)
        diffs.append(phdiff)
        print(f"Phase diff. {phdiff:.2f}")
        plt.plot(grid1, np.angle(cfr1), "--r")
        plt.plot(grid2, np.angle(cfr2), "--b")
        plt.xlim([6.085e10, 6.09e10])
        plt.show()

        # plt.savefig(f"figs/fig_{kk}.png")
        # plt.close()

    print(f"Avg. phase diff. {np.mean(np.abs(phdiff)):.2f}")

    # plt.hist(diffs)
    # plt.show()

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(grid1, np.abs(cfr.subbands_CFR[0][:, 0]))
    # ax[0].plot(grid2, np.abs(cfr.subbands_CFR[1][:, 0]))
    # ax[1].plot(np.angle(cfr.subbands_CFR[0][:, 0]))
    # ax[1].plot(np.angle(cfr.subbands_CFR[1][:, 0]))
    # plt.show()
