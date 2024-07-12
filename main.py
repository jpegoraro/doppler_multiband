import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

CHANNEL_PARAMS = {
    "scatter_amplitudes": [10, 5],
    "velocities": [0.0, 3.0],  # [m/s]
    "delays": [0.0, 20e-9],  # [s]
    "subbands_carriers": [60.48e9, 60.88e9],  # 62.64e9],  # [Hz]
    "subbands_bandwidth": [400e6, 400e6],  # [Hz]
    "sc_spacing": 240e3,  # [Hz]
    "t0": 0.019,  # [s]
    "Tslow": 1e-3,  # [s]
    "slow_time_samples": 64,
    "nominal_CFO": 1e-5,  # [ppm]
    "CFO": "normal",
    "RPO": "uniform",
    "TO": "normal",
}


class ChannelFrequencyResponse:
    def __init__(self, params):
        self.n_paths = len(params["scatter_amplitudes"])
        self.n_subbands = len(params["subbands_carriers"])

        scatter_amplitudes = np.array(params["scatter_amplitudes"])
        scatter_phases = np.array(
            [np.random.normal(0, 1) * 2 * np.pi for _ in range(self.n_paths)]
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

        self.subbands_CFR = {i: None for i in range(self.n_subbands)}
        self.subbands_CIR = {i: None for i in range(self.n_subbands)}

        if params["CFO"] == "normal":
            self.CFOs = np.array(
                [
                    np.random.normal(0, 1, size=(self.slow_time_samples))
                    * params["nominal_CFO"]
                    for _ in range(self.n_subbands)
                ]
            )

        if params["RPO"] == "uniform":
            self.RPOs = np.array(
                [
                    np.random.uniform(0, 1, size=(self.slow_time_samples)) * 2 * np.pi
                    for _ in range(self.n_subbands)
                ]
            )

        if params["TO"] == "normal":
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

            carrier_freq_mod = (
                self.subbands_carriers[i] if i == 1 else self.subbands_carriers[i]
            )

            # ugly loop to be sure it works
            for j in range(self.n_paths):
                for n in range(self.fast_time_samples):
                    for k in range(self.slow_time_samples):
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
                                * carrier_freq_mod
                                * (self.t0 + k * self.Tslow)
                            )
                        )

            # cfo_component = np.exp(2j * np.pi * self.CFOs[i])
            # self.subbands_CFR[i] *= cfo_component.reshape(1, -1)

            # rpo_component = np.exp(1j * self.RPOs[i])
            # self.subbands_CFR[i] *= rpo_component.reshape(1, -1)

            # fgrid = np.arange(self.fast_time_samples) * self.sc_spacing
            # to_component = np.exp(-2j * np.pi * self.TOs[i] * fgrid.reshape(-1, 1))
            # self.subbands_CFR[i] *= to_component

    def compute_CIR(self):
        for i in range(self.n_subbands):
            self.subbands_CIR[i] = np.fft.ifft(self.subbands_CFR[i], axis=0)

            # plt.imshow(np.abs(self.subbands_CIR[i]), aspect="auto")
            # plt.show()

    def get_carrier_phase_vector(self):
        for i in range(self.n_subbands):
            fast_time_grid = np.arange(self.fast_time_samples) * self.Tfast
            self.carrier_phase_vectors[i] = (
                -2 * np.pi * self.subbands_carriers[i] * fast_time_grid
            )

            # plt.plot(np.exp(1j * self.carrier_phase_vectors[i]).real)
            # plt.show()


class SignalProcessor:
    def __init__(self, subbands_channel):
        self.subb_chn = subbands_channel

    def TO_compensation(self):
        for i in range(self.subb_chn.n_subbands):
            for k in range(self.subb_chn.slow_time_samples):
                cir = self.subb_chn.subbands_CIR[i][:, k]
                pdp = np.abs(cir) ** 2
                peaks, _ = sp.signal.find_peaks(pdp, height=0.05 * np.max(pdp))
                rolled_cir = np.roll(cir, -peaks[0])
                self.subb_chn.subbands_CIR[i][:, k] = rolled_cir
                self.subb_chn.subbands_CFR[i][:, k] = np.fft.fft(
                    rolled_cir, axis=0, n=self.subb_chn.fast_time_samples
                )

    def CFO_compensation(self):
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
            st_cir_dft = np.fft.fft(self.subb_chn.subbands_CIR[i], axis=1)
            approx_range_prof = np.abs(self.subb_chn.subbands_CIR[i][:, 0])
            peaks, values = sp.signal.find_peaks(
                approx_range_prof, height=0.05 * np.max(approx_range_prof)
            )
            # plot the aprox range profile and the peaks
            # plt.close()
            # fig, ax = plt.subplots(1, 2)
            # ax[0].plot(np.abs(st_cir_dft[:, 0]))
            # ax[1].plot(np.angle(st_cir_dft[:, 0]))
            # plt.show()
            # get max doppler peak for each fast-time bin
            max_doppler_idx = np.argmax(np.abs(st_cir_dft), axis=1)
            # # get corresponding phases
            # # initial_doppler_phases = np.angle(
            # #     st_cir_dft[np.arange(st_cir_dft.shape[0]), max_doppler_idx]
            # # )
            initial_doppler_phases = np.zeros_like(max_doppler_idx)
            initial_doppler_phases[peaks[0]] = np.angle(
                st_cir_dft[peaks[0], max_doppler_idx[peaks[0]]]
            )
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
            carrier_phase_vec = np.zeros_like(initial_doppler_phases)
            carrier_phase_vec[peaks[0]] = self.subb_chn.carrier_phase_vectors[i][
                peaks[0]
            ]
            corrected_cir = (
                np.fft.ifft(st_cir_dft, axis=1)
                * np.exp(-1j * initial_doppler_phases[:, np.newaxis])
                * np.exp(-1j * carrier_phase_vec[:, np.newaxis])
            )
            # corrected_cir = np.fft.ifft(st_cir_dft, axis=1)
            self.subb_chn.subbands_CIR[i] = corrected_cir
            self.subb_chn.subbands_CFR[i] = np.fft.fft(
                self.subb_chn.subbands_CIR[i], axis=0
            )
            # print()
            # plt.imshow(np.abs(st_cir_dft)[:30], aspect="auto")
            # plt.show()


if __name__ == "__main__":

    cfr = ChannelFrequencyResponse(CHANNEL_PARAMS)
    cfr.generate_subbands_CFR()
    cfr.compute_CIR()
    cfr.get_carrier_phase_vector()

    proc = SignalProcessor(cfr)
    proc.TO_compensation()
    proc.CFO_compensation()

    cfr1 = cfr.subbands_CFR[0][:, 0]
    grid1 = np.arange(cfr.fast_time_samples) * cfr.sc_spacing + cfr.subbands_carriers[0]
    cfr2 = cfr.subbands_CFR[1][:, 0]
    grid2 = np.arange(cfr.fast_time_samples) * cfr.sc_spacing + cfr.subbands_carriers[1]
    cfr2 = cfr.subbands_CFR[1][:, 0]
    plt.plot(grid1, cfr1.real, "r")
    plt.plot(grid2, cfr2.real, "b")

    proc.Doppler_compensation()

    cfr1 = cfr.subbands_CFR[0][:, 0]
    cfr2 = cfr.subbands_CFR[1][:, 0]
    plt.plot(grid1, cfr1.real, "--r")
    plt.plot(grid2, cfr2.real, "--b")
    plt.show()

    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(grid1, np.abs(cfr.subbands_CFR[0][:, 0]))
    # ax[0].plot(grid2, np.abs(cfr.subbands_CFR[1][:, 0]))
    # ax[1].plot(np.angle(cfr.subbands_CFR[0][:, 0]))
    # ax[1].plot(np.angle(cfr.subbands_CFR[1][:, 0]))
    # plt.show()
