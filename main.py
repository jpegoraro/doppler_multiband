import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

CHANNEL_PARAMS = {
    "scatter_amplitudes": [10, 5],
    "velocities": [0.0, 3.0],  # [m/s]
    "delays": [0.0, 20e-9],  # [s]
    "subbands_carriers": [60.48e9, 62.64e9],  # [Hz]
    "subbands_bandwidth": [400e6, 400e6],  # [Hz]
    "sc_spacing": 240e3,  # [Hz]
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
            fgrid = np.arange(self.fast_time_samples) * self.sc_spacing

            # ugly loop to be sure it works
            for j in range(self.n_paths):
                for n in range(self.fast_time_samples):
                    for k in range(self.slow_time_samples):
                        self.subbands_CFR[i][n, k] += (
                            self.scatter_coeff[j]
                            * np.exp(  # delay term
                                -2j
                                * np.pi
                                * (fgrid[n] + self.subbands_carriers[i])
                                * self.delays[j]
                            )
                            * np.exp(  # doppler term
                                2j
                                * np.pi
                                * (self.velocities[j] / 3e8)
                                * self.subbands_carriers[i]
                                * np.arange(self.slow_time_samples)[k]
                                * self.Tslow
                            )
                        )

                # path_delay_component = self.scatter_coeff[j] * np.exp(
                #     -2j * np.pi * (fgrid + self.subbands_carriers[i]) * self.delays[j]
                # )
                # self.subbands_CFR[i] += np.tile(
                #     path_delay_component, (self.slow_time_samples, 1)
                # ).T

                # # plt.imshow(self.subbands_CFR[i].real, aspect="auto")
                # # plt.show()

                # path_doppler_component = np.exp(
                #     2j
                #     * np.pi
                #     * (self.velocities[j] / 3e8)
                #     * self.subbands_carriers[i]
                #     * np.arange(self.slow_time_samples)
                #     * self.Tslow
                # )  # add doppler shift

                # self.subbands_CFR[i] *= np.tile(
                #     path_doppler_component, (self.fast_time_samples, 1)
                # )

            cfo_component = np.exp(2j * np.pi * self.CFOs[i])
            self.subbands_CFR[i] *= cfo_component.reshape(1, -1)

            rpo_component = np.exp(1j * self.RPOs[i])
            self.subbands_CFR[i] *= rpo_component.reshape(1, -1)

            to_component = np.exp(-2j * np.pi * self.TOs[i] * fgrid.reshape(-1, 1))
            self.subbands_CFR[i] *= to_component

            # plt.imshow(np.real(self.subbands_CFR[i]), aspect="auto")
            # plt.show()

    def compute_CIR(self):
        for i in range(self.n_subbands):
            self.subbands_CIR[i] = np.fft.ifft(self.subbands_CFR[i], axis=0)

            # plt.imshow(np.abs(self.subbands_CIR[i]), aspect="auto")
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
                self.subb_chn.subbands_CFR[i][:, k] = np.fft.fft(rolled_cir, axis=0)

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

            plt.imshow(np.abs(st_cir_dft)[:30], aspect="auto")
            plt.show()


if __name__ == "__main__":

    cfr = ChannelFrequencyResponse(CHANNEL_PARAMS)
    cfr.generate_subbands_CFR()
    cfr.compute_CIR()

    proc = SignalProcessor(cfr)
    proc.TO_compensation()
    proc.CFO_compensation()
    proc.Doppler_compensation()
