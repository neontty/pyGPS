import numpy as np
from constants import *
from pygps.ca_code import *
from scipy import signal
from matplotlib import pyplot as plt


class CoarseAcquisition:
    def __init__(self, f_sampling,
                 coherent_integration_periods, noncoherent_integration_periods,
                 doppler_max=5e3, doppler_bins=21,
                 ):

        self.doppler_values = np.linspace(-doppler_max, doppler_max, doppler_bins)
        self.f_sampling = f_sampling
        self.coherent_integration_periods = coherent_integration_periods
        self.noncoherent_integration_periods = noncoherent_integration_periods

        self.samples_per_coherent_period = self.f_sampling * L1_CODE_PERIOD * self.coherent_integration_periods

        self.reference_signals = {}

        self.generate_all_l1_ca_reference_fft()


    # coherent integration gain is 10*log10(n_coherent_periods)
    # noncoherent integration gain is ?
    # TODO remember what the gain is for noncoherent integration
    # we must rely on noncoherent more than coherent in order to get most of our gain because the nav bits encoded
    # over top of the spreading code would cause a long coherent integration to lose magnitude when parts of our
    # reference signal end up being anti-correlated with the real signal while the nav bits flip.
    def generate_l1_coarse_acquisition_reference(self, prn):

        ##### generate reference signals for correlation:
        t = np.arange(1,self.samples_per_coherent_period +1) / self.f_sampling

        # generate a matrix of constant waves, each column representing a doppler to mix against the baseband signal
        doppler_carriers = np.exp(-1j * 2 * np.pi * np.matmul(self.doppler_values.reshape(-1, 1), t.reshape(1, -1)))

        # calculate new chipping rates based on each doppler rates:
        f_code_doppler = L1_CHIPPING_RATE * (1 + (self.doppler_values / L1_FREQ))

        ca_code = L1_CA_CODES[prn]

        chips_per_sample = f_code_doppler / self.f_sampling

        resampling_index_matrix = np.repeat(np.arange(int(self.samples_per_coherent_period))[:, np.newaxis],
                                            len(self.doppler_values), axis=1)
        resampling_index_matrix = (resampling_index_matrix * chips_per_sample % len(ca_code)).astype(np.int)

        doppler_affected_ca_code = ca_code[resampling_index_matrix.transpose()]

        # modulate resampled CA code onto doppler reference waves. remember to shift our ca code around -1, 1 to cause a phase shift
        reference_signals = doppler_carriers * (doppler_affected_ca_code * 2 - 1)

        # now we have our reference signals.
        return reference_signals.transpose()

    def generate_all_l1_ca_reference_fft(self):
        for prn in L1_PRNS:
            self.reference_signals[prn] = np.conj(np.fft.fft(self.generate_l1_coarse_acquisition_reference(prn), axis=0))

    def plot_correlation_map(self, corrmap):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Make data.
        X = self.doppler_values
        Y = np.arange(self.samples_per_coherent_period)
        X, Y = np.meshgrid(X, Y)
        Z = corrmap

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def l1_coarse_acquire(self, baseband_data, prn):
        if prn not in self.reference_signals:
            raise ValueError

        # # resize our incoming signal so that it's a multiple of the number of coherent period samples by non-coherent period
        # baseband_data = baseband_data[: int(self.samples_per_coherent_period * self.noncoherent_integration_periods)]

        # reshape our incoming signal by number of non-coherent periods
        baseband_data = baseband_data.reshape((-1, self.noncoherent_integration_periods))

        correlation_map = np.zeros((int(self.samples_per_coherent_period), len(self.doppler_values)), dtype=complex)

        for nch_idx in range(0, self.noncoherent_integration_periods):

            # convolve our reference signals by our base_signal blocks by multiplying in the frequency domain:
            data_fft = np.fft.fft(baseband_data[:, nch_idx])

            data_fft_repeated = np.repeat(data_fft[:, np.newaxis], len(self.doppler_values), axis=1)

            correlation_map += np.abs(np.fft.ifft(data_fft_repeated * self.reference_signals[prn], axis=0))

        correlation_map = correlation_map ** 2

        self.plot_correlation_map(correlation_map)

        [code_phase_offset_in_samples, doppler_bin_idx] = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
        coarse_doppler_est = self.doppler_values[doppler_bin_idx]

        # TODO use a discriminator to detect if acquisition was successful

        # TODO: FUTURE: would be cool to try to train a binary classifier to detect signal presence using the
        #  correlation map as a rastered image for input.

        # return our estimated doppler error and code phase offset:
        return coarse_doppler_est, code_phase_offset_in_samples


    def l1_fine_acquire(self):
        pass
