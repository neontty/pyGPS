from acquisition import *
from pygps.ca_code import *

f_sampling = 5e6
f_intermediate = 1.27e6  #1.25e6

def main():
    input_file = '../../resources/ECE475/trigr_data.dat'
    #input_file='../../resources/ECE475/marcusrm_project5/g072602f.dat'
    #input_file='../../resources/ECE475/marcusrm_project6/simGPSL1_4SVs_10ms.dat'
    with open(input_file, 'rb') as fd:
        file_bytes = fd.read()
        #raw_data = np.frombuffer(file_bytes, dtype=np.int8)
        #data = raw_data.astype(np.float64).view(complex)
        raw_data = np.frombuffer(file_bytes, dtype=np.int16)
        data = raw_data.astype(complex)

        print("raw data shape: ", raw_data.shape)
        print("data shape:", data.shape)
        print("estimated seconds = ", data.shape[0] / f_sampling)

        print("hi")

        ca = CoarseAcquisition(f_sampling, coherent_integration_periods=5, noncoherent_integration_periods=20)

        prns = [1, 2, 5, 10, 12, 23, 25, 29, 31]

        # convert signal to baseband.
        ca_samples = int(ca.samples_per_coherent_period * ca.noncoherent_integration_periods)
        t = np.arange(1, ca_samples + 1) / f_sampling
        baseband_data = data[:ca_samples] * np.exp(-1j*2*np.pi*f_intermediate*t )

        for prn in prns:
            # estimate our coarse frequency error and code phase:
            frequency_error, code_phase = ca.l1_coarse_acquire(baseband_data, prn)




if __name__ == '__main__':
    main()

