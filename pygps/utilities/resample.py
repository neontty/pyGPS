




from pygps.ca_code import generate_L1_spreading_code


code1 = generate_L1_spreading_code(23)



import numpy as np

somevector = np.tile(np.asarray([0,1]), 100)

print(somevector)

from scipy import signal

test1 = signal.resample_poly(somevector, up=5, down=2)

print(test1)



