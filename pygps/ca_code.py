
import numpy as np
import json
from constants import *

# G1 and G2 tap assignments based on PRN (index 1-32)
L1_CODE_PHASE_ASSIGNMENTS = np.asarray(
    [[2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 8], [2, 9], [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8],
     [8, 9], [9, 10], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6], [5, 7], [6, 8], [7, 9], [8, 10],
     [1, 6], [2, 7], [3, 8], [4, 9], [5, 10], [4, 10], [1, 7], [2, 8], [4, 10]])


def generate_L1_spreading_code(prn):
    """
    Generate one period of L1 CA Code for a specific prn
    :param prn: 0-32
    :return: binary array of CA Code
    """
    output = np.zeros((L1_N_CHIPS,), dtype=int)
    g1_registers = np.ones((10,), dtype=bool)
    g2_registers = np.ones((10,), dtype=bool)

    for idx in range(L1_N_CHIPS):
        # get output for this stage:
        g1_output = g1_registers[9]
        g2_tap_idx_1 = L1_CODE_PHASE_ASSIGNMENTS[prn - 1, 0]
        g2_tap_idx_2 = L1_CODE_PHASE_ASSIGNMENTS[prn - 1, 1]
        g2_output = g2_registers[g2_tap_idx_1 - 1] ^ g2_registers[g2_tap_idx_2 - 1]
        output[idx] = g1_output ^ g2_output

        # update and shift G1 registers
        g1_push_front = g1_registers[2] ^ g1_registers[9]
        g1_registers = np.roll(g1_registers, 1)
        g1_registers[0] = g1_push_front

        # update and shift G2 registers
        g2_push_front = g2_registers[1] ^ g2_registers[2] ^ g2_registers[5] ^ g2_registers[7] ^ g2_registers[8] ^ \
                        g2_registers[9]
        g2_registers = np.roll(g2_registers, 1)
        g2_registers[0] = g2_push_front

    return output


L1_CA_CODES = {}


def generate_all_l1_ca_codes():
    for prn in L1_PRNS:
        L1_CA_CODES[prn] = generate_L1_spreading_code(prn)


generate_all_l1_ca_codes()
# this file should act like a singleton so all other modules can access its data.


# TODO put reference and assert in pytest
reference_first_10_chips_octal = {1: 0o1440, 2: 0o1620, 3: 0o1710, 4: 0o1744, 5: 0o1133, 6: 0o1455, 7: 0o1131,
                                  8: 0o1454, 9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750, 13: 0o1764, 14: 0o1772,
                                  15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467, 19: 0o1633, 20: 0o1715, 21: 0o1746,
                                  22: 0o1763, 23: 0o1063, 24: 0o1706, 25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774,
                                  29: 0o1127, 30: 0o1453, 31: 0o1625, 32: 0o1712}


def verify_l1_ca_codes():
    for prn in L1_PRNS:
        reference_bits = [int(x) for x in bin(reference_first_10_chips_octal[prn])[2:]]
        if not np.all(reference_bits == L1_CA_CODES[prn][:10]):
            return False
    return True


def resample_l1_ca_code_with_doppler(prn, f_sampling, f_doppler, n_samples):
    # calculate new chipping rates based on each doppler rates:
    f_code_doppler = L1_CHIPPING_RATE * (1 + (f_doppler / L1_FREQ))

    ca_code = L1_CA_CODES[prn]

    chips_per_sample = f_code_doppler / f_sampling

    resampling_indices = (np.arange(n_samples) * chips_per_sample % len(ca_code)).astype(np.int)

    return ca_code[resampling_indices]



if __name__ == '__main__':
    verify_l1_ca_codes()
    print(L1_CA_CODES)

    with open('utilities/ca_codes.json', 'w') as fd:
        fd.write(json.dumps(L1_CA_CODES))

