import math
from time import time

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

from optparse import OptionParser

ker = SourceModule("""
__global__ void
check_prime(unsigned long long *input, bool *output)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    unsigned long long num = input[i];
    if (num == 2) {
        output[i] = true;
        return;
    } else if (num < 3 || num % 2 == 0) {
        return;
    } 
    unsigned long long limit = (long) sqrt((double) num) + 1;
    for (unsigned long long i = 3; i <= limit; i += 2) {
        if (num % i == 0) {
            return;
        }
    }
    output[i] = true;
}
""")

ker2 = SourceModule("""
__global__ void check_prime2(const unsigned long *IN, bool *OUT) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long num = IN[id];
    unsigned long limit = (unsigned long) sqrt((double) num) + 1;

    if (num == 2 || num == 3) {
        OUT[id] = true;
        return;
    } else if (num == 1 || num % 2 == 0) {
        return;
    }
    if (limit < 9) {
        for (unsigned long i = 3; i <= limit; i++) {
            if (num % i == 0) {
                return;
            }
        }
    } else {
        if (num > 3 && num % 3 == 0) {
            return;
        }
        for (unsigned long i = 9; i <= (limit + 6); i += 6) {
            if (num % (i - 2) == 0 || num % (i - 4) == 0) {
                return;
            }
        }
    }

    OUT[id] = true;
}
""")


def calc_primes(start: int = 1, grid_size: int = 1000, block_size: int = 1024):
    check_prime = ker2.get_function("check_prime2")

    primes = []
    if start < 2:
        primes = [2]
        start = 3
    if start % 2 == 0:
        start = start + 1

    testvec = np.arange(start, block_size * grid_size * 2 + start, step=2).astype(np.uint)

    testvec_gpu = gpuarray.to_gpu(testvec)
    outvec_gpu = gpuarray.to_gpu(np.full(block_size * grid_size, False, dtype=bool))
    t1 = time()
    check_prime(testvec_gpu, outvec_gpu, block=(block_size, 1, 1), grid=(grid_size, 1, 1))
    result = outvec_gpu.get()
    t2 = time()

    for idx, val in enumerate(result):
        if val:
            primes.append(testvec[idx])

    print(len(primes))
    print('checked ' + str(block_size * grid_size) + ' numbers')
    print('last prime: ' + str(primes[-1]))
    print('The GPU needed ' + str(t2 - t1) + ' seconds')

    with open(options.timings_output, 'a') as file:
        file.write(str(start) + "," + str((t2 - t1) * 1000) + "\n")

    return primes


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-e", "--end", dest="end",
                      help="numbers to check without even numbers", default="1000000000", type="int")
    parser.add_option("--numbers-per-step", dest="numbers_per_step",
                      help="amount of uneven numbers checked in each step (even number are skipped)", default="8388608",
                      type="int")
    parser.add_option("--output", dest="output",
                      help="name of the file, where the primes should be stored", default="primes.txt", type="string")
    parser.add_option("--timings-output", dest="timings_output",
                      help="name of the csv file, where the timing is logged as csv", default="timings.csv",
                      type="string")
    (options, args) = parser.parse_args()

    block_size = 1024
    start = 1
    grid_size = int(math.ceil(options.numbers_per_step / block_size))  # half of numbers per
    resulting_numbers_per_step = block_size * grid_size
    last_number_checked = start - 1

    with open(options.timings_output, 'w') as file:
        file.write("offset, duration\n")

    while last_number_checked < options.end:
        calculated_primes = calc_primes(last_number_checked + 1, grid_size, block_size)
        with open(options.output, 'w') as file:
            file.write("\n".join([str(p) for p in calculated_primes]))
        last_number_checked = last_number_checked + resulting_numbers_per_step * 2