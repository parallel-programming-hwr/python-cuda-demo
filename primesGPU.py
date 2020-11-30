from time import time

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

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

block_size = 1024
grid_size = 50000

check_prime = ker2.get_function("check_prime2")

testvec = np.arange(1, block_size * grid_size * 2, step=2).astype(np.uint)

testvec_gpu = gpuarray.to_gpu(testvec)
outvec_gpu = gpuarray.to_gpu(np.full(block_size * grid_size, False, dtype=bool))
t1 = time()
check_prime(testvec_gpu, outvec_gpu, block=(block_size, 1, 1), grid=(grid_size, 1, 1))
result = outvec_gpu.get()
t2 = time()

primes = []

for idx, val in enumerate(result):
    if val:
        primes.append(idx)

#print(primes)
print(len(primes))
print('checked the first ' + str(block_size * grid_size) + ' numbers')
print('last prime: ' + str(primes[-1]))
print('The GPU needed ' + str(t2 - t1) + ' seconds')
