import numpy as np
from pycuda import gpuarray
from time import time

# -- initialize the device
import pycuda.autoinit

print((gpuarray.to_gpu(np.array([1], dtype=np.float32)) * 1).get())  # this call speeds up the following gpu
# calculation, because at this point the gpu code gets compiled and cached for the next calls


# compare cpu and gpu
host_data = np.float32(np.random.random(50000000))

t1 = time()
host_data_2x = host_data * np.float32(2)
t2 = time()

print('total time to compute on CPU: %f' % (t2 - t1))

device_data = gpuarray.to_gpu(host_data)

t1 = time()
device_data_2x = device_data * np.float32(2)
t2 = time()

from_device = device_data_2x.get()

print('total time to compute on GPU: %f' % (t2 - t1))
print('Is the host computation the same as the GPU computation? : {}'.format(np.allclose(from_device, host_data_2x)))
