import numpy as np
from pycuda import gpuarray

# -- initialize the device
import pycuda.autoinit

dev = pycuda.autoinit.device

print(dev.name())
print('\t Total Memory: {} megabytes'.format(dev.total_memory() // (1024 ** 2)))

device_attributes = {}
for k, v in dev.get_attributes().items():
    device_attributes[str(k)] = v
    print('\t ' + str(k) + ': ' + str(v))

host_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
host_data_2 = np.array([7, 12, 3, 5, 4], dtype=np.float32)

device_data = gpuarray.to_gpu(host_data)
device_data_2 = gpuarray.to_gpu(host_data_2)

print(host_data * host_data_2)
print((device_data * device_data_2).get())

print(host_data / 2)
print((device_data / 2).get())

print(host_data - host_data_2)
print((device_data - device_data_2).get())
