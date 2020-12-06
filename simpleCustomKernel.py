import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

ker = SourceModule("""
__global__ void scalar_multiply_kernel(float *outvec, float scalar, float *vec)
{
     int i = threadIdx.x;
     outvec[i] = scalar*vec[i];
}
""")  # compile kernel function

scalar_multiply_gpu = ker.get_function("scalar_multiply_kernel")  # get kernel function reference

host_vector = np.random.randn(512).astype(np.float32)  # create array of 512 random numbers
device_vector = gpuarray.to_gpu(host_vector)  # copy into GPUs global memory
out_device_vector = gpuarray.empty_like(device_vector)  # allocate a chunk of empty memory to GPUs global memory

scalar_multiply_gpu(out_device_vector, np.float32(2), device_vector, block=(512, 1, 1), grid=(1, 1, 1))  # launch the kernel
print("Does our kernel work correctly? : {}".format(np.allclose(out_device_vector.get(), 2 * host_vector)))
print(out_device_vector.get())
