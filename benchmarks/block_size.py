import math
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda import gpuarray
from pycuda.compiler import SourceModule

from optparse import OptionParser

ker = SourceModule("""
__global__ void bench_int(const int limit, int *NUMBERS) {
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    int num = NUMBERS[id];
    for (int i = 0; i < limit; i++) {
        num += i;
    }
    NUMBERS[id] = num;
}
""")


def bench_block_size(grid_size: int, block_size: int, calc_count: int):
    fetch_add = ker.get_function("bench_int")

    vector_gpu = gpuarray.to_gpu(np.ones(block_size * grid_size).astype(np.intc))

    startEvent = drv.Event()
    endEvent = drv.Event()
    startEvent.record()

    fetch_add(np.int_(calc_count), vector_gpu, block=(block_size, 1, 1), grid=(grid_size, 1, 1))

    endEvent.record()
    endEvent.synchronize()

    kernel_execution_time = startEvent.time_till(endEvent)
    return (block_size * grid_size)/kernel_execution_time


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--bench-output", dest="bench_output",
                      help="The output file for timings", default="timings.csv", type="string")
    parser.add_option("--calculation-steps", dest="calculation_steps",
                      help=" How many calculations steps should be done per GPU thread", default="1000000", type="int")

    parser.add_option("--grid_size", dest="grid_size",
                      help="number of blocks", default="1024", type="int")
    parser.add_option("--block_size_start", dest="block_size_start",
                      help="initial number of threads per block", default="4",
                      type="int")
    parser.add_option("--block_size_step", dest="block_size_step",
                      help="The amount the block size increases by every step", default="4",
                      type="int")
    parser.add_option("--block_size_stop", dest="block_size_stop",
                      help="maximum number of threads per block, max = 1024", default="1024",
                      type="int")
    parser.add_option("--repetitions", dest="repetitions",
                      help=" The average of n runs that is used instead of using one value only.", default="1",
                      type="int")
    (options, args) = parser.parse_args()
    print(
        "Benchmarking block size. Grid Size: {}, Start: {}, Step: {} ,Stop: {}, Calculations: {}, Repetitions: {}".format(
            options.grid_size, options.block_size_start, options.block_size_step, options.block_size_stop, options.calculation_steps, options.repetitions))

    with open(options.bench_output, 'w') as file:
        file.write("block size,calls per second\n")

    current_block_size = options.block_size_start

    while current_block_size <= options.block_size_stop:
        total_duration = 0
        for i in range(0, options.repetitions):
            duration = bench_block_size(options.grid_size, current_block_size, options.calculation_steps)
            total_duration += duration
        with open(options.bench_output, 'a') as file:
            file.write(str(current_block_size) + "," + str(total_duration/options.repetitions) + "\n")
        current_block_size += options.block_size_step
