import datetime
import math
import time

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
    return (block_size * grid_size) / kernel_execution_time


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = OptionParser()
    parser.add_option("--bench-output", dest="bench_output",
                      help="The output file for timings", default="timings.csv", type="string")
    parser.add_option("--calculation-steps", dest="calculation_steps",
                      help=" How many calculations steps should be done per GPU thread", default="1000000", type="int")

    parser.add_option("--grid_size", dest="grid_size",
                      help="number of blocks", default="8192", type="int")
    parser.add_option("--number_of_threads", dest="number_of_threads",
                      help="If set the grid size is ignored and is adjusted so that the number of threads is the same "
                           "in all cases", type="int", default="1048576")
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
            options.grid_size, options.block_size_start, options.block_size_step, options.block_size_stop,
            options.calculation_steps, options.repetitions))

    filename = str(time.strftime("%Y%m%d-%H%M%S")) + "_block_size_" + str(options.number_of_threads) + "threads_" + str(
        options.calculation_steps) + "claculations_" + str(options.repetitions) + "repetitions_" + options.bench_output
    with open(filename, 'w') as file:
        file.write("block size,calls per second\n")

    current_block_size = options.block_size_start

    while current_block_size <= options.block_size_stop:
        total_duration = 0
        if options.number_of_threads is not None:
            if options.number_of_threads % current_block_size != 0:
                print(str(options.number_of_threads) + " is not dividable by block size of " + str(
                    current_block_size) + " thus will be skipped")
                current_block_size += options.block_size_step
                continue
            else:
                current_grid_size = int(options.number_of_threads / current_block_size)
        else:
            current_grid_size = options.grid_size
        for i in range(0, options.repetitions):
            duration = bench_block_size(current_grid_size, current_block_size, options.calculation_steps)
            total_duration += duration
        with open(filename, 'a') as file:
            file.write(str(current_block_size) + "," + str(total_duration / options.repetitions) + "\n")
        current_block_size += options.block_size_step

    print("finished in " + str((datetime.datetime.now()-start_time)))
