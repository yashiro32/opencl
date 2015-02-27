import pyopencl as cl
import numpy as np
from PIL import Image
import sys
import os

os.environ["PYOPENCL_CTX"]="0"

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

kernel_code_histo = """
    __kernel void histogram(__global unsigned int *input, __global unsigned int *histo, unsigned int i,
        __global unsigned int *zero_bits, __global unsigned int *one_bits) {
        int index = get_global_id(0);

        int numBits = 1;
        int numBins = 1 << numBits;

        unsigned int mask = (numBins - 1) << i;

        unsigned int bin = (input[index] & mask) >> i;
    
        atomic_inc(&histo[bin]);

        if (bin == 0) {
            zero_bits[index] = 1;
        } else if (bin == 1) {
            one_bits[index] = 1;
        }
    }
    """

kernel_code_scan = """
    __kernel void blellochScan(__global unsigned int *histo, __global unsigned int *hist, unsigned int len)
    {
        // __local unsigned int XY[10];
        __local unsigned int XY[];
        
        int i = get_global_id(0);
        int index = get_local_id(0);

        if (i < len) {
            XY[index] = histo[i];
        }

        for (unsigned int stride = 1; stride <= index; stride *= 2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            unsigned int in1 = XY[index-stride];
            barrier(CLK_LOCAL_MEM_FENCE);
            XY[index] += in1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < len) {
            hist[i] = XY[index];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (i > 0 && i < len) {
            unsigned int in1 = hist[i-1];
            barrier(CLK_GLOBAL_MEM_FENCE);
            hist[i] = in1;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (i == 0) {
            hist[i] = 0;
        }
    }
    """

kernel_code_hs = """
    __kernel void hs_scan(__global unsigned int *histo, __global unsigned int *hist, unsigned int len) {
        // __local float XY[10];
        __local float XY[];

        int i = get_global_id(0);
        int index = get_local_id(0);

        if (i < len) {
            XY[index] = histo[i];
        }

        for (unsigned int stride = 1; stride <= index; stride *= 2) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            float in1 = XY[index-stride];
            barrier(CLK_GLOBAL_MEM_FENCE);
            XY[index] += in1;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
        if (i < len) {
            hist[i] = XY[index];
        }
    }
    """

kernel_code_move = """
    __kernel void movePos(__global unsigned int *binScan, __global unsigned int *inputVals,
        __global unsigned int *inputPos, __global unsigned int *outputVals,
        __global unsigned int *outputPos, unsigned int i,
        __global unsigned int *zero_bits, __global unsigned int *one_bits) {

        int index = get_global_id(0);

        int numBits = 1;
        int numBins = 1 << numBits;

        unsigned int mask = (numBins - 1) << i;

        unsigned int bin = (inputVals[index] & mask) >> i;

        unsigned int offset = 0;
        if (bin == 0) {
            offset = zero_bits[index];
        } else if (bin == 1) {
            offset = one_bits[index];
        }

        outputVals[binScan[bin]+offset] = inputVals[index];
        outputPos[binScan[bin]+offset]  = inputPos[index];
        
        /* for (int j = 0; j < 10; j++) {
            unsigned int bin = (inputVals[j] & mask) >> i;
            outputVals[binScan[bin]] = inputVals[j];
            outputPos[binScan[bin]]  = inputPos[j];
            binScan[bin] += 1;
        } */
    }
    """

kernel_code_swap = """
    __kernel void swap(__global unsigned int *inputVals, __global unsigned int *inputPos,
        __global unsigned int *outputVals, __global unsigned int *outputPos) {

        int index = get_global_id(0);

        unsigned int temp = outputVals[index];
        outputVals[index] = inputVals[index];
        inputVals[index] = temp;

        temp = outputPos[index];
        outputPos[index] = inputPos[index];
        inputPos[index] = temp;
    }
    """

kernel_code = """
    __kernel void radixSort(__global unsigned int *inputVals, __global unsigned int *inputPos,
        __global unsigned int * outputVals, __global unsigned int *outputPos, int len, unsigned int i,
        __global unsigned int *histo, __global unsigned int *binScan) {

        int index = get_global_id(0);
        
        unsigned int mask = (numBins - 1) << i;

        unsigned int bin = (vals_src[index] & mask) >> i;
        histogram(histo, bin);

        /* // Perform exclusive prefix sum(scan) on binHistogram to get starting
        // location for each bin
        for (unsigned int j = 1; j < numBins; ++j) {
            binScan[j] = binScan[j-1] + binHistogram[j-1];
        }

        // Gather everything into the correct location
        // need to move values and positions
        for (unsigned int j = 0; j < len; ++j) {
            unsigned int bin = (input[j] & mask) >> i;
            vals_dst[binScan[bin]] = vals_src[j];
            pos_dst[binScan[bin]] = pos_src[j];
            binScan[bin]++;
        }

        int temp;
        temp = *vals_dst;
        *vals_dst = *vals_src;
        *vals_src = temp;

        temp = *pos_dst;
        *pos_dst = pos_src;
        *pos_src = temp;

        memcpy(outputVals, inputVals, sizeof(unsigned int) * len);
        memcpy(outputPos, inputPos, sizeof(unsigned int) * len); */
        
    }
    
    """


h_input_vals = np.array([3,9,10,11,22,56,73,24,15,68], dtype=np.uint32)
d_input_vals = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_input_vals)

size = np.int32(h_input_vals.shape[0])

#h_input_pos = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], dtype=np.uint32)
h_input_pos = np.arange(0, size, dtype=np.uint32)
d_input_pos = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_input_pos)

h_output_vals = np.zeros(size, dtype=np.uint32)
d_output_vals = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_output_vals)

h_output_pos = np.zeros(size, dtype=np.uint32)
d_output_pos = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_output_pos)

numBits = 1
numBins = 1 << numBits

for i in range(8*np.dtype('uint32').itemsize):
    h_histo = np.zeros(2, dtype=np.uint32)
    d_histo = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_histo)

    h_and_zero_bits = np.zeros(size, dtype=np.uint32)
    d_and_zero_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_and_zero_bits)

    h_and_one_bits = np.zeros(size, dtype=np.uint32)
    d_and_one_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_and_one_bits)

    
    mask = (numBins - 1) << i;
    
    """#Perform histogram of data & mask into bins
    h_hist = np.zeros(2, dtype=np.uint32)
    for j in range(size):
        bin = (h_input_vals[j] & mask) >> i;
        h_hist[bin] += 1;
    
    #Perform exclusive prefix sum(scan) on binHistogram to get starting
    #location for each bin
    h_scan = np.zeros(2, dtype=np.uint8)
    for j in range(1,numBins):
         h_scan[j] = h_scan[j-1] + h_hist[j-1]

    #Gather everything into the correct location
    #need to move vals and positions
    for j in range(size):
        bin = (h_input_vals[j] & mask) >> i
        h_output_vals[h_scan[bin]] = h_input_vals[j]
        h_output_pos[h_scan[bin]]  = h_input_pos[j]
        h_scan[bin] += 1 
    
    #swap the buffers (pointers only)
    for j in range(size):
        temp = h_output_vals[j]
        h_output_vals[j] = h_input_vals[j]
        h_input_vals[j] = temp

        temp = h_output_pos[j]
        h_output_pos[j] = h_input_pos[j]
        h_input_pos[j] = temp """

    
    prg = cl.Program(ctx, kernel_code_histo).build()
    prg.histogram(queue, (size,1), None, d_input_vals, d_histo, np.uint32(i), d_and_zero_bits, d_and_one_bits)

    zero_bits = np.empty_like(h_and_zero_bits)
    cl.enqueue_copy(queue, zero_bits, d_and_zero_bits)
    one_bits = np.empty_like(h_and_one_bits)
    cl.enqueue_copy(queue, one_bits, d_and_one_bits)

    print zero_bits
    print one_bits


    h_scan_zero_bits = np.zeros(size, dtype=np.uint32)
   
    h_scan_one_bits = np.zeros(size, dtype=np.uint32)

    d_scan_zero_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan_zero_bits)
    d_scan_one_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan_one_bits)

    d_temp = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan_zero_bits)
    
    prg = cl.Program(ctx, kernel_code_scan).build()
    prg.blellochScan(queue, (size,1), None, d_and_zero_bits, d_scan_zero_bits, size)

    prg = cl.Program(ctx, kernel_code_scan).build()
    prg.blellochScan(queue, (size,1), None, d_and_one_bits, d_scan_one_bits, size)

    #prg = cl.Program(ctx, kernel_code_hs).build()
    #prg.hs_scan(queue, (size,1), None, d_and_zero_bits, d_scan_zero_bits, size)

    #prg = cl.Program(ctx, kernel_code_hs).build()
    #prg.hs_scan(queue, (size,1), None, d_and_one_bits, d_scan_one_bits, size)

    scan_zero_bits = np.empty_like(h_scan_zero_bits)
    cl.enqueue_copy(queue, scan_zero_bits, d_scan_zero_bits)
    scan_one_bits = np.empty_like(h_scan_one_bits)
    cl.enqueue_copy(queue, scan_one_bits, d_scan_one_bits)

    """for j in range(1,size):
        h_scan_zero_bits[j] = h_scan_zero_bits[j-1] + zero_bits[j-1]
        h_scan_one_bits[j] = h_scan_one_bits[j-1] + one_bits[j-1]

    d_scan_zero_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan_zero_bits)
    d_scan_one_bits = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan_one_bits)"""

    print scan_zero_bits
    print scan_one_bits

    
    h_bin_scan = np.zeros(2, dtype=np.uint32)
    

    #prg = cl.Program(ctx, kernel_code_scan).build()
    #prg.blellochScan(queue, (numBins/2,1), None, d_histo, d_bin_scan, np.uint32(1))

    h_hist = np.empty_like(h_histo)
    cl.enqueue_copy(queue, h_hist, d_histo)
    h_scan = np.zeros(2, dtype=np.uint32)
    for j in range(1,numBins):
         h_scan[j] = h_scan[j-1] + h_hist[j-1]

    d_bin_scan = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_scan)
    
    
    prg = cl.Program(ctx, kernel_code_move).build()
    prg.movePos(queue, (size,1), None, d_bin_scan, d_input_vals, d_input_pos, d_output_vals, d_output_pos, np.uint32(i), d_scan_zero_bits, d_scan_one_bits)

    """for j in range(size):
        bin = (h_input_vals[j] & mask) >> i
        h_output_vals[h_scan[bin]] = h_input_vals[j]
        h_output_pos[h_scan[bin]]  = h_input_pos[j]
        h_scan[bin] += 1

    d_input_vals = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_input_vals)
    d_input_pos = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_input_pos)
    d_output_vals = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_output_vals)
    d_output_pos = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_output_pos)

    for j in range(size):
        temp = h_output_vals[j]
        h_output_vals[j] = h_input_vals[j]
        h_input_vals[j] = temp

        temp = h_output_pos[j]
        h_output_pos[j] = h_input_pos[j]
        h_input_pos[j] = temp"""
     

    prg = cl.Program(ctx, kernel_code_swap).build()
    prg.swap(queue, (size,1), None, d_input_vals, d_input_pos, d_output_vals, d_output_pos)
    
    h_input_vals = np.empty_like(h_input_vals)
    cl.enqueue_copy(queue, h_input_vals, d_input_vals)

    h_input_pos = np.empty_like(h_input_pos)
    cl.enqueue_copy(queue, h_input_pos, d_input_pos)

    h_output_vals = np.empty_like(h_output_vals)
    cl.enqueue_copy(queue, h_output_vals, d_output_vals)

    h_output_pos = np.empty_like(h_output_pos)
    cl.enqueue_copy(queue, h_output_pos, d_output_pos)
    
"""res = np.empty_like(h_histo)
cl.enqueue_copy(queue, res, d_histo)
print res"""

print h_input_vals
