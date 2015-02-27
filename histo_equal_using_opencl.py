import pyopencl as cl
import numpy
from PIL import Image
import sys
import os

os.environ["PYOPENCL_CTX"] = "0"


def histogramEqualizationLUT(rhisto, ghisto, bhisto, nrows, ncols):
    rhist = numpy.zeros(256, dtype=numpy.int32)
    ghist = numpy.zeros(256, dtype=numpy.int32)
    bhist = numpy.zeros(256, dtype=numpy.int32)

    sumr = 0.0
    sumg = 0.0
    sumb = 0.0

    scale_factor = float(255.0 / (nrows * ncols))

    for i in range(256):
        sumr += rhisto[i]
        valr = int(sumr * scale_factor)
        if valr > 255:
            rhist[i] = 255
        elif valr < 0:
            rhist[i] = 0
        else:
            rhist[i] = valr

        sumg += ghisto[i]
        valg = int(sumg * scale_factor)
        if valg > 255:
            ghist[i] = 255
        elif valg < 0:
            ghist[i] = 0
        else:
            ghist[i] = valg

        sumb += bhisto[i]
        valb = int(sumb * scale_factor)
        if valb > 255:
            bhist[i] = 255
        elif valb < 0:
            bhist[i] = 0
        else:
            bhist[i] = valb

    return (rhist, ghist, bhist)


img = Image.open("C:\python_scripts\input-300x200.jpg")
img_arr = numpy.asarray(img).astype(numpy.uint8)
dim = img_arr.shape

host_arr = img_arr.reshape(-1)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

histo = numpy.zeros(256, dtype=numpy.int32)

d_rhisto = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=histo)
d_ghisto = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=histo)
d_bhisto = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=histo)

a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_arr)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, host_arr.nbytes)

kernel_code = """
    __kernel void imageHistogram(__global const unsigned char *a, __global int *rhisto, __global int *ghisto, __global int *bhisto)
    {
        int rowid = get_global_id(0);
        int colid = get_global_id(1);

        int ncols = %d;
        int npix = %d; // Number of pixels, 3 for RGB 4 for RGBA

        // int index = rowid * ncols * npix + colid * npix;
        int index  = (rowid * ncols + colid) * npix;

        int red = a[index + 0];
        int green = a[index + 1];
        int blue = a[index + 2];

        // Increase the values of colors
        atomic_inc(&rhisto[red]);
        atomic_inc(&ghisto[green]);
        atomic_inc(&bhisto[blue]);
    }
    """ % (dim[1], dim[2])

kernel_code2 = """
    __kernel void blellochScan(__global int *histo, __global int *hist, int len)
    {
        // int len = 256;

        int nrows = %d;
        int ncols = %d;
        
        int tid = get_global_id(0);

        // long sum = 0;

        __local int btemp[];
	int offset = 1;
        
	if (tid < len)
	{
		btemp[2*tid] = histo[2*tid];
		btemp[2*tid+1] = histo[2*tid+1];
	}
 
	for (int d = len >> 1; d > 0; d >>= 1)
	{
		// __syncthreads();
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			btemp[bi] += btemp[ai];
		}
		offset *= 2;
	}
 
	// __syncthreads();
	barrier(CLK_GLOBAL_MEM_FENCE);
 
	if (tid == 0)
	{
		btemp[len-1] = 0;
	}
 
	for (int d = 1; d < len; d *= 2)
	{
		offset >>= 1;
		// __syncthreads();
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			int t = btemp[ai];
			btemp[ai] = btemp[bi];
			btemp[bi] += t;
		}
	}
 
	// __syncthreads();
	barrier(CLK_GLOBAL_MEM_FENCE);

	
        float scale_factor = (float)(255.f / (nrows * ncols));
	int val = (int)(btemp[2*tid] * scale_factor);
        int val2 = (int)(btemp[2*tid+1] * scale_factor);
	
        if (val > 255)
            val = 255;
    
        if (val2 > 255)
            val2 = 255;
        
	if (tid < len)
	{
            // hist[2*tid] = btemp[2*tid];
	    // hist[2*tid+1] = btemp[2*tid+1];    
	    hist[2*tid] = val;
	    hist[2*tid+1] = val2;
	}
        
    }
    """ % (dim[0], dim[1])

kernel_code3 = """
    __kernel void histoEqual(__global const unsigned char *a, __global unsigned char *c, __global int *rhisto, __global int *ghisto, __global int *bhisto)
    {
        int rowid = get_global_id(0);
        int colid = get_global_id(1);

        int ncols = %d;
        int npix = %d; // Number of pixels, 3 for RGB 4 for RGBA

        // int index = rowid * ncols * npix + colid * npix;
        int index  = (rowid * ncols + colid) * npix;
        
        int red  = a[index + 0];
        int green = a[index + 1];
        int blue = a[index + 2];

        red = rhisto[red];
        green = ghisto[green];
        blue = bhisto[blue];

        c[index + 0] = red;
        c[index + 1] = green;
        c[index + 2] = blue;
    }
    """ % (dim[1], dim[2])

kernel_code2_v2 = """
    __kernel void histoLUT(__global int *histo, __global int *hist) {
        int index = get_global_id(0);

        int nrows = %d;
        int ncols = %d;

        long sum = 0;

        float scale_factor = (float)(255.f / (nrows * ncols));

        for (int i = 0; i <= index; i++) {
            sum += histo[i];
        }

        int val = (int)(sum * scale_factor);
        if (val > 255)
            hist[index] = 255;
        else if (val < 0)
            hist[index] = 0;
        else
            hist[index] = val;
    }
    """ % (dim[0], dim[1])

kernel_code2_v3 = """
    __kernel void hs_scan(__global int *histo, __global int *hist, int len, __global int *btemp, __global int * atemp) {
        int nrows = %d;
        int ncols = %d;
    
        int index = get_global_id(0);

        // __local int btemp[256];
        // __local int atemp[256];

        atemp[index] = btemp[index] = histo[index];
        barrier(CLK_GLOBAL_MEM_FENCE);

        // int offset = 1;

        for( int offset = 1; offset < len; offset <<= 1 ) {
            if ((index - offset) >= 0) {
                btemp[index] += btemp[index - offset];
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        
        float scale_factor = (float)(255.f / (nrows * ncols));
	int val = (int)(btemp[index] * scale_factor);
        
        if (val > 255)
            val = 255;
        
        if (index < len) {
            // hist[index] = btemp[index];
            hist[index] = val;
        }
    }
    """ % (dim[0], dim[1])

prg = cl.Program(ctx, kernel_code).build()

prg.imageHistogram(queue, (dim[0], dim[1]), None, a_buf, d_rhisto, d_ghisto, d_bhisto)

h_rhisto = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_rhisto, d_rhisto)

"""h_ghisto = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_ghisto, d_ghisto)

h_bhisto = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_bhisto, d_bhisto)

(h_rhist, h_ghist, h_bhist) = histogramEqualizationLUT(h_rhisto, h_ghisto, h_bhisto, dim[0], dim[1])

d_rhist = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_rhist)
d_ghist = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_ghist)
d_bhist = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_bhist)"""


"""# Test Hillis/Steele Scan kernal function and compare with Blelloch Scan kernel function
size = numpy.int32(10)

arr = numpy.array([0,0,1,0,1,1,0,0,0,1], dtype=numpy.int32)
d_arr = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arr)

h_res = numpy.zeros(size, dtype=numpy.int32)
d_res = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_res)

d_temp = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_res)
d_temp2 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_res)

prgTest = cl.Program(ctx, kernel_code2).build()
#prgTest.hs_scan(queue, (size,1), None, d_arr, d_res, size, d_temp, d_temp2)
prgTest.blellochScan(queue, (size,1), None, d_arr, d_res, size)

res = numpy.empty_like(h_res)
cl.enqueue_copy(queue, res, d_res)
print res"""


d_rhist = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=histo)
d_ghist = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=histo)
d_bhist = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=histo)

d_temp = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=histo)
d_temp2 = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=histo)

#nn_buf = cl.LocalMemory(512 * numpy.dtype('int32').itemsize)

length = numpy.int32(histo.shape[0])
prg2 = cl.Program(ctx, kernel_code2_v3).build()
#prg2.blellochScan(queue, (256/2,1), None, d_rhisto, d_rhist, length)
#prg2.blellochScan(queue, (256/2,1), None, d_ghisto, d_ghist, length)
#prg2.blellochScan(queue, (256/2,1), None, d_bhisto, d_bhist, length)
prg2.hs_scan(queue, (256,1), None, d_rhisto, d_rhist, length, d_temp, d_temp2)
prg2.hs_scan(queue, (256,1), None, d_ghisto, d_ghist, length, d_temp, d_temp2)
prg2.hs_scan(queue, (256,1), None, d_bhisto, d_bhist, length, d_temp, d_temp2)


h_rhist = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_rhist, d_rhist)

h_ghist = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_ghist, d_ghist)

h_bhist = numpy.zeros(256, dtype=numpy.int32)
cl.enqueue_copy(queue, h_bhist, d_bhist)

#print h_rhist


prg3 = cl.Program(ctx, kernel_code3).build()
prg3.histoEqual(queue, (dim[0], dim[1]), None, a_buf, dest_buf, d_rhist, d_ghist, d_bhist)

result = numpy.empty_like(host_arr)
cl.enqueue_copy(queue, result, dest_buf)

result_reshaped = result.reshape(dim)
img2 = Image.fromarray(result_reshaped, "RGB")
img2.save("histo_equal.bmp")


    


