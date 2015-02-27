import pyopencl as cl
import numpy
from PIL import Image
import sys
import os

os.environ["PYOPENCL_CTX"]="0"

img = Image.open("C:\python_scripts\download.png")
img_arr = numpy.asarray(img).astype(numpy.uint8)
dim = img_arr.shape

host_arr = img_arr.reshape(-1)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=host_arr)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, host_arr.nbytes)

kernel_code = """
    __kernel void copyImage(__global const unsigned char *a, __global unsigned char *c)
    {
        int rowid = get_global_id(0);
        int colid = get_global_id(1);

        int ncols = %d;
        int npix = %d; //number of pixels, 3 for RGB 4 for RGBA

        // int index = rowid * ncols * npix + colid * npix;
        int index  = (rowid * ncols + colid) * npix;
        
        unsigned char I = (.299f * a[index + 0]) + (.587f * a[index + 1]) + (.114f * a[index + 2]);
        // unsigned char I = (a[index + 0] + a[index + 1] + a[index + 2]) / 3;

        c[index + 0] = I;
        c[index + 1] = I;
        c[index + 2] = I;
        
        /* unsigned char I = ((unsigned char)(a[index + 0] * .299f) << 16) | ((unsigned char)(a[index + 1] * .587f) << 8) | (unsigned char)(a[index + 2] * .114f);
        
        c[index + 0] = ((0xff0000 & I) >> 16);
        c[index + 1] = ((0x00ff00 & I) >> 8);
        c[index + 2] = (0x0000ff & I); */
        
        /* c[index + 0] = a[index + 0] * .299f;
        c[index + 1] = a[index + 1] * .587f;
        c[index + 2] = a[index + 2] * .114f; */
    }
    """ % (dim[1], dim[2])

prg = cl.Program(ctx, kernel_code).build()

prg.copyImage(queue, (dim[0], dim[1]) , None, a_buf, dest_buf)

result = numpy.empty_like(host_arr)
cl.enqueue_copy(queue, result, dest_buf)

result_reshaped = result.reshape(dim)
img2 = Image.fromarray(result_reshaped, "RGB")
img2.save("new_image_gpu.bmp")
