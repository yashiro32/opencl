import os
import glob
import cv2 as cv
import numpy as np
import pyopencl as cl

def filter_image( ):

    platforms = cl.get_platforms()
    devices = platforms[0].get_devices( cl.device_type.ALL )
    context = cl.Context( [devices[0]] )
    cQ = cl.CommandQueue( context )
    kernel = """
        __kernel void filter( global uchar* a, global uchar* b ){
                int y = get_global_id(0);
                int x = get_global_id(1);

                int sizex = get_global_size(1);

                if( a[ y*sizex + x ] != 255 )
                        b[ y*sizex + x ] = a[ y*sizex + x ];
            }"""

    program = cl.Program( context, kernel ).build()

    for i in glob.glob("*.png"):

        image = cv.imread( i, 0 )        
        b = np.zeros_like( image, dtype = np.uint8 )
        rdBuf = cl.Buffer( 
                context,
                cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                hostbuf = image
                          )

        wrtBuf = cl.Buffer( 
                context,
                cl.mem_flags.WRITE_ONLY,
                b.nbytes
                          )

        program.filter( cQ, image.shape, None, rdBuf, wrtBuf ).wait()
        cl.enqueue_copy( cQ, b, wrtBuf ).wait()
        cv.imshow( 'a', b )
        cv.waitKey( 0 )

def Filter( ):
    os.chdir('D:\image')
    filter_image( )
    cv.destroyAllWindows()
