/*
  A program to generate an image of the Mandelbrot set.

  Usage: ./mandelbrot > output
         where "output" will be a binary image, 1 byte per pixel
         The program will print instructions on stderr as to how to
         process the output to produce a JPG file.

  Michael Ashley / UNSW / 13-Mar-2003
  
  Edited for paralellisation with OpenMP by David van Erkelens and
  Jelte Fennema. 
*/

// Define the range in x and y here:

const double yMin = -1.0;
const double yMax = +1.0;
const double xMin = -2.0;
const double xMax = +0.5;

// And here is the resolution:

const double dxy = 0.0025;

#include <stdio.h>
#include <limits.h>
#include <sys/time.h>
#include "omp.h"

void timer_start(void);
double timer_end(void);

struct timeval timer;

void timer_start(void)
{
    gettimeofday(&timer, NULL);
}

double timer_end(void)
{
    struct timeval t2;
    double result;
    gettimeofday(&t2, NULL);
    result = t2.tv_sec - timer.tv_sec;
    result += ((t2.tv_usec - timer.tv_usec) / 1000000.0);
    return result;
}

int main(void) 
{
    double cx, cy;
    double zx, zy, new_zx;
    unsigned char n;
    int nx, ny, iter, i;
    double time;

    // The Mandelbrot calculation is to iterate the equation
    // z = z*z + c, where z and c are complex numbers, z is initially
    // zero, and c is the coordinate of the point being tested. If
    // the magnitude of z remains less than 2 for ever, then the point
    // c is in the Mandelbrot set. We write out the number of iterations
    // before the magnitude of z exceeds 2, or UCHAR_MAX, whichever is
    // smaller.

    timer_start();
    cy = yMin;
    iter = (int)(yMax - yMin) / dxy;
    #pragma omp parallel for private(cx, zx, zy, n, new_zx) firstprivate(cy)
    for (i = 0; i < iter; i++) 
    {
        for (cx = xMin; cx < xMax; cx += dxy) 
        {
            zx = 0.0;
            zy = 0.0;
            n = 0;
            while ((zx*zx + zy*zy < 4.0) && (n != UCHAR_MAX)) 
            {
                new_zx = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
	        zx = new_zx;
	        n++;
            }
            fprintf(stdout, "%d ", n);
        }
        cy += dxy;
    }

    // Now calculate the image dimensions. We use exactly the same
    // for loops as above, to guard against any potential rounding errors.

    nx = 0;
    ny = 0;
    for (cx = xMin; cx < xMax; cx += dxy) 
    {
        nx++;
    }
    for (cy = yMin; cy < yMax; cy += dxy) 
    {
        ny++;
    }
    time = timer_end();
    fprintf(stderr, "The program took %g seconds.\n", time);
    fprintf (stderr, "To process the image: convert -depth 8 -size %dx%d "
            "gray:output out.jpg\n", nx, ny);
    return 0;
}

