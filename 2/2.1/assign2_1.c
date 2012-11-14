/*
 * assign1_1.c
 *
 * Contains code for setting up and finishing the simulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "file.h"
#include "timer.h"
#include "simulate.h"
#include "mpi.h"

typedef double (*func_t)(double x);

/*
 * Simple gauss with mu=0, sigma^1=1
 */
double gauss(double x)
{
    return exp((-1 * x * x) / 2);
}


/*
 * Fills a given array with samples of a given function. This is used to fill
 * the initial arrays with some starting data, to run the simulation on.
 *
 * The first sample is placed at array index `offset'. `range' samples are
 * taken, so your array should be able to store at least offset+range doubles.
 * The function `f' is sampled `range' times between `sample_start' and
 * `sample_end'.
 */
void fill(double *array, int offset, int range, double sample_start,
        double sample_end, func_t f)
{
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = f(sample_start + i * dx);
    }
}


int main(int argc, char *argv[])
{
    double *old, *cur, *new, *ret, *old_array, *current_array, *next_array;
    int t_max, i_max, num_tasks, my_rank, rc, local_size;
    double time;
    MPI_Request request;
    MPI_Status status;

    /* Parse commandline args */
    if (argc < 3) {
        printf("Usage: %s i_max t_max num_threads [initial_data]\n", argv[0]);
        printf(" - i_max: number of discrete amplitude points, should be >2\n");
        printf(" - t_max: number of discrete timesteps, should be >=1\n");
        printf(" - num_threads: number of threads to use for simulation, "
                "should be >=1\n");
        printf(" - initial_data: select what data should be used for the first "
                "two generation.\n");
        printf("   Available options are:\n");
        printf("    * sin: one period of the sinus function at the start.\n");
        printf("    * sinfull: entire data is filled with the sinus.\n");
        printf("    * gauss: a single gauss-function at the start.\n");
        printf("    * file <2 filenames>: allows you to specify a file with on "
                "each line a float for both generations.\n");

        return EXIT_FAILURE;
    }

    if ((rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS) {
        fprintf(stderr, "Unable to setup MPI");

        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    i_max = atoi(argv[1]);
    t_max = atoi(argv[2]);

    if (i_max < 3) {
        printf("argument error: i_max should be >2.\n");
        return EXIT_FAILURE;
    }
    if (t_max < 1) {
        printf("argument error: t_max should be >=1.\n");
        return EXIT_FAILURE;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks); // Determine number of tasks
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Determine task id

    local_size = i_max / num_tasks;

    old = calloc(local_size + 2, sizeof(double));
    cur = calloc(local_size + 2, sizeof(double));
    new = calloc(local_size + 2, sizeof(double));

    // Master sends data and workers receive it
    if (!my_rank) {
        /* Allocate and initialize buffers. */
        old_array = calloc(i_max, sizeof(double));
        current_array = calloc(i_max, sizeof(double));
        next_array = calloc(i_max, sizeof(double));

        if (old_array == NULL || current_array == NULL || next_array == NULL) {
            fprintf(stderr, "Could not allocate enough memory, aborting.\n");
            return EXIT_FAILURE;
        }

        /* How should we will our first two generations? This is determined by the
         * optional further commandline arguments.
         * */
        if (argc > 3) {
            if (strcmp(argv[3], "sin") == 0) {
                fill(old_array, 1, i_max/4, 0, 2*3.14, sin);
                fill(current_array, 2, i_max/4, 0, 2*3.14, sin);
            } else if (strcmp(argv[3], "sinfull") == 0) {
                fill(old_array, 1, i_max-2, 0, 10*3.14, sin);
                fill(current_array, 2, i_max-3, 0, 10*3.14, sin);
            } else if (strcmp(argv[3], "gauss") == 0) {
                fill(old_array, 1, i_max/4, -3, 3, gauss);
                fill(current_array, 2, i_max/4, -3, 3, gauss);
            } else if (strcmp(argv[3], "file") == 0) {
                if (argc < 6) {
                    printf("No files specified!\n");
                    return EXIT_FAILURE;
                }
                file_read_double_array(argv[4], old_array, i_max);
                file_read_double_array(argv[5], current_array, i_max);
            } else {
                printf("Unknown initial mode: %s.\n", argv[3]);
                return EXIT_FAILURE;
            }
        } else {
            /* Default to sinus. */
            fill(old_array, 1, i_max/4, 0, 2*3.14, sin);
            fill(current_array, 2, i_max/4, 0, 2*3.14, sin);
        }

        memcpy(cur + 1, current_array, sizeof(double) * local_size);
        memcpy(old + 1, old_array, sizeof(double) * local_size);
        for (int i = 1; i < num_tasks; i++) {
            MPI_Isend(old_array + (local_size * i), local_size, MPI_DOUBLE, i,
                    0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
            MPI_Isend(current_array + (local_size * i), local_size, MPI_DOUBLE,
                    i, 1, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
    }
    else {
        MPI_Recv(old + 1, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                &status);
        MPI_Recv(cur + 1, local_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
                &status);
    }

    if (my_rank == num_tasks - 1) {
        new[local_size + 1] = 0;
    }


    timer_start();

    /* Call the actual simulation that should be implemented in simulate.c. */
    ret = simulate(local_size, t_max, num_tasks, my_rank, old, cur, new);

    if (my_rank) {
        MPI_Send(ret + 1, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }
    else {
        memcpy(current_array, ret + 1, sizeof(double) * local_size);
        for (int i = 1; i < num_tasks; i++) {
            MPI_Recv(current_array + (i * local_size), local_size, MPI_DOUBLE, i, 0,
                    MPI_COMM_WORLD, &status);
        }
    }

    MPI_Finalize(); // Shutdown MPI runtime

    time = timer_end();

    printf("Took %g seconds\n", time);
    printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));

    file_write_double_array("result.txt", current_array, i_max);

    if (!my_rank) {
        free(old_array);
        free(current_array);
        free(next_array);
    }
    free(old);
    free(cur);
    free(new);

    return EXIT_SUCCESS;
}
