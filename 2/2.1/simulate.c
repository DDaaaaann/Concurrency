/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"
#include "mpi.h"


/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int local_size, const int t_max, int num_tasks,
        int my_rank, double *old, double *cur, double *new)
{
    int left = my_rank - 1, right = my_rank + 1;
    MPI_Request request;
    MPI_Status status;


    // Calculate stuff, send halo cells and switch over arrays
    for (int t = 2; t <= t_max; t++) {
        double *temp;
        // Sending halo cells
        if (my_rank) {
            MPI_Isend(cur + 1, 1, MPI_DOUBLE, left, t, MPI_COMM_WORLD,
                    &request);
            MPI_Request_free(&request);
        }
        if (my_rank != num_tasks - 1) {
            MPI_Isend(cur + local_size, 1, MPI_DOUBLE, right, t,
                    MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        // Receiving halo cells
        if (my_rank != num_tasks - 1) {
            MPI_Recv(cur + local_size + 1, 1, MPI_DOUBLE, right, t,
                    MPI_COMM_WORLD, &status);
        }
        if (my_rank) {
            MPI_Recv(cur, 1, MPI_DOUBLE, left, t, MPI_COMM_WORLD, &status);
        }

        for (int i = 1; i <= local_size; i++) {
            new[i] = 2 * cur[i] - old[i] + 0.2 * (cur[i-1] -
                    (2 * cur[i] - cur [i + 1]));
        }

        temp = old;
        old = cur;
        cur = new;
        new = temp;

    }

    /* You should return a pointer to the array with the final results. */
    return cur;
}


