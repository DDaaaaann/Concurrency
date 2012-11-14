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
double *simulate(const int i_max, const int t_max, double *old_array,
        double *current_array, double *next_array)
{
    int *argc, num_tasks, my_rank, rc, local_size, left, right;
    double *cur, *old, *new;
    char ***argv;
    MPI_Request request;
    MPI_Status status;

    if ((rc = MPI_Init(argc, argv)) != MPI_SUCCESS) {
        fprintf(stderr, "Unable to setup MPI");

        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks); // Determine number of tasks
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Determine task id

    local_size = i_max / num_tasks;
    left = my_rank - 1;
    right = my_rank + 1;
    cur = malloc(sizeof(double) * (local_size + 2));
    old = malloc(sizeof(double) * (local_size + 2));
    new = malloc(sizeof(double) * (local_size + 2));

    // Master sends data and workers receive it
    if (!my_rank) {
        for (int i = 1; i < num_tasks; i++) {
            MPI_Isend(old_array + (local_size * i), local_size, MPI_DOUBLE, i,
                    0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
            MPI_Isend(current_array + (local_size * i), local_size, MPI_DOUBLE,
                    i, 1, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        free(old);
        free(cur);
        free(new);
        old = old_array;
        cur = current_array;
        new = next_array;
        new[0] = 0;
        new[i_max - 1] = 0;
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
    if (my_rank) {
        MPI_Send(cur + 1, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }
    else {
        for (int i = 1; i < num_tasks; i++) {
            MPI_Recv(cur + (i * local_size), local_size, MPI_DOUBLE, i, 0,
                    MPI_COMM_WORLD, &status);
        }
    }





    MPI_Finalize(); // Shutdown MPI runtime

    /* You should return a pointer to the array with the final results. */
    return cur;
}


