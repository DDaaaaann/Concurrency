/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include "omp.h"
#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old, double *cur, double *new)
{
    /*
     * Your implementation should go here.
     */

    for (int t = 2; t <= t_max; t++) {
        double *temp;

        # pragma omp parallel for
        for (int i = 1; i < i_max - 1; i++) {
            new[i] = 2 * cur[i] - old[i] + 0.2 * (cur[i-1] -
                    (2 * cur[i] - cur [i + 1]));
        }

        temp = old;
        old = cur;
        cur = new;
        new = temp;
    }

    return cur;
}
