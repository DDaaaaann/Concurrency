/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_RESET "\x1b[0m"

/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    /*
     * After each timestep, you should swap the buffers around. Watch out none
     * of the threads actually use the buffers at that time.
     */
    pthread_t thread_ids[num_threads];
    calc_info_t *info;
    void *result;

    for (int j = 0; j < t_max; j++) {
        next_array[0] = 0;
        next_array[i_max - 1] = 0;
        for (int i = 0; i < num_threads; i++) {
            int error;

            info = malloc(sizeof(calc_info_t));
            info->old_array = old_array;
            info->current_array = current_array;
            info->next_array = next_array;
            info->i_start = i * i_max / num_threads;
            info->i_end = (i + 1) * i_max / num_threads;

            if (info->i_start == 0) {
                info->i_start = 1;
            }
            if (info->i_end == i_max) {
                info->i_end = i_max - 1;
            }


            error = pthread_create(&thread_ids[i], NULL, &calculate, info);
            if (error) {
                printf("failed to create thread, errorcode: %d\n", error);
                return current_array;
            }
        }

        for(int i = 0; i < num_threads; i++) {
            pthread_join(thread_ids[i], &result);
            free(result);
        }

        free(old_array);
        old_array = current_array;
        current_array = next_array;
        next_array = malloc(sizeof(double) * i_max);
    }

    /* You should return a pointer to the array with the final results. */
    return current_array;
}

void *calculate(void *argument) {
    calc_info_t *info = (calc_info_t*) argument;
    double *current_array = info->current_array, *old_array = info->old_array;
    for (int i = info->i_start; i < info->i_end; i++) {
        info->next_array[i] = 2 * current_array[i] - old_array[i] + 0.2 *
            (current_array[i-1] - (2 * current_array[i] - current_array[i+1]));
    }
    return argument;
}
