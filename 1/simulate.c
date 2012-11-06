/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */


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
    int finished_threads = 0;
    arrays_t *arrays = malloc(sizeof(arrays_t));
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t thread_done = PTHREAD_COND_INITIALIZER,
                   arrays_switched = PTHREAD_COND_INITIALIZER;

    // initialize array struct;
    next_array = malloc(sizeof(double) * i_max);
    next_array[0] = 0;
    next_array[i_max - 1] = 0;

    arrays->old_array = old_array;
    arrays->current_array = current_array;
    arrays->next_array = next_array;

    for (int i = 0; i < num_threads; i++) {
        int error;

        info = malloc(sizeof(calc_info_t));
        info->arrays = arrays;
        info->i_start = i * i_max / num_threads;
        info->i_end = (i + 1) * i_max / num_threads;
        info->t_max = t_max;
        info->finished_threads = &finished_threads;
        info->lock = &lock;
        info->thread_done = &thread_done;
        info->arrays_switched = &arrays_switched;

        if (info->i_start == 0) {
            info->i_start = 1;
        }
        if (info->i_end == i_max) {
            info->i_end = i_max - 1;
        }


        error = pthread_create(&thread_ids[i], NULL, &calculate, info);
        if (error) {
            return current_array;
        }
    }

    for (int i = 0; i < t_max; i++) {
        // wait until all threads have finished filling of next_array
        pthread_mutex_lock(&lock);
        while (finished_threads < num_threads) {
            pthread_cond_wait(&thread_done, &lock);
        }

        // switch arrays around and let the threads know when finished so they
        // can calculate the next couple;
        double *temp = old_array;
        arrays->old_array = arrays->current_array;
        arrays->current_array = arrays->next_array;
        arrays->next_array = arrays->old_array;
        finished_threads = 0;
        pthread_cond_broadcast(&arrays_switched);
        pthread_mutex_unlock(&lock);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(thread_ids[i], &result);
        free(result);
    }


    /* You should return a pointer to the array with the final results. */

    return arrays->current_array;
}

void *calculate(void *argument) {
    calc_info_t *info = (calc_info_t*) argument;
    arrays_t *arrays = info->arrays;
    for (int j = 0; j < info->t_max; j++) {
        for (int i = info->i_start; i < info->i_end; i++) {
            arrays->next_array[i] = 2 * arrays->current_array[i] -
                arrays->old_array[i] + 0.2 * (arrays->current_array[i-1] -
                (2 * arrays->current_array[i] - arrays->current_array[i+1]));
        }


        // wait for the rest of the thread to finish and the main thread to
        // switch the arrays around

        pthread_mutex_lock(info->lock);
        *(info->finished_threads) += 1;
        pthread_cond_signal(info->thread_done);
        pthread_cond_wait(info->arrays_switched, info->lock);
        pthread_mutex_unlock(info->lock);
    }
    return argument;
}
