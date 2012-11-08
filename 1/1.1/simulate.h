/*
 * simulate.h
 */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


double *simulate(const int i_max, const int t_max, const int num_cpus,
        double *old_array, double *current_array, double *next_array);

void *calculate(void*);
typedef struct {
    double *old_array;
    double *current_array;
    double *next_array;
} arrays_t;

typedef struct {
    arrays_t *arrays;
    int i_start;
    int i_end;
    int t_max;
    int *finished_threads;
    pthread_mutex_t *lock;
    pthread_cond_t *thread_done;
    pthread_cond_t *arrays_switched;
} calc_info_t;
