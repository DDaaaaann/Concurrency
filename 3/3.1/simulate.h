/*
 * simulate.h
 */

#pragma once

double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old, double *cur, double *new, int schedule,
        const int chunk_size);
