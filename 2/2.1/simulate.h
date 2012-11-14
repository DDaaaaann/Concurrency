/*
 * simulate.h
 */

#pragma once

double *simulate(const int local_size, const int t_max, int num_tasks,
        int my_rank, double *old, double *cur, double *new);
