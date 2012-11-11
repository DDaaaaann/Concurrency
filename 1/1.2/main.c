/*
 * BROTIP #2486: Since you can't live forever,
 * create something that lives forever.
 *
 * Infinite prime number generator.
 * By Jelte Fennema and David van Erkelens
 * Department of Computer Science, University of Amsterdam
 * As published on github.com/David1209
 *
 * Contact: {David.vanErkelens, Jelte.Fennema}@student.uva.nl
 *
 * 8 november 2012
 */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"

/*
 * The main function will only be executed in the first thread. This is the
 * natural number generator which sends natural numbers down the pipeline
 * to be processed by the other threads.
 */
int main()
{
    /* Creating the buffer for the next consumer, and setting the
     * first prime number
     */
    Buffer *to_cons;
    int i = 2;

    /* Create all the pthread settings like the mutex, and set the
     * buffer variables to the correct values.
     */
    pthread_t next_thread;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t items = PTHREAD_COND_INITIALIZER,
            space = PTHREAD_COND_INITIALIZER;
    to_cons = malloc(sizeof(Buffer));
    to_cons->buflock = &lock;
    to_cons->items = &items;
    to_cons->space = &space;
    to_cons->used = 0;
    to_cons->nextin = 0;
    to_cons->nextout = 0;

    /* Create the next thread. */
    pthread_create(&next_thread, NULL, thread_routine, (void *)to_cons);
    while(1)
    {
        /* Ask for permission to use the critical zone. */
        pthread_mutex_lock(to_cons->buflock);

        /* Continue only if there's space, else wait for space. */
        while(to_cons->used == BUFSIZE)
        {
            pthread_cond_wait(to_cons->space, to_cons->buflock);
        }
        /* Available space, add a new number to the buffer. */
        to_cons->data[to_cons->nextin] = i;
        to_cons->nextin ++;
        to_cons->nextin %= BUFSIZE;
        to_cons->used ++;

        /* Unlock and signal */
        pthread_cond_signal(to_cons->items);
        pthread_mutex_unlock(to_cons->buflock);
        i++;
    }
    return 0;
}

/* The function used by all threads except the first one: this function
 * reads the numbers from the pipeline, checks if it's the first one it reads
 * (meaning it's a prime!). When it's the first, it will be stored for later
 * use. When the number can be divided by the prime, it is discarded. When
 * it can't be divided by the prime, it's send further down the pipeline
 * to be processed by the other threads and possibly be identified as a prime.
 */
void *thread_routine(void *a)
{
    /* Creating all the necessary variables */
    int prime = 0, next = 0;
    pthread_t next_thread;
    Buffer *to_cons, *from_prod;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t items = PTHREAD_COND_INITIALIZER,
            space = PTHREAD_COND_INITIALIZER;
    to_cons = malloc(sizeof(Buffer));

    /* The buffer send by the previous thread - the 'producer' */
    from_prod = (Buffer *) a;
    to_cons->buflock = &lock;
    to_cons->items = &items;
    to_cons->space = &space;
    to_cons->used = 0;
    to_cons->nextin = 0;
    to_cons->nextout = 0;
    while(1)
    {
        /* Lock the buffer for use by this thread. */
        int number;
        pthread_mutex_lock(from_prod->buflock);
        /* Check whether there are items in the pipeline */
        while(from_prod->used == 0)
        {
            pthread_cond_wait(from_prod->items, from_prod->buflock);
        }
        /* Available items */
        if(prime == 0)
        {
            /* If the number is the first number this thread processes... */
            prime = from_prod->data[from_prod->nextout];
            from_prod->nextout ++;
            from_prod->nextout %= BUFSIZE;
            from_prod->used --;
            /* It's a prime! */
            printf("%d\n", prime);
            pthread_cond_signal(from_prod->space);
            pthread_mutex_unlock(from_prod->buflock);
            continue;
        }
        /* Not the first number. Two options: discaring (if dividable)... */
        number = from_prod->data[from_prod->nextout];
        from_prod->nextout ++;
        from_prod->nextout %= BUFSIZE;
        from_prod->used --;
        pthread_cond_signal(from_prod->space);
        pthread_mutex_unlock(from_prod->buflock);
        if(number % prime == 0)
        {
            continue;
        }
        /* ... or sending it down the pipeline. */
        if(next == 0)
        {
            /* Create new thread if there isn't a next thread yet. */
            pthread_create(&next_thread, NULL, &thread_routine,
                    (void *)to_cons);
            next = 1337;
        }
        /* Lock the next buffer - this thread is now functioning as a
         * producer instead of a consumer. */
        pthread_mutex_lock(to_cons->buflock);
        while(to_cons->used == BUFSIZE)
        {
            pthread_cond_wait(to_cons->space, to_cons->buflock);
        }
        /* Available space */
        to_cons->data[to_cons->nextin] = number;
        to_cons->nextin ++;
        to_cons->nextin %= BUFSIZE;
        to_cons->used ++;
        pthread_cond_signal(to_cons->items);
        pthread_mutex_unlock(to_cons->buflock);
    }
}
