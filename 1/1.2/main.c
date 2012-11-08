/*
 * BROTIP #2486: Since you can't live forever,
 * create something that lives forever.
 * 
 * Infinite prime number generator.
 * By Jelte Fennema and David van Erkelens
 * As published on github.com/David1209
 */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"

typedef struct {
    int data[BUFSIZE];
    int used;
    int nextin;
    int nextout;
    pthread_mutex_t *buflock;
    pthread_cond_t *items;
    pthread_cond_t *space;
} Buffer;

int main()
{
    Buffer *to_cons;
    int i = 2;
    pthread_t next_thread;
    pthread_attr_t attr;
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
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
    pthread_create(&next_thread, NULL, thread_routine, (void *)to_cons);
    while(1)
    {
        pthread_mutex_lock(to_cons->buflock);
        while(to_cons->used == BUFSIZE)
        {
            pthread_cond_wait(to_cons->space, to_cons->buflock);
        }
        /* Available space */
        //printf("Adding %d to the first queue.\n", i);
        to_cons->data[to_cons->nextin] = i;
        to_cons->nextin ++;
        to_cons->nextin %= BUFSIZE;
        to_cons->used ++;
        pthread_cond_signal(to_cons->items);
        pthread_mutex_unlock(to_cons->buflock);
        i++;
    }
    return 0;
}

void *thread_routine(void *a)
{
    int prime = 0, next = 0;
    pthread_t next_thread;
    Buffer *to_cons, *from_prod;
    pthread_attr_t attr;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t items = PTHREAD_COND_INITIALIZER,
            space = PTHREAD_COND_INITIALIZER;
    to_cons = malloc(sizeof(Buffer));
    from_prod = (Buffer *) a;
    to_cons->buflock = &lock;
    to_cons->items = &items;
    to_cons->space = &space;
    to_cons->used = 0;
    to_cons->nextin = 0;
    to_cons->nextout = 0;
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
    while(1)
    {
        int number;
        pthread_mutex_lock(from_prod->buflock);
        while(!(from_prod->used > 0))
        {
            pthread_cond_wait(from_prod->items, from_prod->buflock);
        }
        //printf("Processing %d from the queue.\n", from_prod->data[from_prod->nextout]);
        /* Available items */
        if(prime == 0)
        {
            prime = from_prod->data[from_prod->nextout];
            from_prod->nextout ++;
            from_prod->nextout %= BUFSIZE;
            from_prod->used --;
            //printf("Prime found: %d\n", prime);
            printf("%d ", prime);
            pthread_mutex_unlock(from_prod->buflock);
            continue;
        }
        number = from_prod->data[from_prod->nextout];
        from_prod->nextout ++;
        from_prod->nextout %= BUFSIZE;
        from_prod->used --;
        pthread_mutex_unlock(from_prod->buflock);
        if(number % prime == 0)
        {
            continue;
        }
        if(next == 0)
        {
            //printf("Creating new thread for %d\n", number);
            pthread_create(&next_thread, NULL, &thread_routine, 
                    (void *)to_cons);
            next = 1337;
        }
        pthread_mutex_lock(to_cons->buflock);
        //printf("testje.\n");
        while(!(to_cons->used < BUFSIZE))
        {
            pthread_cond_wait(to_cons->space, to_cons->buflock);
        }
        /* Available space */
        //printf("Sending %d to the next thread.\n", number);
        to_cons->data[to_cons->nextin] = number;
        to_cons->nextin ++;
        to_cons->nextin %= BUFSIZE;
        to_cons->used ++;
        pthread_cond_signal(to_cons->items);
        pthread_mutex_unlock(to_cons->buflock);
    }
}