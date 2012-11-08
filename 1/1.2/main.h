/*
 * Headerfile for the infinite prime generator
 * By David van Erkelens and Jelte Fennema
 * As published on github.com/David1209
 */

#define BUFSIZE 50

typedef struct {
    int data[BUFSIZE];
    int used;
    int nextin;
    int nextout;
    pthread_mutex_t *buflock;
    pthread_cond_t *items;
    pthread_cond_t *space;
} Buffer;

void *thread_routine(void *a);