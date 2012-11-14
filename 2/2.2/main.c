/*
 * Collective Communication
 * MPI Assignment (CPP Assignment 2.2)
 * By David van Erkelens and Jelte Fennema
 * Department Of Computer Science
 * University of Amsterdam
 *
 * As published on github.com/David1209
 */

// TODO: - Cleanup
//       - Commenting

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define TAG 1337

int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm communicator);

int main(int argc, char *argv[])
{
    int rc, sendtask, nrtasks, this;
    char message[128];
    MPI_Status status;
    rc = MPI_Init(&argc, &argv);
    if(rc != MPI_SUCCESS)
    {
        fprintf(stderr, "Unable to set up MPI.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    sendtask = 5;
    MPI_Comm_size(MPI_COMM_WORLD, &nrtasks);
    if(sendtask >= nrtasks)
        sendtask = nrtasks - 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &this);
    if(this == sendtask)
        strcpy(message, "This is the message we're sending.");
    MYMPI_Bcast(message, 128, MPI_CHAR, sendtask, MPI_COMM_WORLD);
    if(this != sendtask)
        printf("Process %d received message \"%s\".\n", this, message);
    MPI_Finalize();
    return 0;
}

int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
    int root, MPI_Comm communicator)
{
    int process, number;
    MPI_Status status;
    MPI_Comm_rank(communicator, &process);
    MPI_Comm_size(communicator, &number);
    if(process == root)
    {
        int i;
        for(i = 0; i < number; i++)
        {
            if(i == root) continue;
            MPI_Send(buffer, count, datatype, i, TAG, communicator);
            //printf("Process %d send the message to process %d\n", root, i);
        }
    }
    else
    {
        MPI_Recv(buffer, count, datatype, root, TAG, communicator, &status);
        //printf("Message received in process %d from process %d.\n",
        //    process, root);
    }
}
