#pragma once

#include "CgGate/rts.h"

#include <iostream>
#include <mpi.h>

void hello_world_k()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Hello world from pysol from " << rank << "!\n";
}

void print(int val)
{
    std::cout << "Val: " << val << "\n";
}