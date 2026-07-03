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

void init_vec(int ii, int n, float* a)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = ii;
    }
}

void print_vec(int ii, int n, const float* a)
{
    std::cout << "A[" << ii << "]: ";
    for (int i = 0; i < n; ++i)
    {
        std::cout << a[i] << ' ';
    }
    std::cout << "\n";
}