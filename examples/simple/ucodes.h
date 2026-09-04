#pragma once

#include "CgGate/rts.h"

#include <iostream>
#include <fstream>
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

void print_iter(int iter)
{
	if (iter % 100 == 0)
	{
		std::cout << "IT: " << iter << "\n";
	}
}

void init_el(int ii, int jj, int nb, int mb, int n, int m, double* a)
{
	const auto in_n = n - 2;
	const auto in_m = m - 2;

	const double t_disc = 1.0;
	const double t_area = 65.0;

	const auto glob_n = in_n * nb;
	const auto glob_m = in_m * mb;

    const double radius = glob_n / 6.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
	    	const auto ind = i * m + j;
            /* Distance of point i, j from the origin */
            const auto glob_i = ii * n + i;
            const auto glob_j = jj * m + j;
            const auto dx = glob_i - glob_n / 2 + 1;
            const auto dy = glob_j - glob_m / 2 + 1;
            if (dx * dx + dy * dy < radius * radius) {
                a[ind] = t_disc;
            } else {
                a[ind] = t_area;
            }
        }
    }
}

void print_el(int ii, int jj, int n, int m, const double* a)
{
	const auto in_n = n - 2;
	const auto in_m = m - 2;
	std::ofstream of{"of_" + std::to_string(ii) + "_" + std::to_string(jj), std::ios::binary};
	of.write(reinterpret_cast<const char*>(&in_n), sizeof(int));
	of.write(reinterpret_cast<const char*>(&in_m), sizeof(int));
	for (int i = 1; i <= in_n; ++i)
	{
		for (int j = 1; j <= in_m; ++j)
		{
			of.write(reinterpret_cast<const char*>(&a[i * m + j]), sizeof(double));
		}
	}
}

void calc_main(
	double a,
	double dt,
	double dx,
	double dy,
	int n,
	int m,
	const double* prev,
	int in_n_unused,
	int in_m_unused,
	double* curr)
{
	const auto in_n = n - 2;
	const auto in_m = m - 2;
	const auto dx2 = dx * dx;
	const auto dy2 = dy * dy;

	for (std::size_t i = 0; i < m; ++i) {
		curr[i] = prev[i];
		curr[(n - 1) * m + i] = prev[(n - 1) * m + i];
	}
	for (std::size_t i = 0; i < n; ++i)
	{
		curr[i * m] = prev[i * m];
		curr[i * m + m -1] = prev[i * m + m -1];
	}
	for (int i = 1; i <= in_n; i++) {
		for (int j = 1; j <= in_m; j++) {
			int ind = i * m + j;
			int ip = (i + 1) * m + j;
			int im = (i - 1) * m + j;
			int jp = i * m + j + 1;
			int jm = i * m + j - 1;
			curr[ind] = prev[ind] + a * dt *
		    	((prev[ip] - 2.0 * prev[ind] + prev[im]) / dx2 +
		    	(prev[jp] - 2.0 * prev[ind] + prev[jm]) / dy2);
		}
	}
}

void init_el3d(int ii, int jj, int kk, int n, int m, int l, double* a)
{
	a[0] = ii;
	a[1] = jj;
	a[2] = kk;
	for (int i = 3; i < n * m * l; ++i)
	{
		a[i] = -1;
	}
}

void print_el3d(int ii, int jj, int kk, int n, int m, int l, const double* a)
{
        std::cout << "II: " << ii << ", JJ: " << jj << ", KK: " << kk << "\n";
        for (int i = 0; i < n * m * l; ++i)
        {
        	std::cout << a[i] << ' ';
        }
        std::cout << "\n";
}