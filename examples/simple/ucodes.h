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
		std::cout << std::format("{}", a[i]) << ' ';
	}
	std::cout << "\n";
}

// Fills the interior of a 3D block with a value that uniquely identifies the
// owning block (ii, jj, kk), and marks every ghost cell (the one-cell-thick
// layer on all six faces) with a sentinel so pysol_update_array's halo
// exchange can be verified by check_ghost_test3d below.
void init_ghost_test3d(int ii, int jj, int kk, int n, int m, int l, double* a)
{
	const double id = static_cast<double>(ii) * 1000000.0 + static_cast<double>(jj) * 1000.0 + static_cast<double>(kk);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			for (int k = 0; k < l; ++k)
			{
				const auto idx = k * n * m + i * m + j;
				const bool interior = (i > 0 && i < n - 1 && j > 0 && j < m - 1 && k > 0 && k < l - 1);
				a[idx] = (interior ? id : -1.0);
			}
		}
	}
}

// Verifies the result of pysol_update_array on a 3D array initialized by
// init_ghost_test3d. Because every interior cell of a block holds a single
// constant equal to the owning block's id, the ghost layer written by the
// halo exchange must, for any face-aligned ghost cell, equal the id of the
// block on that side (or the block's own id at a domain boundary, where the
// exchange falls back to the block itself). Edge/corner ghost cells are
// combinations of up to three neighbouring blocks, so they are accepted
// against the set of candidate ids rather than a single expected value.
// Writes "<mismatches> <unset>" to check3d_<ii>_<jj>_<kk> so the driving
// Python script can assert a clean pass across every block.
void check_ghost_test3d(int ii, int jj, int kk, int nb, int mb, int lb, int n, int m, int l, const double* a)
{
	const auto block_id = [](int bi, int bj, int bk) {
		return static_cast<double>(bi) * 1000000.0 + static_cast<double>(bj) * 1000.0 + static_cast<double>(bk);
	};
	const auto neighbor_id = [&](int di, int dj, int dk) {
		int ni = ii + di;
		int nj = jj + dj;
		int nk = kk + dk;
		if (ni < 0 || ni >= nb) ni = ii;
		if (nj < 0 || nj >= mb) nj = jj;
		if (nk < 0 || nk >= lb) nk = kk;
		return block_id(ni, nj, nk);
	};
	const double own_id = block_id(ii, jj, kk);

	int mismatches = 0;
	int unset = 0;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			for (int k = 0; k < l; ++k)
			{
				const auto idx = k * n * m + i * m + j;
				const bool ghost_i = i == 0 || i == n - 1;
				const bool ghost_j = j == 0 || j == m - 1;
				const bool ghost_k = k == 0 || k == l - 1;

				if (!ghost_i && !ghost_j && !ghost_k)
				{
					if (a[idx] != own_id) ++mismatches;
					continue;
				}
				if (ghost_i && ghost_k) {
					continue;
				}
				if (ghost_j && ghost_k) {
					continue;
				}
				if (ghost_i && ghost_j) {
					continue;
				}
				if (a[idx] < 0.0)
				{
					++unset;
					continue;
				}

				double candidates[4];
				int ncand = 0;
				candidates[ncand++] = own_id;
				if (ghost_i) candidates[ncand++] = neighbor_id(i == 0 ? -1 : 1, 0, 0);
				if (ghost_j) candidates[ncand++] = neighbor_id(0, j == 0 ? -1 : 1, 0);
				if (ghost_k) candidates[ncand++] = neighbor_id(0, 0, k == 0 ? -1 : 1);

				bool ok = false;
				for (int c = 0; c < ncand; ++c)
				{
					if (a[idx] == candidates[c]) { ok = true; break; }
				}
				if (!ok) ++mismatches;
			}
		}
	}

	std::ofstream of{"check3d_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(kk)};
	of << mismatches << " " << unset << "\n";
	std::cout << "CHECK3D[" << ii << "," << jj << "," << kk << "] mismatches=" << mismatches
	          << " unset=" << unset << (mismatches == 0 && unset == 0 ? " PASS" : " FAIL") << "\n";
}

void printv(double v)
{
	std::cout << "V = " << v << "\n";
}

void init_el_3d(int ii, int jj, int kk, int n, int m, int l, double* a)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			for (int k = 0; k < l; ++k)
			{
				const auto idx = k * n * m + i * m + j;
				a[idx] = (ii * jj * kk);
			}
		}
	}
}
