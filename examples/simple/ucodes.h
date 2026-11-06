#pragma once

#include "CgGate/rts.h"

#include <iostream>
#include <fstream>
#include <string>
#include <type_traits>
#include <mutex>

#include <mpi.h>

std::mutex mtx_;

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

// Fills an entire 3D block (every cell, not just an interior - this
// exchange has no separate ghost layer) with a value that uniquely
// identifies the owning block (ii, jj, kk), for
// pysol_update_array_3d_26_full_sum's additive halo exchange to be
// verified by check_ghost_test3d_sum26 below.
void init_ghost_test3d_sum26(int ii, int jj, int kk, int n, int m, int p, double* a)
{
	const double id = static_cast<double>(ii) * 1000000.0 + static_cast<double>(jj) * 1000.0 + static_cast<double>(kk);
	for (int i = 0; i < n * m * p; ++i) a[i] = id;
}

// Verifies the result of pysol_update_array_3d_26_full_sum on a 3D array
// initialized by init_ghost_test3d_sum26. Every block is first its own
// uniform id, and the exchange then additively deposits, on top, every
// EXISTING neighbor's id into the 2-cell-wide slab shared with it - so the
// expected value of any cell is its own id plus the id of every
// neighboring block whose shared overlap slab covers that cell. A
// direction's slab covers cell (li, lj, lk) along a given axis if that
// axis's direction digit is "mid" (no restriction) or if the cell's index
// on that axis falls in the matching 2-wide lo/hi band - this reproduces,
// per axis, the same band selection _write_ghost_26_full_sum_<type> uses
// (see rts.cpp's write_ghost_26_full_sum), just applied to scalar block
// ids instead of raw array data; a direction with no actual neighbor block
// (domain boundary) contributes nothing, matching the exchange's own
// boundary handling (no self-substitution for an additive sum). Writes a
// single mismatch count to check3d_sum26_<ii>_<jj>_<kk> so the driving
// Python script can assert a clean pass across every block.
void check_ghost_test3d_sum26(int ii, int jj, int kk, int nb, int mb, int lb,
                               int n, int m, int p, const double* a)
{
	const auto block_id = [](int bi, int bj, int bk) {
		return static_cast<double>(bi) * 1000000.0 + static_cast<double>(bj) * 1000.0 + static_cast<double>(bk);
	};
	const auto flat = [m, p](int i, int j, int k) { return (i * m + j) * p + k; };
	const double own_id = block_id(ii, jj, kk);

	int mismatches = 0;
	for (int li = 0; li < n; ++li)
	{
		for (int lj = 0; lj < m; ++lj)
		{
			for (int lk = 0; lk < p; ++lk)
			{
				double expected = own_id;
				for (int dx = 0; dx <= 2; ++dx)
				{
					const bool x_ok = dx == 1 || (dx == 0 && li <= 1) || (dx == 2 && li >= n - 2);
					if (!x_ok) continue;
					for (int dy = 0; dy <= 2; ++dy)
					{
						const bool y_ok = dy == 1 || (dy == 0 && lj <= 1) || (dy == 2 && lj >= m - 2);
						if (!y_ok) continue;
						for (int dz = 0; dz <= 2; ++dz)
						{
							if (dx == 1 && dy == 1 && dz == 1) continue; // self - already in own_id
							const bool z_ok = dz == 1 || (dz == 0 && lk <= 1) || (dz == 2 && lk >= p - 2);
							if (!z_ok) continue;

							const int ni = ii + (dx - 1);
							const int nj = jj + (dy - 1);
							const int nk = kk + (dz - 1);
							if (ni < 0 || ni >= nb || nj < 0 || nj >= mb || nk < 0 || nk >= lb) continue;

							expected += block_id(ni, nj, nk);
						}
					}
				}

				if (a[flat(li, lj, lk)] != expected) ++mismatches;
			}
		}
	}

	std::ofstream of{"check3d_sum26_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(kk)};
	of << mismatches << "\n";
	std::cout << "CHECK3D_SUM26[" << ii << "," << jj << "," << kk << "] mismatches=" << mismatches
	          << (mismatches == 0 ? " PASS" : " FAIL") << "\n";
}

// Adds a fixed increment (1.0) to every interior cell of a block. The
// ghost layer is copied forward from a_in unchanged (it is about to be
// refreshed by the next pysol_update_array call anyway) rather than left
// as whatever "a" happened to already hold: A's backing storage is a
// multi-slot ring buffer (nversions>1, confirmed in the generated .cpp for
// update_array_3d_loop_test.py), so a_in and a are genuinely distinct
// physical buffers, not the same memory under two version labels - "a"
// starts out as unrelated leftover data from several versions ago, not a
// copy of a_in, until this kernel copies it explicitly. Used by
// update_array_3d_loop_test.py to modify the array between each of many
// repeated exchange iterations.
void modify_ghost_test3d(int ii, int jj, int kk,
	int n, int m, int l, const double* a_in, int on, int om, int ol,  double* a)
{
	(void)ii; (void)jj; (void)kk; (void)on; (void)om; (void)ol;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			for (int k = 0; k < l; ++k)
			{
				const auto idx = k * n * m + i * m + j;
				a[idx] = a_in[idx];
				const bool interior = (i > 0 && i < n - 1 && j > 0 && j < m - 1 && k > 0 && k < l - 1);
				if (interior) a[idx] += 1.0;
			}
		}
	}
}

// Same check as check_ghost_test3d, but every block id is offset by a
// uniform constant - e.g. the total of many modify_ghost_test3d increments
// applied before this check runs, so the exchanged values are expected to
// be "id + offset" rather than plain "id". Writes to
// check3d_loop_<ii>_<jj>_<kk> (a distinct prefix from check_ghost_test3d's
// output) so the two tests' result files never collide.
void check_ghost_test3d_offset(int ii, int jj, int kk, int nb, int mb, int lb, double offset,
                                int n, int m, int l, const double* a)
{
	const auto block_id = [offset](int bi, int bj, int bk) {
		return static_cast<double>(bi) * 1000000.0 + static_cast<double>(bj) * 1000.0 + static_cast<double>(bk) + offset;
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

	std::ofstream of{"check3d_loop_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(kk)};
	of << mismatches << " " << unset << "\n";
	std::cout << "CHECK3D_LOOP[" << ii << "," << jj << "," << kk << "] mismatches=" << mismatches
	          << " unset=" << unset << (mismatches == 0 && unset == 0 ? " PASS" : " FAIL") << "\n";
}

void print_el2d(int ii, int jj, int n, int m, const double* a)
{
	std::cout << "II: " << ii << ", JJ: " << jj << "\n";
	for (int i = 0; i < n * m; ++i)
	{
		std::cout << std::format("{}", a[i]) << ' ';
	}
	std::cout << "\n";
}

// 2D analogue of init_ghost_test3d: fills the interior of a 2D block with a
// value that uniquely identifies the owning block (ii, jj), and marks every
// ghost cell (the one-cell-thick layer on all four edges) with a sentinel so
// pysol_update_array's halo exchange can be verified by check_ghost_test2d*
// below.
void init_ghost_test2d(int ii, int jj, int n, int m, double* a)
{
	const double id = static_cast<double>(ii) * 1000.0 + static_cast<double>(jj);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			const auto idx = i * m + j;
			const bool interior = (i > 0 && i < n - 1 && j > 0 && j < m - 1);
			a[idx] = (interior ? id : -1.0);
		}
	}
}

// 2D analogue of check_ghost_test3d. Every interior cell of a block holds a
// single constant equal to the owning block's id, so the ghost layer
// written by the halo exchange must, for any edge-aligned ghost cell, equal
// the id of the block on that side (or the block's own id at a domain
// boundary, where the exchange falls back to the block itself). Corner
// ghost cells are a combination of two neighbouring blocks' halos and a
// diagonal neighbor never populated by a face-wise exchange, so they are
// skipped entirely rather than checked. Writes "<mismatches> <unset>" to
// check2d_<ii>_<jj> so the driving Python script can assert a clean pass
// across every block.
void check_ghost_test2d(int ii, int jj, int nb, int mb, int n, int m, const double* a)
{
	const auto block_id = [](int bi, int bj) {
		return static_cast<double>(bi) * 1000.0 + static_cast<double>(bj);
	};
	const auto neighbor_id = [&](int di, int dj) {
		int ni = ii + di;
		int nj = jj + dj;
		if (ni < 0 || ni >= nb) ni = ii;
		if (nj < 0 || nj >= mb) nj = jj;
		return block_id(ni, nj);
	};
	const double own_id = block_id(ii, jj);

	int mismatches = 0;
	int unset = 0;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			const auto idx = i * m + j;
			const bool ghost_i = i == 0 || i == n - 1;
			const bool ghost_j = j == 0 || j == m - 1;

			if (!ghost_i && !ghost_j)
			{
				if (a[idx] != own_id) ++mismatches;
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

			double candidates[2];
			int ncand = 0;
			candidates[ncand++] = own_id;
			if (ghost_i) candidates[ncand++] = neighbor_id(i == 0 ? -1 : 1, 0);
			if (ghost_j) candidates[ncand++] = neighbor_id(0, j == 0 ? -1 : 1);

			bool ok = false;
			for (int c = 0; c < ncand; ++c)
			{
				if (a[idx] == candidates[c]) { ok = true; break; }
			}
			if (!ok) ++mismatches;
		}
	}

	std::ofstream of{"check2d_" + std::to_string(ii) + "_" + std::to_string(jj)};
	of << mismatches << " " << unset << "\n";
	std::cout << "CHECK2D[" << ii << "," << jj << "] mismatches=" << mismatches
	          << " unset=" << unset << (mismatches == 0 && unset == 0 ? " PASS" : " FAIL") << "\n";
}

// 2D analogue of modify_ghost_test3d: adds a fixed increment (1.0) to every
// interior cell of a block, leaving the ghost layer untouched - it is about
// to be refreshed by the next pysol_update_array call anyway. Used by
// update_array_2d_loop_test.py to modify the array between each of many
// repeated exchange iterations.
void modify_ghost_test2d(int ii, int jj,
	int n, int m, const double* a_in, int on, int om, double* a)
{
	(void)ii; (void)jj; (void)on; (void)om;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			const auto idx = i * m + j;
			a[idx] = a_in[idx];
			const bool interior = (i > 0 && i < n - 1 && j > 0 && j < m - 1);
			if (interior) a[idx] += 1.0;
		}
	}
}

// Same check as check_ghost_test2d, but every block id is offset by a
// uniform constant - e.g. the total of many modify_ghost_test2d increments
// applied before this check runs, so the exchanged values are expected to
// be "id + offset" rather than plain "id". Writes to
// check2d_loop_<ii>_<jj> (a distinct prefix from check_ghost_test2d's
// output) so the two tests' result files never collide.
void check_ghost_test2d_offset(int ii, int jj, int nb, int mb, double offset,
                                int n, int m, const double* a)
{
	const auto block_id = [offset](int bi, int bj) {
		return static_cast<double>(bi) * 1000.0 + static_cast<double>(bj) + offset;
	};
	const auto neighbor_id = [&](int di, int dj) {
		int ni = ii + di;
		int nj = jj + dj;
		if (ni < 0 || ni >= nb) ni = ii;
		if (nj < 0 || nj >= mb) nj = jj;
		return block_id(ni, nj);
	};
	const double own_id = block_id(ii, jj);

	int mismatches = 0;
	int unset = 0;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			const auto idx = i * m + j;
			const bool ghost_i = i == 0 || i == n - 1;
			const bool ghost_j = j == 0 || j == m - 1;

			if (!ghost_i && !ghost_j)
			{
				if (a[idx] != own_id) ++mismatches;
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

			double candidates[2];
			int ncand = 0;
			candidates[ncand++] = own_id;
			if (ghost_i) candidates[ncand++] = neighbor_id(i == 0 ? -1 : 1, 0);
			if (ghost_j) candidates[ncand++] = neighbor_id(0, j == 0 ? -1 : 1);

			bool ok = false;
			for (int c = 0; c < ncand; ++c)
			{
				if (a[idx] == candidates[c]) { ok = true; break; }
			}
			if (!ok) ++mismatches;
		}
	}

	std::ofstream of{"check2d_loop_" + std::to_string(ii) + "_" + std::to_string(jj)};
	of << mismatches << " " << unset << "\n";
	std::cout << "CHECK2D_LOOP[" << ii << "," << jj << "] mismatches=" << mismatches
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

// A GrowingArray<T> cell is not a single T value: it is a dynamically
// growing rts::NonUpdatableGeneralGrowingVectorHolder<N, T>::Element, whose
// public `data_` member is a std::vector<T> that kernels append to (hence
// "growing"). Because ucodes.h is included before the struct type generated
// from a .sol "type T : struct {...};" declaration (itself emitted from a
// @pysol_struct class), these kernels are written as templates over the
// element-pointer type instead of naming the concrete struct/Element types
// directly; the struct type only needs to exist by the time the template is
// actually instantiated from the generated .cpp, which is always true.
template <typename ElementT>
void init_struct_cell3d(int ii, int jj, int kk, ElementT* cell)
{
	using S = typename std::decay_t<decltype(cell->data_)>::value_type;
	cell->data_.emplace_back(S{ii, static_cast<float>(jj), static_cast<double>(kk)});
}

// Verifies that the cell at (ii, jj, kk) holds exactly the single struct
// value init_struct_cell3d wrote there, and writes "<mismatches>" to
// struct3d_<ii>_<jj>_<kk> so the driving Python script can assert a clean
// pass across every cell.
template <typename ElementT>
void check_struct_cell3d(int ii, int jj, int kk, const ElementT* cell)
{
	int mismatches = 0;
	if (cell->data_.size() != 1)
	{
		++mismatches;
	}
	else
	{
		const auto& el = cell->data_[0];
		if (el.a != ii) ++mismatches;
		if (el.b != static_cast<float>(jj)) ++mismatches;
		if (el.c != static_cast<double>(kk)) ++mismatches;
	}

	std::ofstream of{"struct3d_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(kk)};
	of << mismatches << "\n";
}

// Write the result of a sum()/max()/min() reduction to a file so the driving
// Python script can compare it against an independently computed expected
// value and report PASS/FAIL.
void write_sum_result(double v)
{
	std::ofstream of{"reduce_sum_result"};
	of << v << "\n";
}

void write_max_result(double v)
{
	std::ofstream of{"reduce_max_result"};
	of << v << "\n";
}

void write_min_result(double v)
{
	std::ofstream of{"reduce_min_result"};
	of << v << "\n";
}

/*=========================================================================
 * GrowingArray race reproduction (growing_array_race_test.py)
 *
 * Standalone copy of examples/pic3d_v2's 27-direction GrowingArray
 * particle-mailbox migration pattern (push_particles_block/
 * merge_particles_block in that example's own ucodes.h), stripped of all
 * PIC physics (no force field, no density/potential solve) so that a
 * "token"'s entire trajectory is a closed-form, bit-reproducible
 * constant-velocity-plus-wall-reflection calculation a pure-Python ground
 * truth can replay exactly (see growing_array_race_test.py's verify()) -
 * any mismatch between that ground truth and what this code actually
 * produces under multithreading can only be the GrowingArray exchange
 * itself, never a physics/floating-point discrepancy between two
 * different formula implementations.
 *=========================================================================*/

// Deterministic lattice-in-disk quiet start, exactly like pic3d_v2's
// generate_particles_disk_k (same candidate-lattice-filtered-to-own-box
// idiom - called once per fragment, each filtering the SAME global
// npx*npy candidate lattice down to the points inside its own owned box
// [ii*hx,(ii+1)*hx) x [jj*hy,(jj+1)*hy) x [kk*hz,(kk+1)*hz)), but every
// token also gets a unique id (its position in the candidate lattice) and
// a nonzero, deterministic velocity computed from that id using only
// integer modulo plus double +,-,* and / of small integer constants - all
// IEEE-754-exact/correctly-rounded operations, so growing_array_race_test.py's
// pure-Python ground truth reproduces the exact same double values bit for bit.
template <typename ElementT>
void init_tokens_block(double cx, double cy, double cz, double radius,
                        double hx, double hy, double hz,
                        int ii, int jj, int kk,
                        int npx, int npy, double vmax, ElementT* cell)
{
	double x1 = ii * hx, x2 = x1 + hx;
	double y1 = jj * hy, y2 = y1 + hy;
	double z1 = kk * hz, z2 = z1 + hz;

	double dlx = (2.0 * radius) / (npx - 1);
	double dly = (2.0 * radius) / (npy - 1);
	for (int i = 0; i < npx; ++i)
	{
		double x = cx - radius + i * dlx;
		for (int j = 0; j < npy; ++j)
		{
			double y = cy - radius + j * dly;
			double dx = x - cx, dy = y - cy;
			if (dx * dx + dy * dy > radius * radius) continue;
			if (x < x1 || x >= x2 || y < y1 || y >= y2 || cz < z1 || cz >= z2) continue;

			int id = i * npy + j;
			typename std::decay_t<decltype(cell->data_)>::value_type p;
			p.id = id;
			p.x = x; p.y = y; p.z = cz;
			p.u = vmax * (2.0 * ((id % 7) / 6.0) - 1.0);
			p.v = vmax * (2.0 * ((id % 11) / 10.0) - 1.0);
			p.w = vmax * (2.0 * ((id % 13) / 12.0) - 1.0);
			cell->data_.emplace_back(p);
		}
	}
}

// Ballistic push (no force) + wall reflection + route into one of 27
// mailboxes based on which side of THIS fragment's own owned box the new
// position ends up on - byte-for-byte the same reflect+route logic as
// pic3d_v2's push_particles_block, just without the force-field gather
// (tokens move at constant velocity, assigned once at init above).
template <typename ElementTIn, typename ElementTOut>
void push_tokens_block(double tau, double xn, double yn, double zn,
                        double hx, double hy, double hz,
                        int ii, int jj, int kk,
                        const ElementTIn* cell_old,
                        ElementTOut* mail_self,
                        ElementTOut* mail_000, ElementTOut* mail_001, ElementTOut* mail_002,
                        ElementTOut* mail_010, ElementTOut* mail_011, ElementTOut* mail_012,
                        ElementTOut* mail_020, ElementTOut* mail_021, ElementTOut* mail_022,
                        ElementTOut* mail_100, ElementTOut* mail_101, ElementTOut* mail_102,
                        ElementTOut* mail_110,                       ElementTOut* mail_112,
                        ElementTOut* mail_120, ElementTOut* mail_121, ElementTOut* mail_122,
                        ElementTOut* mail_200, ElementTOut* mail_201, ElementTOut* mail_202,
                        ElementTOut* mail_210, ElementTOut* mail_211, ElementTOut* mail_212,
                        ElementTOut* mail_220, ElementTOut* mail_221, ElementTOut* mail_222)
{

	using P = typename std::decay_t<decltype(mail_self->data_)>::value_type;
	mail_self->data_.clear();
	mail_000->data_.clear(); mail_001->data_.clear(); mail_002->data_.clear();
	mail_010->data_.clear(); mail_011->data_.clear(); mail_012->data_.clear();
	mail_020->data_.clear(); mail_021->data_.clear(); mail_022->data_.clear();
	mail_100->data_.clear(); mail_101->data_.clear(); mail_102->data_.clear();
	mail_110->data_.clear();                          mail_112->data_.clear();
	mail_120->data_.clear(); mail_121->data_.clear(); mail_122->data_.clear();
	mail_200->data_.clear(); mail_201->data_.clear(); mail_202->data_.clear();
	mail_210->data_.clear(); mail_211->data_.clear(); mail_212->data_.clear();
	mail_220->data_.clear(); mail_221->data_.clear(); mail_222->data_.clear();

	double x1 = ii * hx, x2 = x1 + hx;
	double y1 = jj * hy, y2 = y1 + hy;
	double z1 = kk * hz, z2 = z1 + hz;

	for (const auto& old_p : cell_old->data_)
	{
		P p = old_p;

		p.x += tau * p.u;
		p.y += tau * p.v;
		p.z += tau * p.w;

		if (p.x < 0.0)       { p.x = -p.x;         p.u = -p.u; }
		else if (p.x >= xn)  { p.x = 2 * xn - p.x;  p.u = -p.u; }
		if (p.y < 0.0)       { p.y = -p.y;          p.v = -p.v; }
		else if (p.y >= yn)  { p.y = 2 * yn - p.y;  p.v = -p.v; }
		if (p.z < 0.0)       { p.z = -p.z;          p.w = -p.w; }
		else if (p.z >= zn)  { p.z = 2 * zn - p.z;  p.w = -p.w; }

		int dx = (p.x < x1) ? 0 : (p.x >= x2 ? 2 : 1);
		int dy = (p.y < y1) ? 0 : (p.y >= y2 ? 2 : 1);
		int dz = (p.z < z1) ? 0 : (p.z >= z2 ? 2 : 1);

		ElementTOut* dst;
		if (dx == 1 && dy == 1 && dz == 1) dst = mail_self;
		else if (dx==0 && dy==0 && dz==0) dst = mail_000;
		else if (dx==0 && dy==0 && dz==1) dst = mail_001;
		else if (dx==0 && dy==0 && dz==2) dst = mail_002;
		else if (dx==0 && dy==1 && dz==0) dst = mail_010;
		else if (dx==0 && dy==1 && dz==1) dst = mail_011;
		else if (dx==0 && dy==1 && dz==2) dst = mail_012;
		else if (dx==0 && dy==2 && dz==0) dst = mail_020;
		else if (dx==0 && dy==2 && dz==1) dst = mail_021;
		else if (dx==0 && dy==2 && dz==2) dst = mail_022;
		else if (dx==1 && dy==0 && dz==0) dst = mail_100;
		else if (dx==1 && dy==0 && dz==1) dst = mail_101;
		else if (dx==1 && dy==0 && dz==2) dst = mail_102;
		else if (dx==1 && dy==1 && dz==0) dst = mail_110;
		else if (dx==1 && dy==1 && dz==2) dst = mail_112;
		else if (dx==1 && dy==2 && dz==0) dst = mail_120;
		else if (dx==1 && dy==2 && dz==1) dst = mail_121;
		else if (dx==1 && dy==2 && dz==2) dst = mail_122;
		else if (dx==2 && dy==0 && dz==0) dst = mail_200;
		else if (dx==2 && dy==0 && dz==1) dst = mail_201;
		else if (dx==2 && dy==0 && dz==2) dst = mail_202;
		else if (dx==2 && dy==1 && dz==0) dst = mail_210;
		else if (dx==2 && dy==1 && dz==1) dst = mail_211;
		else if (dx==2 && dy==1 && dz==2) dst = mail_212;
		else if (dx==2 && dy==2 && dz==0) dst = mail_220;
		else if (dx==2 && dy==2 && dz==1) dst = mail_221;
		else dst = mail_222;

		dst->data_.emplace_back(p);
	}
}

// Identical in shape to pic3d_v2's merge_particles_block: this fragment's
// own survivors (from_self, written by ITS OWN push call) plus whatever
// each of its (up to 26) neighbors routed towards it, each gated by its
// own existsXXX flag (0 at a fragment-grid edge, where pic3d.py-style
// callers pass a harmless self-reference that must never actually be
// read).
//
// IMPORTANT, found while building this reproduction: that "harmless self-
// reference" is NOT actually harmless for a GrowingArray mailbox. Tokens
// that reach the OUTERMOST fragment-grid edge (ii/jj/kk == 0 or == n-1)
// are lost 100% DETERMINISTICALLY - reproducibly, even at OMP_NUM_THREADS=1
// - whenever a token's current fragment is itself at that edge. This is a
// SEPARATE bug from the threaded race this file is meant to reproduce (see
// growing_array_race_test.py's module docstring): it is not timing-
// dependent at all, and was isolated by lowering the per-token velocity
// (VMAX) until no token could reach a grid edge within NUM_TS steps, which
// alone made the deterministic loss disappear while leaving the genuine,
// thread-count-dependent race (reproducible at the real OMP_NUM_THREADS=10
// configuration) intact. Root cause not pursued further - every push_tokens_
// block/merge_particles_block-pattern mailbox GrowingArray is declared with
// nversions=1 (confirmed in the generated .cpp), so the clamped self-
// reference's GetInPtr call still resolves to a live, valid Element (version
// is irrelevant with nversions=1), and the exists flag correctly gates the
// data copy itself - but something about that still-dereferenced, always-
// "exists=0" read appears to corrupt bookkeeping for the GENUINE output at
// the SAME call (see the growing_array_race_test.py history/VMAX comment
// for how this was isolated). Likely an RTS/sgc task-dependency inference
// issue specific to GrowingArray + this clamp idiom, not an error in this
// file's kernels - out of scope to fix here (see pic3d_v2's own docstring
// for why RTS-level fixes are out of this port's scope), but worth flagging
// since it is a DIFFERENT bug from the one this test targets and could
// explain a different class of surprises in future GrowingArray-based ports
// that use the same clamp-at-grid-edge idiom.
template <typename ElementT>
void merge_tokens_block(
	const ElementT* from_self,
	const ElementT* mail000, int exists000, const ElementT* mail001, int exists001, const ElementT* mail002, int exists002,
	const ElementT* mail010, int exists010, const ElementT* mail011, int exists011, const ElementT* mail012, int exists012,
	const ElementT* mail020, int exists020, const ElementT* mail021, int exists021, const ElementT* mail022, int exists022,
	const ElementT* mail100, int exists100, const ElementT* mail101, int exists101, const ElementT* mail102, int exists102,
	const ElementT* mail110, int exists110,                                         const ElementT* mail112, int exists112,
	const ElementT* mail120, int exists120, const ElementT* mail121, int exists121, const ElementT* mail122, int exists122,
	const ElementT* mail200, int exists200, const ElementT* mail201, int exists201, const ElementT* mail202, int exists202,
	const ElementT* mail210, int exists210, const ElementT* mail211, int exists211, const ElementT* mail212, int exists212,
	const ElementT* mail220, int exists220, const ElementT* mail221, int exists221, const ElementT* mail222, int exists222,
	ElementT* cell_new)
{

	cell_new->data_.clear();
	for (const auto& p : from_self->data_) cell_new->data_.emplace_back(p);

	if (exists000) for (const auto& p : mail000->data_) cell_new->data_.emplace_back(p);
	if (exists001) for (const auto& p : mail001->data_) cell_new->data_.emplace_back(p);
	if (exists002) for (const auto& p : mail002->data_) cell_new->data_.emplace_back(p);
	if (exists010) for (const auto& p : mail010->data_) cell_new->data_.emplace_back(p);
	if (exists011) for (const auto& p : mail011->data_) cell_new->data_.emplace_back(p);
	if (exists012) for (const auto& p : mail012->data_) cell_new->data_.emplace_back(p);
	if (exists020) for (const auto& p : mail020->data_) cell_new->data_.emplace_back(p);
	if (exists021) for (const auto& p : mail021->data_) cell_new->data_.emplace_back(p);
	if (exists022) for (const auto& p : mail022->data_) cell_new->data_.emplace_back(p);
	if (exists100) for (const auto& p : mail100->data_) cell_new->data_.emplace_back(p);
	if (exists101) for (const auto& p : mail101->data_) cell_new->data_.emplace_back(p);
	if (exists102) for (const auto& p : mail102->data_) cell_new->data_.emplace_back(p);
	if (exists110) for (const auto& p : mail110->data_) cell_new->data_.emplace_back(p);
	if (exists112) for (const auto& p : mail112->data_) cell_new->data_.emplace_back(p);
	if (exists120) for (const auto& p : mail120->data_) cell_new->data_.emplace_back(p);
	if (exists121) for (const auto& p : mail121->data_) cell_new->data_.emplace_back(p);
	if (exists122) for (const auto& p : mail122->data_) cell_new->data_.emplace_back(p);
	if (exists200) for (const auto& p : mail200->data_) cell_new->data_.emplace_back(p);
	if (exists201) for (const auto& p : mail201->data_) cell_new->data_.emplace_back(p);
	if (exists202) for (const auto& p : mail202->data_) cell_new->data_.emplace_back(p);
	if (exists210) for (const auto& p : mail210->data_) cell_new->data_.emplace_back(p);
	if (exists211) for (const auto& p : mail211->data_) cell_new->data_.emplace_back(p);
	if (exists212) for (const auto& p : mail212->data_) cell_new->data_.emplace_back(p);
	if (exists220) for (const auto& p : mail220->data_) cell_new->data_.emplace_back(p);
	if (exists221) for (const auto& p : mail221->data_) cell_new->data_.emplace_back(p);
	if (exists222) for (const auto& p : mail222->data_) cell_new->data_.emplace_back(p);
}

// Dumps this fragment's final token ids (flat, 4-byte ints, no header) to
// race_ids_<ii>_<jj>_<kk>.bin so growing_array_race_test.py's verify() can
// compare the set of ids actually present in every fragment against its
// independently-computed, deterministic ground truth. One file per
// fragment (never one shared file across fragments) specifically to keep
// this dump itself race-free - any mismatch this uncovers can only come
// from the GrowingArray push/merge exchange above, not from file I/O
// contention.
template <typename ElementT>
void dump_token_ids_block(int ii, int jj, int kk, const ElementT* cell)
{
	std::ofstream of{"race_ids_" + std::to_string(ii) + "_" + std::to_string(jj) + "_" + std::to_string(kk) + ".bin",
	                  std::ios::binary};
	for (const auto& p : cell->data_)
	{
		int id = p.id;
		of.write(reinterpret_cast<const char*>(&id), sizeof(id));
	}
}
