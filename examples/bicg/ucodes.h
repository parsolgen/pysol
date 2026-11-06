#pragma once

#include "CgGate/rts.h"

#include <mkl_spblas.h>
#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <vector>
#include <optional>
#include <fstream>
#include <iostream>

/*=========================================================================
 * pysol port of ~/work/tests/bicg/bicg.sol: unpreconditioned BiCG (block
 * conjugate-gradient squared's non-stabilized cousin) solving A x = b for
 * a general (non-symmetric) sparse matrix, blocked into an nb x nb grid of
 * CSR sub-blocks via CSRMatrix/CSRBlock.
 *
 * Matrix reading (read_matrix/init_params) and the MKL block sparse
 * matvec/matvec-transpose (csr_mv/csr_mv_t, including their lazy
 * mats/t_mats per-block MKL handle cache) are carried over from the
 * reference ucodes.h unchanged - this is the pre-existing global state the
 * task allows keeping. The GPU/goopax variants of csr_mv in the reference
 * are dropped (no pysol example links a GPU kernel path; the goopax/tbb
 * libraries are still linked per examples/bicg/config.json purely for
 * link-compatibility with PSRts, matching every other pysol example).
 *
 * Dot products ("double beta = (Zr, Rl);" in bicg.sol) use pysol's `@`
 * operator (a @ b -> (a, b) in the generated .sol), so no dot-product
 * kernel is needed here. axpby_block is the one kernel this port adds
 * beyond the reference's set, needed because pysol still has no
 * array-level "x + a*y" axpy sugar - every axpy/scale vector operation
 * goes through this explicit per-block kernel call instead, the same way
 * every other pysol example (pic3d, lattice-gas) does its array math.
 *=========================================================================*/

std::vector<std::optional<sparse_matrix_t>> mats;
std::vector<std::optional<sparse_matrix_t>> t_mats;

int mat_n = -1;
int mat_nnz = -1;

int* mat_ia = nullptr;
int* mat_ja = nullptr;
double* mat_a = nullptr;

double t1 = 0.;

void init_mat_desc(int nb)
{
	mats.resize(nb * nb);
	t_mats.resize(nb * nb);
}

void read_matrix(const std::string& file_name)
{
	std::ifstream ifs{file_name, std::ios::binary};
	if (!ifs.is_open()) {
		std::cerr << "Error opening matrix file\n";
		rts::abort();
		return;
	}
	ifs.read(reinterpret_cast<char*>(&mat_n), sizeof(int));
	ifs.read(reinterpret_cast<char*>(&mat_nnz), sizeof(int));

	mat_ia = new int[mat_n + 1];
	mat_ja = new int[mat_nnz];
	mat_a = new double[mat_nnz];

	ifs.read(reinterpret_cast<char*>(mat_ia), sizeof(int) * (mat_n + 1));
	ifs.read(reinterpret_cast<char*>(mat_ja), sizeof(int) * (mat_nnz));
	ifs.read(reinterpret_cast<char*>(mat_a), sizeof(double) * (mat_nnz));
}

void init_params(int nb, int& n, int& nnz, int& blockn)
{
	const auto args = rts::get_command_line_args();
	if (args.size() < 2) {
		std::cerr << "Bad arguments: expected a matrix file path\n";
		rts::abort();
		return;
	}

	const auto file_path = args[1];
	read_matrix(file_path);

	if (mat_nnz == -1 || mat_n == -1) {
		std::cerr << "Incorrect matrix file\n";
		rts::abort();
		return;
	}

	n = mat_n;
	nnz = mat_nnz;
	blockn = n / nb;
}

// ia/ja/a are declared in bicg.py as Array(1, 0, [...], [], Type.*) - empty
// block dims, which sgc compiles to rts::ContiguousArray1{I,D} rather than
// the block-holder classes other arrays here use (needed because this RTS
// build has no declare_vector_i: Array<int> with a non-empty block
// dimension would try to use it and fail to compile).
void init_mat_arrays(int n, int nnz, int ia_n1, int* ia, int ja_n1, int* ja, int a_n1, double* a)
{
	std::copy(mat_ia, mat_ia + n + 1, ia);
	std::copy(mat_ja, mat_ja + nnz, ja);
	std::copy(mat_a, mat_a + nnz, a);
}

void print(int value)
{
	std::cout << value << "\n";
}

void print_d(double value)
{
	std::cout << "d = " << value << "\n";
}

void init_vec(int iblock, int blockn, int block_n1, double* block)
{
	for (int i = 0; i < blockn; ++i) {
		block[i] = 1;
	}
}

void init_zero_vec(int iblock, int blockn, int block_n1, double* block)
{
	for (int i = 0; i < blockn; ++i) {
		block[i] = 0;
	}
}

void csr_mv(
	int nb,
	int blockn,
	const rts::CSRMatrixBlock<int, double>& block,
	int x_n1,
	const double* x,
	int ysrc_n1,
	const double* ysrc,
	int ydst_n1,
	double* ydst)
{
	if (!mats[nb * block.i + block.j]) {
		sparse_matrix_t mat_block_desc;
		mkl_sparse_d_create_csr(
			&mat_block_desc,
			SPARSE_INDEX_BASE_ZERO,
			block.n,
			block.m,
			block.ia,
			&block.ia[1],
			block.ja,
			block.a);
		mats[nb * block.i + block.j] = std::move(mat_block_desc);
	}
	matrix_descr desc;
	desc.type = SPARSE_MATRIX_TYPE_GENERAL;
	desc.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., mats[nb * block.i + block.j].value(), desc,
		x, 1., ydst);
}

void csr_mv_t(
	int nb,
	int blockn,
	const rts::CSRMatrixBlock<int, double>& block,
	int x_n1,
	const double* x,
	int ysrc_n1,
	const double* ysrc,
	int ydst_n1,
	double* ydst)
{
	if (!t_mats[nb * block.i + block.j]) {
		sparse_matrix_t mat_block_desc;
		mkl_sparse_d_create_csr(
			&mat_block_desc,
			SPARSE_INDEX_BASE_ZERO,
			block.n,
			block.m,
			block.ia,
			&block.ia[1],
			block.ja,
			block.a);
		t_mats[nb * block.i + block.j] = std::move(mat_block_desc);
	}
	matrix_descr desc;
	desc.type = SPARSE_MATRIX_TYPE_GENERAL;
	desc.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_mv(SPARSE_OPERATION_CONJUGATE_TRANSPOSE, 1., t_mats[nb * block.i + block.j].value(), desc,
		x, 1., ydst);
}

// dst[k] = a*x[k] + b*y[k] - covers both a pure scale (b=0) and an axpy
// (a=1) with one kernel. Named "dst", not "out" - "out" is a .sol
// reserved keyword (the output-argument direction marker).
void axpby_block(int blockn, double a, int x_n1, const double* x, double b, int y_n1, const double* y,
                  int dst_n1, double* dst)
{
	for (int i = 0; i < blockn; ++i) {
		dst[i] = a * x[i] + b * y[i];
	}
}

void sqrt(double a, double& b)
{
	b = std::sqrt(a);
}

void time_start()
{
	t1 = MPI_Wtime();
}

void print_time()
{
	std::cout << "Time taken: " << MPI_Wtime() - t1 << "\n";
}
