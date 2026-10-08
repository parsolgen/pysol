#pragma once

#include "CgGate/rts.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <mpi.h>

/*=========================================================================
 * pysol reimplementation of pic3d.cpp (3D electrostatic PIC solver).
 *
 * Mesh fields phi/rho/resid2 are "fragmented": represented as blocked
 * pysol Arrays with a 1-node ghost layer per side (see exchange_3d.sol /
 * pysol_update_array, used throughout examples/simple). The Gauss-Seidel
 * /SOR Poisson solve runs per-block and exchanges phi's ghost nodes
 * between sweeps.
 *
 * A particle may be anywhere in the domain, so the electric field and
 * number density it needs are kept as GLOBAL (single-block) pysol Arrays
 * spanning the whole mesh. Particles themselves live in a
 * GrowingArray<Particle> per species (one dynamically growing vector),
 * per the "use GrowingArray to store particles" requirement. Two small
 * "bridge" kernels (distribute_rho_block / compute_efield_block) copy
 * between the global and fragmented representations.
 *
 * Every kernel argument that is a pysol Array gets its own block-size
 * integers (n, m, l) inserted immediately before it by the .sol generator
 * (one triple per array argument, even when two arrays share the same
 * shape) - every signature below is written to match that calling
 * convention exactly.
 *
 * All physics (XtoL, trilinear gather/scatter, wall reflection, SOR
 * update, electric field finite differences) mirrors
 * examples/pic3d/pic3d.cpp; only the simulation size (grid, particle
 * counts, time steps) is reduced so the example compiles and runs
 * quickly.
 *=========================================================================*/

inline int flat3(int i, int j, int k, int m, int l)
{
	return (i * m + j) * l + k;
}

/*=========================================================================
 * Mesh setup
 *=========================================================================*/

// Zero-fills a fragmented phi block (interior + ghost).
void zero_block3d(int n, int m, int l, double* a)
{
	for (int i = 0; i < n * m * l; ++i)
	{
		a[i] = 0.0;
	}
}

// Node volumes on the GLOBAL mesh: full cell volume on interior nodes,
// halved per boundary face. Mirrors computeNodeVolumes().
void compute_node_volumes(double dhx, double dhy, double dhz,
                           int gn, int gm, int gl,
                           double* node_vol)
{
	for (int i = 0; i < gn; ++i)
	{
		for (int j = 0; j < gm; ++j)
		{
			for (int k = 0; k < gl; ++k)
			{
				double V = dhx * dhy * dhz;
				if (i == 0 || i == gn - 1) V *= 0.5;
				if (j == 0 || j == gm - 1) V *= 0.5;
				if (k == 0 || k == gl - 1) V *= 0.5;
				node_vol[flat3(i, j, k, gm, gl)] = V;
			}
		}
	}
}

/*=========================================================================
 * Particle loading / push (GrowingArray<Particle> kernels)
 *=========================================================================*/

// Quiet start: loads a uniform npx*npy*npz macroparticle lattice over box
// [x1, x2) representing number density num_den, with zero initial
// velocity (the electric field is exactly zero before the first Poisson
// solve, so addParticle()'s half-step velocity rewind in the original is
// a no-op here and is omitted). Mirrors loadParticlesBoxQS()+addParticle().
template <typename ElementT>
void load_particles_box_qs(double x1x, double x1y, double x1z,
                            double x2x, double x2y, double x2z,
                            double num_den, int npx, int npy, int npz,
                            ElementT* cell)
{
	using P = typename std::decay_t<decltype(cell->data_)>::value_type;

	double box_vol = (x2x - x1x) * (x2y - x1y) * (x2z - x1z);
	double num_mp_tot = (double)(npx - 1) * (npy - 1) * (npz - 1);
	double num_real = num_den * box_vol;
	double mpw = num_real / num_mp_tot;

	double dlx = (x2x - x1x) / (npx - 1);
	double dly = (x2y - x1y) / (npy - 1);
	double dlz = (x2z - x1z) / (npz - 1);

	for (int i = 0; i < npx; ++i)
	{
		for (int j = 0; j < npy; ++j)
		{
			for (int k = 0; k < npz; ++k)
			{
				double px = x1x + i * dlx;
				double py = x1y + j * dly;
				double pz = x1z + k * dlz;
				if (px == x2x) px -= 1e-4 * dlx;
				if (py == x2y) py -= 1e-4 * dly;
				if (pz == x2z) pz -= 1e-4 * dlz;

				double w = 1.0;
				if (i == 0 || i == npx - 1) w *= 0.5;
				if (j == 0 || j == npy - 1) w *= 0.5;
				if (k == 0 || k == npz - 1) w *= 0.5;

				cell->data_.emplace_back(P{px, py, pz, 0.0, 0.0, 0.0, mpw * w});
			}
		}
	}
}

// Particle push: v += (q/m)E dt, x += v dt, specular reflection on the
// domain walls. E is gathered (trilinear) from the GLOBAL ef_x/ef_y/ef_z
// arrays, since a particle may be anywhere in the domain. Mirrors
// pushParticles() + XtoL() + gatherEF() + gather().
//
// ParSolGen arrays (and GrowingArray cells) are single-assignment: a
// kernel that modifies one must read the OLD version and write a fresh
// NEW version, like calc_main() in examples/simple/ucodes.h (prev -> curr)
// does for a plain Array. Here the NEW cell is rebuilt one particle at a
// time from the OLD cell, each with its position/velocity advanced.
//
// "Fresh" NEW version does not mean "empty": a GrowingArray's backing
// store only has a finite number of physical slots (nversions), so
// array[pv+1,...] can be a slot last used several iterations ago and
// still holding that old call's particles - cell_new is explicitly
// cleared before being rebuilt so leftover data from a previous reuse of
// this slot is never appended to.
template <typename ElementTIn, typename ElementTOut>
void push_particles(double dt, double qm_dt,
                     double x0x, double x0y, double x0z,
                     double xmx, double xmy, double xmz,
                     double dhx, double dhy, double dhz,
                     int gn1, int gm1, int gl1, const double* ef_x,
                     int gn2, int gm2, int gl2, const double* ef_y,
                     int gn3, int gm3, int gl3, const double* ef_z,
                     const ElementTIn* cell_old, ElementTOut* cell_new)
{
	(void)gn1; (void)gn2; (void)gn3; (void)gl3;
	const int gm = gm1, gl = gl1;
	(void)gm2; (void)gl2; (void)gm3;

	using P = typename std::decay_t<decltype(cell_new->data_)>::value_type;

	cell_new->data_.clear();

	auto gather = [&](const double* f, int i, int j, int k, double di, double dj, double dk) -> double {
		return f[flat3(i,     j,     k,     gm, gl)] * (1 - di) * (1 - dj) * (1 - dk) +
		       f[flat3(i + 1, j,     k,     gm, gl)] * (di)     * (1 - dj) * (1 - dk) +
		       f[flat3(i + 1, j + 1, k,     gm, gl)] * (di)     * (dj)     * (1 - dk) +
		       f[flat3(i,     j + 1, k,     gm, gl)] * (1 - di) * (dj)     * (1 - dk) +
		       f[flat3(i,     j,     k + 1, gm, gl)] * (1 - di) * (1 - dj) * (dk) +
		       f[flat3(i + 1, j,     k + 1, gm, gl)] * (di)     * (1 - dj) * (dk) +
		       f[flat3(i + 1, j + 1, k + 1, gm, gl)] * (di)     * (dj)     * (dk) +
		       f[flat3(i,     j + 1, k + 1, gm, gl)] * (1 - di) * (dj)     * (dk);
	};

	for (const auto& old_p : cell_old->data_)
	{
		P p = old_p;

		double lcx = (p.pos_x - x0x) / dhx;
		double lcy = (p.pos_y - x0y) / dhy;
		double lcz = (p.pos_z - x0z) / dhz;
		int i = (int)lcx; double di = lcx - i;
		int j = (int)lcy; double dj = lcy - j;
		int k = (int)lcz; double dk = lcz - k;

		double efx = gather(ef_x, i, j, k, di, dj, dk);
		double efy = gather(ef_y, i, j, k, di, dj, dk);
		double efz = gather(ef_z, i, j, k, di, dj, dk);

		p.vel_x += efx * qm_dt;
		p.vel_y += efy * qm_dt;
		p.vel_z += efz * qm_dt;
		p.pos_x += p.vel_x * dt;
		p.pos_y += p.vel_y * dt;
		p.pos_z += p.vel_z * dt;

		if (p.pos_x < x0x)       { p.pos_x = 2 * x0x - p.pos_x; p.vel_x *= -1.0; }
		else if (p.pos_x >= xmx) { p.pos_x = 2 * xmx - p.pos_x; p.vel_x *= -1.0; }
		if (p.pos_y < x0y)       { p.pos_y = 2 * x0y - p.pos_y; p.vel_y *= -1.0; }
		else if (p.pos_y >= xmy) { p.pos_y = 2 * xmy - p.pos_y; p.vel_y *= -1.0; }
		if (p.pos_z < x0z)       { p.pos_z = 2 * x0z - p.pos_z; p.vel_z *= -1.0; }
		else if (p.pos_z >= xmz) { p.pos_z = 2 * xmz - p.pos_z; p.vel_z *= -1.0; }

		cell_new->data_.emplace_back(p);
	}
}

// Number density: scatter macroparticle weights (trilinear) onto the
// GLOBAL den array, then divide by node volume. Mirrors
// computeNumberDensity() + scatter().
template <typename ElementT>
void compute_density(double x0x, double x0y, double x0z,
                      double dhx, double dhy, double dhz,
                      const ElementT* cell,
                      int gn1, int gm1, int gl1, const double* node_vol,
                      int gn2, int gm2, int gl2, double* den)
{
	(void)gn1; (void)gn2;
	const int gm = gm1, gl = gl1;

	for (int n = 0; n < gn2 * gm2 * gl2; ++n) den[n] = 0.0;

	for (const auto& p : cell->data_)
	{
		double lcx = (p.pos_x - x0x) / dhx;
		double lcy = (p.pos_y - x0y) / dhy;
		double lcz = (p.pos_z - x0z) / dhz;
		int i = (int)lcx; double di = lcx - i;
		int j = (int)lcy; double dj = lcy - j;
		int k = (int)lcz; double dk = lcz - k;
		double v = p.mpw;

		den[flat3(i,     j,     k,     gm, gl)] += v * (1 - di) * (1 - dj) * (1 - dk);
		den[flat3(i + 1, j,     k,     gm, gl)] += v * (di)     * (1 - dj) * (1 - dk);
		den[flat3(i + 1, j + 1, k,     gm, gl)] += v * (di)     * (dj)     * (1 - dk);
		den[flat3(i,     j + 1, k,     gm, gl)] += v * (1 - di) * (dj)     * (1 - dk);
		den[flat3(i,     j,     k + 1, gm, gl)] += v * (1 - di) * (1 - dj) * (dk);
		den[flat3(i + 1, j,     k + 1, gm, gl)] += v * (di)     * (1 - dj) * (dk);
		den[flat3(i + 1, j + 1, k + 1, gm, gl)] += v * (di)     * (dj)     * (dk);
		den[flat3(i,     j + 1, k + 1, gm, gl)] += v * (1 - di) * (dj)     * (dk);
	}

	for (int n = 0; n < gn2 * gm2 * gl2; ++n)
	{
		if (node_vol[n] != 0.0) den[n] /= node_vol[n];
		else den[n] = 0.0;
	}
}

// Writes the current particle count of a species into `count`, an "out"
// scalar kernel argument, for screen diagnostics.
template <typename ElementT>
void count_particles(const ElementT* cell, int& count)
{
	count = (int)cell->data_.size();
}

// Sums every cell of a single-block GLOBAL array (den_ion/den_elec) directly,
// as an "out" scalar kernel argument - sum()/max()/min() go through a
// reduce() that walks the program's main fragmented block distributor, which
// does not match a single-block global array's (1x1x1) grid, so den_ion/
// den_elec are summed with a plain kernel instead, for screen diagnostics.
void sum_block(int gn, int gm, int gl, const double* block, double& result)
{
	double total = 0.0;
	for (int n = 0; n < gn * gm * gl; ++n) total += block[n];
	result = total;
}

/*=========================================================================
 * Charge density / fragmentation bridge
 *=========================================================================*/

// rho = q_ion*den_ion + q_elec*den_elec, on the GLOBAL mesh. Mirrors
// computeChargeDensity().
void compute_charge_density(double q_ion, double q_elec,
                             int gn1, int gm1, int gl1, const double* den_ion,
                             int gn2, int gm2, int gl2, const double* den_elec,
                             int gn3, int gm3, int gl3, double* rho)
{
	(void)gn1; (void)gm1; (void)gl1; (void)gn2; (void)gm2; (void)gl2;
	const int n = gn3 * gm3 * gl3;
	for (int idx = 0; idx < n; ++idx)
	{
		rho[idx] = den_ion[idx] * q_ion + den_elec[idx] * q_elec;
	}
}

// Copies the interior nodes owned by block (ii,jj,kk) out of the GLOBAL
// rho into the block's own fragmented rho array. The block's ghost layer
// is left untouched - the SOR stencil below never reads a neighbor's rho,
// only its own node, so rho needs no ghost exchange.
void distribute_rho_block(int ii, int jj, int kk, int nb, int mb, int lb,
                           int gn, int gm, int gl, const double* rho_global,
                           int bn2, int bm2, int bl2, double* rho_block)
{
	const int bn = bn2 - 2, bm = bm2 - 2, bl = bl2 - 2;
	(void)nb; (void)mb; (void)lb; (void)gn;

	for (int li = 1; li <= bn; ++li)
	{
		for (int lj = 1; lj <= bm; ++lj)
		{
			for (int lk = 1; lk <= bl; ++lk)
			{
				int gi = ii * bn + (li - 1);
				int gj = jj * bm + (lj - 1);
				int gk = kk * bl + (lk - 1);
				rho_block[flat3(li, lj, lk, bm2, bl2)] = rho_global[flat3(gi, gj, gk, gm, gl)];
			}
		}
	}
}

// Copies the interior nodes owned by block (ii,jj,kk) out of the block's
// own fragmented phi into the GLOBAL phi array - the inverse of
// distribute_rho_block() above, used only to reassemble phi for VTK output
// (see write_vti()), since the real fields.vti output is one GLOBAL,
// ghost-free mesh.
void gather_phi_block(int ii, int jj, int kk, int nb, int mb, int lb,
                       int pn, int pm, int pl, const double* phi_block,
                       int gn, int gm, int gl, double* phi_global)
{
	const int bn = pn - 2, bm = pm - 2, bl = pl - 2;
	(void)nb; (void)mb; (void)lb; (void)gn;

	for (int li = 1; li <= bn; ++li)
	{
		for (int lj = 1; lj <= bm; ++lj)
		{
			for (int lk = 1; lk <= bl; ++lk)
			{
				int gi = ii * bn + (li - 1);
				int gj = jj * bm + (lj - 1);
				int gk = kk * bl + (lk - 1);
				phi_global[flat3(gi, gj, gk, gm, gl)] = phi_block[flat3(li, lj, lk, pm, pl)];
			}
		}
	}
}

/*=========================================================================
 * Poisson solver (fragmented, per-block Gauss-Seidel/SOR)
 *=========================================================================*/

// One SOR sweep over the nodes OWNED by block (ii,jj,kk): local indices
// 1..b{n,m,l}, except that a block sitting on a GLOBAL domain edge skips
// its outermost owned row/column/layer - that node is the fixed (phi=0)
// Dirichlet boundary, exactly the node excluded by the original's
// "for (i=1;i<ni-1;i++)" loop. Mirrors solvePotential()'s inner sweep.
//
// ParSolGen Arrays are single-assignment, so this reads the OLD phi block
// and writes a fresh NEW one - like calc_main() in examples/simple/ucodes.h
// (prev -> curr). The untouched nodes (ghost layer and the Dirichlet-fixed
// boundary row, if any) are copied forward from old to new first, exactly
// as calc_main() copies its unchanged boundary rows/columns, then the
// owned interior is overwritten with the updated SOR value.
void sor_sweep_block(int ii, int jj, int kk, int nb, int mb, int lb,
                      double idx2, double idy2, double idz2, double diag,
                      double eps0, double omega,
                      int rn, int rm, int rl, const double* rho_block,
                      int pn_old, int pm_old, int pl_old, const double* phi_old,
                      int pn_new, int pm_new, int pl_new, double* phi_new)
{
	(void)rn; (void)rm; (void)rl; (void)pn_new; (void)pm_new; (void)pl_new;
	const int bn = pn_old - 2, bm = pm_old - 2, bl = pl_old - 2;
	const int si = pm_old * pl_old, sj = pl_old, sk = 1;

	for (int n = 0; n < pn_old * pm_old * pl_old; ++n) phi_new[n] = phi_old[n];

	const int i_lo = (ii == 0) ? 2 : 1, i_hi = (ii == nb - 1) ? bn - 1 : bn;
	const int j_lo = (jj == 0) ? 2 : 1, j_hi = (jj == mb - 1) ? bm - 1 : bm;
	const int k_lo = (kk == 0) ? 2 : 1, k_hi = (kk == lb - 1) ? bl - 1 : bl;

	for (int i = i_lo; i <= i_hi; ++i)
	{
		for (int j = j_lo; j <= j_hi; ++j)
		{
			for (int k = k_lo; k <= k_hi; ++k)
			{
				int n = flat3(i, j, k, pm_old, pl_old);
				double phi_sor = (rho_block[n] / eps0 +
				                  idx2 * (phi_old[n - si] + phi_old[n + si]) +
				                  idy2 * (phi_old[n - sj] + phi_old[n + sj]) +
				                  idz2 * (phi_old[n - sk] + phi_old[n + sk])) / diag;
				phi_new[n] = phi_old[n] + omega * (phi_sor - phi_old[n]);
			}
		}
	}
}

// Residual-squared on the same owned-node range as sor_sweep_block(), zero
// elsewhere (including ghost/boundary nodes) so sum() over the whole
// fragmented array matches the original's L2 norm computed only over
// i=1..ni-2. Mirrors solvePotential()'s residual check.
void residual_block(int ii, int jj, int kk, int nb, int mb, int lb,
                     double idx2, double idy2, double idz2, double diag,
                     double eps0,
                     int rn, int rm, int rl, const double* rho_block,
                     int pn, int pm, int pl, const double* phi_block,
                     int sn, int sm, int sl, double* resid2_block)
{
	(void)rn; (void)rm; (void)rl;
	const int bn = pn - 2, bm = pm - 2, bl = pl - 2;
	const int si = pm * pl, sj = pl, sk = 1;
	zero_block3d(sn, sm, sl, resid2_block);

	const int i_lo = (ii == 0) ? 2 : 1, i_hi = (ii == nb - 1) ? bn - 1 : bn;
	const int j_lo = (jj == 0) ? 2 : 1, j_hi = (jj == mb - 1) ? bm - 1 : bm;
	const int k_lo = (kk == 0) ? 2 : 1, k_hi = (kk == lb - 1) ? bl - 1 : bl;

	for (int i = i_lo; i <= i_hi; ++i)
	{
		for (int j = j_lo; j <= j_hi; ++j)
		{
			for (int k = k_lo; k <= k_hi; ++k)
			{
				int n = flat3(i, j, k, pm, pl);
				double R = -phi_block[n] * diag +
				           rho_block[n] / eps0 +
				           idx2 * (phi_block[n - si] + phi_block[n + si]) +
				           idy2 * (phi_block[n - sj] + phi_block[n + sj]) +
				           idz2 * (phi_block[n - sk] + phi_block[n + sk]);
				resid2_block[n] = R * R;
			}
		}
	}
}

/*=========================================================================
 * Electric field (fragmented phi -> global ef)
 *=========================================================================*/

// E = -grad(phi): central differences using the (ghost-exchanged)
// neighboring block where available, 2nd-order one-sided differences at
// the true global domain edges. Writes directly into the GLOBAL
// ef_x/ef_y/ef_z arrays at each node's global offset. Mirrors
// computeElectricField().
void compute_efield_block(int ii, int jj, int kk,
                           double dhx, double dhy, double dhz,
                           int pn, int pm, int pl, const double* phi_block,
                           int gnx, int gmx, int glx, double* ef_x,
                           int gny, int gmy, int gly, double* ef_y,
                           int gnz, int gmz, int glz, double* ef_z)
{
	(void)gny; (void)gmy; (void)gly; (void)gnz; (void)gmz; (void)glz;
	const int bn = pn - 2, bm = pm - 2, bl = pl - 2;
	const int si = pm * pl, sj = pl, sk = 1;
	const int gn = gnx, gm = gmx, gl = glx;

	for (int li = 1; li <= bn; ++li)
	{
		for (int lj = 1; lj <= bm; ++lj)
		{
			for (int lk = 1; lk <= bl; ++lk)
			{
				int n = flat3(li, lj, lk, pm, pl);
				int gi = ii * bn + (li - 1);
				int gj = jj * bm + (lj - 1);
				int gk = kk * bl + (lk - 1);
				int gidx = flat3(gi, gj, gk, gm, gl);

				double Ex, Ey, Ez;

				if (gi == 0)
					Ex = -(-3 * phi_block[n] + 4 * phi_block[n + si] - phi_block[n + 2 * si]) / (2 * dhx);
				else if (gi == gn - 1)
					Ex = -(phi_block[n - 2 * si] - 4 * phi_block[n - si] + 3 * phi_block[n]) / (2 * dhx);
				else
					Ex = -(phi_block[n + si] - phi_block[n - si]) / (2 * dhx);

				if (gj == 0)
					Ey = -(-3 * phi_block[n] + 4 * phi_block[n + sj] - phi_block[n + 2 * sj]) / (2 * dhy);
				else if (gj == gm - 1)
					Ey = -(phi_block[n - 2 * sj] - 4 * phi_block[n - sj] + 3 * phi_block[n]) / (2 * dhy);
				else
					Ey = -(phi_block[n + sj] - phi_block[n - sj]) / (2 * dhy);

				if (gk == 0)
					Ez = -(-3 * phi_block[n] + 4 * phi_block[n + sk] - phi_block[n + 2 * sk]) / (2 * dhz);
				else if (gk == gl - 1)
					Ez = -(phi_block[n - 2 * sk] - 4 * phi_block[n - sk] + 3 * phi_block[n]) / (2 * dhz);
				else
					Ez = -(phi_block[n + sk] - phi_block[n - sk]) / (2 * dhz);

				ef_x[gidx] = Ex;
				ef_y[gidx] = Ey;
				ef_z[gidx] = Ez;
			}
		}
	}
}

/*=========================================================================
 * Diagnostics
 *=========================================================================*/

void print_step(int ts, int n_ion, int n_elec,
                 double sum_den_ion, double sum_den_elec,
                 int it, double l2resid2, double max_phi, double min_phi)
{
	std::cout << "ts=" << ts << " n_ion=" << n_ion << " n_elec=" << n_elec
	          << " sum(den_ion)=" << sum_den_ion
	          << " sum(den_elec)=" << sum_den_elec
	          << " solver_it=" << it << " L2resid2=" << l2resid2
	          << " max(phi)=" << max_phi << " min(phi)=" << min_phi << "\n";
}

// Writes the GLOBAL mesh fields to "results/fields_NNNNN.vti", mirroring
// Output::fields() in examples/pic3d/Output.cpp exactly: same VTK
// ImageData header (Origin/Spacing/WholeExtent), same PointData DataArray
// tags (NodeVol, phi, rho, per-species number density, then the 3-component
// ef vector), and the same ASCII node order - Field_::operator<< there
// loops "for k { for j { for i } }" (so k is outermost, i fastest in the
// text), which this replicates via flat3(i, j, k, gm, gl) on the row-major
// (i slowest, k fastest in memory) global arrays.
void write_vti(int ts, double x0x, double x0y, double x0z,
                double dhx, double dhy, double dhz,
                int gn1, int gm1, int gl1, const double* node_vol,
                int gn2, int gm2, int gl2, const double* phi,
                int gn3, int gm3, int gl3, const double* rho,
                int gn4, int gm4, int gl4, const double* den_ion,
                int gn5, int gm5, int gl5, const double* den_elec,
                int gn6, int gm6, int gl6, const double* ef_x,
                int gn7, int gm7, int gl7, const double* ef_y,
                int gn8, int gm8, int gl8, const double* ef_z)
{
	(void)gn2; (void)gm2; (void)gl2; (void)gn3; (void)gm3; (void)gl3;
	(void)gn4; (void)gm4; (void)gl4; (void)gn5; (void)gm5; (void)gl5;
	(void)gn6; (void)gm6; (void)gl6; (void)gn7; (void)gm7; (void)gl7;
	(void)gn8; (void)gm8; (void)gl8;
	const int gn = gn1, gm = gm1, gl = gl1;

	if (ts % 100 != 0) {
		return;
	}

	std::filesystem::create_directories("results");

	std::ostringstream name;
	name << "results/fields_" << std::setfill('0') << std::setw(5) << ts << ".vti";

	std::ofstream out(name.str());
	if (!out.is_open())
	{
		std::cerr << "Could not open " << name.str() << "\n";
		return;
	}

	// ImageData is vtk format for structured Cartesian meshes
	out << "<VTKFile type=\"ImageData\">\n";
	out << "<ImageData Origin=\"" << x0x << " " << x0y << " " << x0z << "\" ";
	out << "Spacing=\"" << dhx << " " << dhy << " " << dhz << "\" ";
	out << "WholeExtent=\"0 " << gn - 1 << " 0 " << gm - 1 << " 0 " << gl - 1 << "\">\n";

	// output data stored on nodes (point data)
	out << "<PointData>\n";

	auto write_scalar = [&](const char* field_name, const double* f) {
		out << "<DataArray Name=\"" << field_name
		    << "\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
		for (int k = 0; k < gl; ++k)
		{
			for (int j = 0; j < gm; ++j)
				for (int i = 0; i < gn; ++i)
					out << f[flat3(i, j, k, gm, gl)] << " ";
			out << "\n";
		}
		out << "</DataArray>\n";
	};

	write_scalar("NodeVol", node_vol);
	write_scalar("phi", phi);
	write_scalar("rho", rho);
	write_scalar("nd.ion", den_ion);
	write_scalar("nd.elec", den_elec);

	// electric field, 3 component vector
	out << "<DataArray Name=\"ef\" NumberOfComponents=\"3\" format=\"ascii\" type=\"Float64\">\n";
	for (int k = 0; k < gl; ++k)
	{
		for (int j = 0; j < gm; ++j)
		{
			for (int i = 0; i < gn; ++i)
			{
				int n = flat3(i, j, k, gm, gl);
				out << ef_x[n] << " " << ef_y[n] << " " << ef_z[n] << " ";
			}
		}
		out << "\n";
	}
	out << "</DataArray>\n";

	// close out tags
	out << "</PointData>\n";
	out << "</ImageData>\n";
	out << "</VTKFile>\n";
	out.close();
}
