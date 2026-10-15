/*=============================================================================
 * Kernels for pic3d_v2: pysol port of pic3d_common's self-gravitating N-body
 * PIC ("PIC3D" here models gravity, not electrostatics - see pic3d_common's
 * mesh3d.h/pic3d.cpp: RHS is 4*PI*rho, G=1, a "sun" point mass sits at a
 * (here static) center, particles reflect off the domain walls).
 *
 * Ported from pic3d_common/luna5/pic3d_luna5.{fa,cpp} and
 * pic3d_common/common/{pic3d.h,mesh3d.h,pic3d.cpp} - see pic3d.py's module
 * docstring for the full list of deliberate simplifications (zero initial
 * velocity instead of the reference's circular-orbit quiet start; a
 * lattice-in-disk instead of the reference's ring/bisection particle
 * loader; a reduced 13->2 diagnostics vector; a static center since
 * vx=vy=vz=0 in the reference's own default pic3d.par anyway).
 *
 * EVERYTHING is fragmented into an nfx x nfy x nfz fragment grid, exactly
 * matching pic3d_common's own domain decomposition, with NO global,
 * non-fragmented array or static C++ buffer anywhere - this is what makes
 * it extend to more than one MPI process. Three things carry genuinely
 * fragmented, cross-fragment data, and each is exchanged with the SAME 26
 * (face+edge+corner) neighbor pattern pic3d_common itself uses (not just
 * the 6-face pattern pysol_update_array provides - that's too coarse for
 * this algorithm: a particle or a unit of scattered mass can sit exactly
 * on a fragment-grid edge or corner, which only a 26-neighbor exchange
 * reaches):
 *   - "ro" (mass density): compute_density_raw_block scatters each
 *     fragment's own particles (trilinear/CIC, cell-centered) into a
 *     ghosted raw buffer; exchange_density_additive_block then does the
 *     reference's additive 26-neighbor merge (mesh3d_borders_sum_26_*).
 *   - "fp" (particle-sourced potential): poisson_jacobi_sweep_block does
 *     one weighted-Jacobi sweep (plain Jacobi, w=1 in the reference's own
 *     default pic3d.par - no SOR/Gauss-Seidel approximation needed here,
 *     unlike pysol's other pic3d port); exchange_potential_copy_block
 *     does the reference's copy 26-neighbor ghost refresh
 *     (mesh3d_borders_26_*), run after every sweep.
 *   - particles: push_particles_block routes each particle into one of
 *     27 mailboxes (self + 26 directions) based on which side of its OWN
 *     fragment's owned box it ends up on after one kick-drift step (this
 *     reference's combined/simultaneous 3-axis direction code, not an
 *     axis-priority scheme - a particle that crosses two or three fragment
 *     faces at once is still routed correctly in a single hop);
 *     merge_particles_block is each fragment's sole producer of its own
 *     next-version particle list, combining its own survivors with
 *     whatever its (up to 26) neighbors routed towards it - mirroring
 *     particles_move_sort/particles_gather via the same mailbox idea
 *     pysol's other pic3d port already uses for its own (7-way) particle
 *     exchange, just widened to the full 26+1 directions this reference
 *     actually needs.
 *
 * Direction naming (matches pic3d_common's own bXYZ convention exactly):
 * a 3-digit suffix XYZ, one digit per axis (x,y,z), each 0=low neighbor
 * (-1), 1=self (0), 2=high neighbor (+1); 111 (self) is never one of the
 * 26 - it is handled separately (unconditionally, no "exists" flag needed,
 * since a fragment is always its own self-neighbor).
 *
 * "exists" flags: since a fragment grid has real boundaries, a given
 * direction's neighbor fragment does not always exist (e.g. direction 000
 * needs BOTH a lower-x AND a lower-y AND a lower-z neighbor to all exist
 * simultaneously - if any one of those axes is already at the edge of the
 * fragment grid, there is no such diagonal fragment at all). Each of the
 * three exchange kernels below takes one "existsXYZ" int per direction;
 * pic3d.py computes these once per fragment from simple has_lo_x/
 * has_hi_x/... flags (1 when that axis has a real neighbor, 0 at a
 * fragment-grid edge) and passes a harmless, always-valid fallback array
 * reference (the fragment's own data) for directions that don't exist -
 * every kernel below guards each direction's body with `if (existsXYZ)`,
 * so that fallback reference is never actually dereferenced.
 *=============================================================================*/

#pragma once

#include "CgGate/rts.h"

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

inline int flat3(int i, int j, int k, int m, int l)
{
	return (i * m + j) * l + k;
}

const double GRAV_PI = 3.1415926535897932384626433832795;

/*=========================================================================
 * Basic fragment-local helpers
 *=========================================================================*/

void zero_block3d(int n, int m, int l, double* a)
{
	for (int i = 0; i < n * m * l; ++i) a[i] = 0.0;
}

void copy_block3d(int n, int m, int l, const double* a, double* b)
{
	for (int i = 0; i < n * m * l; ++i) b[i] = a[i];
}

/*=========================================================================
 * Mass deposition (cloud-in-cell, cell-centered) - mirrors density() in
 * pic3d_common/common/pic3d.cpp, scaled per-particle by am/(hx*hy*hz)
 * instead of once at the end (equivalent, since the scatter is linear).
 * den_raw has a 1-cell ghost per side (dn = bn+2, etc, where bn is this
 * fragment's own interior cell count per axis); any particle's trilinear
 * footprint that spills past the fragment's own interior lands in that
 * ghost margin, for exchange_density_additive_block to pick up.
 *=========================================================================*/
template <typename ElementT>
void compute_density_raw_block(double hx, double hy, double hz, double am,
                                int ii, int jj, int kk, int bn, int bm, int bl,
                                const ElementT* cell,
                                int dn, int dm, int dl, double* den_raw)
{
	for (int n = 0; n < dn * dm * dl; ++n) den_raw[n] = 0.0;
	const double scale = am / (hx * hy * hz);

	for (const auto& p : cell->data_)
	{
		double rx = p.x / hx - 0.5;
		double ry = p.y / hy - 0.5;
		double rz = p.z / hz - 0.5;
		int gi = (int)std::floor(rx); double di = rx - gi;
		int gj = (int)std::floor(ry); double dj = ry - gj;
		int gk = (int)std::floor(rz); double dk = rz - gk;

		int li = gi - ii * bn + 1;
		int lj = gj - jj * bm + 1;
		int lk = gk - kk * bl + 1;
		if (li < 1) li = 1; else if (li > bn) li = bn;
		if (lj < 1) lj = 1; else if (lj > bm) lj = bm;
		if (lk < 1) lk = 1; else if (lk > bl) lk = bl;

		double v = scale;
		den_raw[flat3(li,     lj,     lk,     dm, dl)] += v * (1 - di) * (1 - dj) * (1 - dk);
		den_raw[flat3(li + 1, lj,     lk,     dm, dl)] += v * (di)     * (1 - dj) * (1 - dk);
		den_raw[flat3(li + 1, lj + 1, lk,     dm, dl)] += v * (di)     * (dj)     * (1 - dk);
		den_raw[flat3(li,     lj + 1, lk,     dm, dl)] += v * (1 - di) * (dj)     * (1 - dk);
		den_raw[flat3(li,     lj,     lk + 1, dm, dl)] += v * (1 - di) * (1 - dj) * (dk);
		den_raw[flat3(li + 1, lj,     lk + 1, dm, dl)] += v * (di)     * (1 - dj) * (dk);
		den_raw[flat3(li + 1, lj + 1, lk + 1, dm, dl)] += v * (di)     * (dj)     * (dk);
		den_raw[flat3(li,     lj + 1, lk + 1, dm, dl)] += v * (1 - di) * (dj)     * (dk);
	}
}

/*=========================================================================
 * exchange_density_additive_block: the reference's borders_exchange_sum_26
 * (mesh3d_borders_sum_26_get/_set), done directly fragment-to-fragment
 * instead of via an intermediate send/receive buffer - every one of the
 * 26 neighbor overlap regions is independently ADDED (not just copied):
 * a node physically shared by several fragments (2 across a face, 4 along
 * an edge, 8 at a corner) must sum every one of their independent raw
 * scatters, which is exactly what looping over all 26 directions
 * (self-raw baseline + up to 26 overlapping neighbor regions) achieves,
 * even though the write ranges below overlap each other on purpose.
 *=========================================================================*/
void exchange_density_additive_block(
				int sn, int sm, int sl, const double* self_raw,
				int n000, int m000, int l000, const double* nbr000, int exists000,
				int n001, int m001, int l001, const double* nbr001, int exists001,
				int n002, int m002, int l002, const double* nbr002, int exists002,
				int n010, int m010, int l010, const double* nbr010, int exists010,
				int n011, int m011, int l011, const double* nbr011, int exists011,
				int n012, int m012, int l012, const double* nbr012, int exists012,
				int n020, int m020, int l020, const double* nbr020, int exists020,
				int n021, int m021, int l021, const double* nbr021, int exists021,
				int n022, int m022, int l022, const double* nbr022, int exists022,
				int n100, int m100, int l100, const double* nbr100, int exists100,
				int n101, int m101, int l101, const double* nbr101, int exists101,
				int n102, int m102, int l102, const double* nbr102, int exists102,
				int n110, int m110, int l110, const double* nbr110, int exists110,
				int n112, int m112, int l112, const double* nbr112, int exists112,
				int n120, int m120, int l120, const double* nbr120, int exists120,
				int n121, int m121, int l121, const double* nbr121, int exists121,
				int n122, int m122, int l122, const double* nbr122, int exists122,
				int n200, int m200, int l200, const double* nbr200, int exists200,
				int n201, int m201, int l201, const double* nbr201, int exists201,
				int n202, int m202, int l202, const double* nbr202, int exists202,
				int n210, int m210, int l210, const double* nbr210, int exists210,
				int n211, int m211, int l211, const double* nbr211, int exists211,
				int n212, int m212, int l212, const double* nbr212, int exists212,
				int n220, int m220, int l220, const double* nbr220, int exists220,
				int n221, int m221, int l221, const double* nbr221, int exists221,
				int n222, int m222, int l222, const double* nbr222, int exists222,
				int n, int m, int l, double* out)
{
	(void)sn; (void)sm; (void)sl;
	(void)n000; (void)m000; (void)l000; (void)n001; (void)m001; (void)l001; (void)n002; (void)m002; (void)l002;
	(void)n010; (void)m010; (void)l010; (void)n011; (void)m011; (void)l011; (void)n012; (void)m012; (void)l012;
	(void)n020; (void)m020; (void)l020; (void)n021; (void)m021; (void)l021; (void)n022; (void)m022; (void)l022;
	(void)n100; (void)m100; (void)l100; (void)n101; (void)m101; (void)l101; (void)n102; (void)m102; (void)l102;
	(void)n110; (void)m110; (void)l110; (void)n112; (void)m112; (void)l112;
	(void)n120; (void)m120; (void)l120; (void)n121; (void)m121; (void)l121; (void)n122; (void)m122; (void)l122;
	(void)n200; (void)m200; (void)l200; (void)n201; (void)m201; (void)l201; (void)n202; (void)m202; (void)l202;
	(void)n210; (void)m210; (void)l210; (void)n211; (void)m211; (void)l211; (void)n212; (void)m212; (void)l212;
	(void)n220; (void)m220; (void)l220; (void)n221; (void)m221; (void)l221; (void)n222; (void)m222; (void)l222;

	for (int i = 0; i < n * m * l; ++i) out[i] = self_raw[i];

	if (exists000) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr000[flat3(li+(n-2),lj+(m-2),lk+(l-2),m,l)];
	}
	if (exists001) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr001[flat3(li+(n-2),lj+(m-2),lk,m,l)];
	}
	if (exists002) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr002[flat3(li+(n-2),lj+(m-2),lk-(l-2),m,l)];
	}
	if (exists010) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr010[flat3(li+(n-2),lj,lk+(l-2),m,l)];
	}
	if (exists011) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr011[flat3(li+(n-2),lj,lk,m,l)];
	}
	if (exists012) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr012[flat3(li+(n-2),lj,lk-(l-2),m,l)];
	}
	if (exists020) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr020[flat3(li+(n-2),lj-(m-2),lk+(l-2),m,l)];
	}
	if (exists021) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr021[flat3(li+(n-2),lj-(m-2),lk,m,l)];
	}
	if (exists022) {
		for (int li = 0; li <= 1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr022[flat3(li+(n-2),lj-(m-2),lk-(l-2),m,l)];
	}
	if (exists100) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr100[flat3(li,lj+(m-2),lk+(l-2),m,l)];
	}
	if (exists101) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr101[flat3(li,lj+(m-2),lk,m,l)];
	}
	if (exists102) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr102[flat3(li,lj+(m-2),lk-(l-2),m,l)];
	}
	if (exists110) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr110[flat3(li,lj,lk+(l-2),m,l)];
	}
	if (exists112) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr112[flat3(li,lj,lk-(l-2),m,l)];
	}
	if (exists120) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr120[flat3(li,lj-(m-2),lk+(l-2),m,l)];
	}
	if (exists121) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr121[flat3(li,lj-(m-2),lk,m,l)];
	}
	if (exists122) {
		for (int li = 0; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr122[flat3(li,lj-(m-2),lk-(l-2),m,l)];
	}
	if (exists200) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr200[flat3(li-(n-2),lj+(m-2),lk+(l-2),m,l)];
	}
	if (exists201) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr201[flat3(li-(n-2),lj+(m-2),lk,m,l)];
	}
	if (exists202) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= 1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr202[flat3(li-(n-2),lj+(m-2),lk-(l-2),m,l)];
	}
	if (exists210) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr210[flat3(li-(n-2),lj,lk+(l-2),m,l)];
	}
	if (exists211) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr211[flat3(li-(n-2),lj,lk,m,l)];
	}
	if (exists212) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = 0; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr212[flat3(li-(n-2),lj,lk-(l-2),m,l)];
	}
	if (exists220) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr220[flat3(li-(n-2),lj-(m-2),lk+(l-2),m,l)];
	}
	if (exists221) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = 0; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr221[flat3(li-(n-2),lj-(m-2),lk,m,l)];
	}
	if (exists222) {
		for (int li = n-2; li <= n-1; ++li)
		for (int lj = m-2; lj <= m-1; ++lj)
		for (int lk = l-2; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] += nbr222[flat3(li-(n-2),lj-(m-2),lk-(l-2),m,l)];
	}
}

/*=========================================================================
 * exchange_potential_copy_block: the reference's borders_exchange_26
 * (mesh3d_borders_26_get/_set). Copy (not additive) semantics: baseline
 * is a full self-copy (which is what PRESERVES a global-boundary ghost's
 * Dirichlet value when the matching direction doesn't exist - see
 * init_boundary_potential_block), then each EXISTING direction overwrites
 * just its own 1-cell ghost slab with its neighbor's first interior
 * layer. Free-axis ranges are restricted to [1,n-2] (excluding the ghost
 * endpoints) exactly like the reference, so face writes can never clobber
 * an edge or corner ghost that only a diagonal neighbor can supply.
 *=========================================================================*/
void exchange_potential_copy_block(
				int sn, int sm, int sl, const double* self_fi,
				int n000, int m000, int l000, const double* nbr000, int exists000,
				int n001, int m001, int l001, const double* nbr001, int exists001,
				int n002, int m002, int l002, const double* nbr002, int exists002,
				int n010, int m010, int l010, const double* nbr010, int exists010,
				int n011, int m011, int l011, const double* nbr011, int exists011,
				int n012, int m012, int l012, const double* nbr012, int exists012,
				int n020, int m020, int l020, const double* nbr020, int exists020,
				int n021, int m021, int l021, const double* nbr021, int exists021,
				int n022, int m022, int l022, const double* nbr022, int exists022,
				int n100, int m100, int l100, const double* nbr100, int exists100,
				int n101, int m101, int l101, const double* nbr101, int exists101,
				int n102, int m102, int l102, const double* nbr102, int exists102,
				int n110, int m110, int l110, const double* nbr110, int exists110,
				int n112, int m112, int l112, const double* nbr112, int exists112,
				int n120, int m120, int l120, const double* nbr120, int exists120,
				int n121, int m121, int l121, const double* nbr121, int exists121,
				int n122, int m122, int l122, const double* nbr122, int exists122,
				int n200, int m200, int l200, const double* nbr200, int exists200,
				int n201, int m201, int l201, const double* nbr201, int exists201,
				int n202, int m202, int l202, const double* nbr202, int exists202,
				int n210, int m210, int l210, const double* nbr210, int exists210,
				int n211, int m211, int l211, const double* nbr211, int exists211,
				int n212, int m212, int l212, const double* nbr212, int exists212,
				int n220, int m220, int l220, const double* nbr220, int exists220,
				int n221, int m221, int l221, const double* nbr221, int exists221,
				int n222, int m222, int l222, const double* nbr222, int exists222,
				int n, int m, int l, double* out)
{
	(void)sn; (void)sm; (void)sl;
	(void)n000; (void)m000; (void)l000; (void)n001; (void)m001; (void)l001; (void)n002; (void)m002; (void)l002;
	(void)n010; (void)m010; (void)l010; (void)n011; (void)m011; (void)l011; (void)n012; (void)m012; (void)l012;
	(void)n020; (void)m020; (void)l020; (void)n021; (void)m021; (void)l021; (void)n022; (void)m022; (void)l022;
	(void)n100; (void)m100; (void)l100; (void)n101; (void)m101; (void)l101; (void)n102; (void)m102; (void)l102;
	(void)n110; (void)m110; (void)l110; (void)n112; (void)m112; (void)l112;
	(void)n120; (void)m120; (void)l120; (void)n121; (void)m121; (void)l121; (void)n122; (void)m122; (void)l122;
	(void)n200; (void)m200; (void)l200; (void)n201; (void)m201; (void)l201; (void)n202; (void)m202; (void)l202;
	(void)n210; (void)m210; (void)l210; (void)n211; (void)m211; (void)l211; (void)n212; (void)m212; (void)l212;
	(void)n220; (void)m220; (void)l220; (void)n221; (void)m221; (void)l221; (void)n222; (void)m222; (void)l222;

	for (int i = 0; i < n * m * l; ++i) out[i] = self_fi[i];

	if (exists000) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr000[flat3(li+1,lj+1,lk+1,m,l)];
	}
	if (exists001) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr001[flat3(li+1,lj+1,lk,m,l)];
	}
	if (exists002) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr002[flat3(li+1,lj+1,lk-1,m,l)];
	}
	if (exists010) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr010[flat3(li+1,lj,lk+1,m,l)];
	}
	if (exists011) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr011[flat3(li+1,lj,lk,m,l)];
	}
	if (exists012) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr012[flat3(li+1,lj,lk-1,m,l)];
	}
	if (exists020) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr020[flat3(li+1,lj-1,lk+1,m,l)];
	}
	if (exists021) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr021[flat3(li+1,lj-1,lk,m,l)];
	}
	if (exists022) {
		for (int li = 0; li <= 0; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr022[flat3(li+1,lj-1,lk-1,m,l)];
	}
	if (exists100) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr100[flat3(li,lj+1,lk+1,m,l)];
	}
	if (exists101) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr101[flat3(li,lj+1,lk,m,l)];
	}
	if (exists102) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr102[flat3(li,lj+1,lk-1,m,l)];
	}
	if (exists110) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr110[flat3(li,lj,lk+1,m,l)];
	}
	if (exists112) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr112[flat3(li,lj,lk-1,m,l)];
	}
	if (exists120) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr120[flat3(li,lj-1,lk+1,m,l)];
	}
	if (exists121) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr121[flat3(li,lj-1,lk,m,l)];
	}
	if (exists122) {
		for (int li = 1; li <= n-2; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr122[flat3(li,lj-1,lk-1,m,l)];
	}
	if (exists200) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr200[flat3(li-1,lj+1,lk+1,m,l)];
	}
	if (exists201) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr201[flat3(li-1,lj+1,lk,m,l)];
	}
	if (exists202) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 0; lj <= 0; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr202[flat3(li-1,lj+1,lk-1,m,l)];
	}
	if (exists210) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr210[flat3(li-1,lj,lk+1,m,l)];
	}
	if (exists211) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr211[flat3(li-1,lj,lk,m,l)];
	}
	if (exists212) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = 1; lj <= m-2; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr212[flat3(li-1,lj,lk-1,m,l)];
	}
	if (exists220) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 0; lk <= 0; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr220[flat3(li-1,lj-1,lk+1,m,l)];
	}
	if (exists221) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = 1; lk <= l-2; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr221[flat3(li-1,lj-1,lk,m,l)];
	}
	if (exists222) {
		for (int li = n-1; li <= n-1; ++li)
		for (int lj = m-1; lj <= m-1; ++lj)
		for (int lk = l-1; lk <= l-1; ++lk)
			out[flat3(li,lj,lk,m,l)] = nbr222[flat3(li-1,lj-1,lk-1,m,l)];
	}
}

/*=========================================================================
 * Poisson solve (gravity: Laplacian(fi) = 4*PI*ro, G=1) and the sun
 *=========================================================================*/

// Sets the GLOBAL-boundary ghost faces to the point-mass Dirichlet value
// -pm/r (r from the domain center, same "cx,cy,cz" the sun sits at -
// vx=vy=vz=0 in the reference's own default pic3d.par, so the center
// never moves and this only needs to run once, at setup); every other
// cell (interior, and any ghost that borders a REAL neighbor fragment
// instead of the domain edge) is zeroed, matching poisson_init(0.0) -
// exchange_potential_copy_block fills the internal ghosts for real on the
// very first call, and preserves these Dirichlet values everywhere else.
void init_boundary_potential_block(int has_lo_x, int has_hi_x, int has_lo_y, int has_hi_y, int has_lo_z, int has_hi_z,
                                    double hx, double hy, double hz, double cx, double cy, double cz, double pm,
                                    int ii, int jj, int kk, int bn, int bm, int bl,
                                    int n, int m, int l, double* fi)
{
	for (int i = 0; i < n * m * l; ++i) fi[i] = 0.0;

	auto set_face = [&](bool present, int li_fixed, bool fix_i, int lj_fixed, bool fix_j, int lk_fixed, bool fix_k) {
		if (present) return;
		for (int li = fix_i ? li_fixed : 0; li <= (fix_i ? li_fixed : n - 1); ++li)
		for (int lj = fix_j ? lj_fixed : 0; lj <= (fix_j ? lj_fixed : m - 1); ++lj)
		for (int lk = fix_k ? lk_fixed : 0; lk <= (fix_k ? lk_fixed : l - 1); ++lk)
		{
			double gx = fix_i ? (li == 0 ? -0.5 * hx : (ii * bn + bn - 1 + 0.5) * hx) : (ii * bn + (li - 1) + 0.5) * hx;
			double gy = fix_j ? (lj == 0 ? -0.5 * hy : (jj * bm + bm - 1 + 0.5) * hy) : (jj * bm + (lj - 1) + 0.5) * hy;
			double gz = fix_k ? (lk == 0 ? -0.5 * hz : (kk * bl + bl - 1 + 0.5) * hz) : (kk * bl + (lk - 1) + 0.5) * hz;
			double dx = gx - cx, dy = gy - cy, dz = gz - cz;
			double r = std::sqrt(dx * dx + dy * dy + dz * dz);
			if (r > 0.0) fi[flat3(li, lj, lk, m, l)] = -pm / r;
		}
	};

	set_face(has_lo_x != 0, 0,     true,  0, false, 0, false);
	set_face(has_hi_x != 0, n - 1, true,  0, false, 0, false);
	set_face(has_lo_y != 0, 0,     false, 0, true,  0, false);
	set_face(has_hi_y != 0, 0,     false, m - 1, true, 0, false);
	set_face(has_lo_z != 0, 0,     false, 0, false, 0, true);
	set_face(has_hi_z != 0, 0,     false, 0, false, l - 1, true);
}

// One weighted-Jacobi sweep, mirrors poisson3d_step_jacobi exactly (the
// reference's own default w=1.0, i.e. plain Jacobi, not SOR - no
// Gauss-Seidel approximation is needed the way pysol's other pic3d port
// needed one, since the reference ITSELF already uses a single-assignment-
// friendly scheme). Baseline-copies the WHOLE block first (so the ghost -
// whatever exchange_potential_copy_block or init_boundary_potential_block
// last put there - carries forward untouched), then updates every
// interior cell unconditionally; the global-boundary Dirichlet values
// are never touched by this loop at all (it only ever writes indices
// [1, n-2]), so they only need to be set once, at init.
void poisson_jacobi_sweep_block(double hx, double hy, double hz, double w,
                                 int rn, int rm, int rl, const double* ro,
                                 int pn, int pm, int pl, const double* fi_old,
                                 int n, int m, int l, double* fi_new)
{
	(void)rn; (void)rm; (void)rl; (void)pn; (void)pm; (void)pl;
	for (int i = 0; i < n * m * l; ++i) fi_new[i] = fi_old[i];

	double rhx2 = 1.0 / (hx * hx), rhy2 = 1.0 / (hy * hy), rhz2 = 1.0 / (hz * hz);
	double a = 0.5 * w / (rhx2 + rhy2 + rhz2);

	for (int i = 1; i < n - 1; ++i)
	for (int j = 1; j < m - 1; ++j)
	for (int k = 1; k < l - 1; ++k)
	{
		double old_v = fi_old[flat3(i, j, k, m, l)];
		double s = a * ((fi_old[flat3(i - 1, j, k, m, l)] + fi_old[flat3(i + 1, j, k, m, l)]) * rhx2
		              + (fi_old[flat3(i, j - 1, k, m, l)] + fi_old[flat3(i, j + 1, k, m, l)]) * rhy2
		              + (fi_old[flat3(i, j, k - 1, m, l)] + fi_old[flat3(i, j, k + 1, m, l)]) * rhz2
		              - ro[flat3(i, j, k, m, l)] * 4.0 * GRAV_PI)
		        + (1.0 - w) * old_v;
		fi_new[flat3(i, j, k, m, l)] = s;
	}
}

// fi_dobavka ("dobavka" = addition): adds the sun's point-mass potential
// -sm/r at the (static) center, pointwise, ghosts included - matches the
// reference exactly, just without the center-array indirection (vx=vy=
// vz=0 means the center never moves, so cx,cy,cz are passed as plain
// scalars - see pic3d.py).
void add_sun_potential_block(double hx, double hy, double hz, double cx, double cy, double cz, double sm,
                              int ii, int jj, int kk, int bn, int bm, int bl,
                              int pn, int pm, int pl, const double* fp,
                              int n, int m, int l, double* fi)
{
	(void)pn; (void)pm; (void)pl;
	for (int li = 0; li < n; ++li)
	for (int lj = 0; lj < m; ++lj)
	for (int lk = 0; lk < l; ++lk)
	{
		double gx = (ii * bn + (li - 1) + 0.5) * hx;
		double gy = (jj * bm + (lj - 1) + 0.5) * hy;
		double gz = (kk * bl + (lk - 1) + 0.5) * hz;
		double dx = gx - cx, dy = gy - cy, dz = gz - cz;
		double r = std::sqrt(dx * dx + dy * dy + dz * dz);
		double sun = (r > 0.0 && sm != 0.0) ? (-sm / r) : 0.0;
		fi[flat3(li, lj, lk, m, l)] = fp[flat3(li, lj, lk, pm, pl)] + sun;
	}
}

// Staggered backward difference, F = -grad(fi) (acceleration, since mass
// cancels for gravity) - matches force() exactly; index 0 along each
// component's own differencing axis is clamped to itself (giving 0
// there) instead of the reference's "leave unset", since every pysol
// array element needs exactly one producer.
void compute_force_block(double hx, double hy, double hz,
                          int pn, int pm, int pl, const double* fi,
                          int n, int m, int l, double* fx,
                          int n2, int m2, int l2, double* fy,
                          int n3, int m3, int l3, double* fz)
{
	(void)pn; (void)pm; (void)pl; (void)n2; (void)m2; (void)l2; (void)n3; (void)m3; (void)l3;
	for (int i = 0; i < n; ++i)
	for (int j = 0; j < m; ++j)
	for (int k = 0; k < l; ++k)
	{
		int im1 = (i > 0) ? i - 1 : i;
		int jm1 = (j > 0) ? j - 1 : j;
		int km1 = (k > 0) ? k - 1 : k;
		fx[flat3(i, j, k, m, l)] = (fi[flat3(im1, j, k, m, l)] - fi[flat3(i, j, k, m, l)]) / hx;
		fy[flat3(i, j, k, m, l)] = (fi[flat3(i, jm1, k, m, l)] - fi[flat3(i, j, k, m, l)]) / hy;
		fz[flat3(i, j, k, m, l)] = (fi[flat3(i, j, km1, m, l)] - fi[flat3(i, j, k, m, l)]) / hz;
	}
}

/*=========================================================================
 * Particles: initial lattice, push + 27-way migration, merge
 *=========================================================================*/

// Deterministic lattice-in-disk quiet start (XY plane, z=cz), ZERO
// initial velocity - simpler than the reference's ring/bisection
// fill_particles_coords + circular-orbit fill_particles_velocity (see
// pic3d.py's module docstring). Called on EVERY fragment, each filtering
// the SAME global npx*npy candidate lattice down to the points that fall
// within ITS OWN owned box [x1,x2)x[y1,y2)x[z1,z2) - mirrors the
// reference's own fill_particles_coords (each fragment generates only its
// own particles directly, no initial migration needed). This matters for
// nfx>2: loading everything onto one fragment and relying on push+merge
// to sort it out only works if every particle is at most one fragment-hop
// away, which fails as soon as the fragment grid is 3 or more wide in any
// axis (a particle can then be two hops from its real owning fragment,
// but the mailbox scheme only migrates one hop per timestep).
template <typename ElementT>
void generate_particles_disk_k(double cx, double cy, double cz, double radius,
                                double hx, double hy, double hz,
                                int ii, int jj, int kk, int bn, int bm, int bl,
                                int npx, int npy, ElementT* cell)
{
	double x1 = (ii * bn) * hx, x2 = (ii * bn + bn) * hx;
	double y1 = (jj * bm) * hy, y2 = (jj * bm + bm) * hy;
	double z1 = (kk * bl) * hz, z2 = (kk * bl + bl) * hz;

	double dlx = (2.0 * radius) / (npx - 1);
	double dly = (2.0 * radius) / (npy - 1);
	for (int i = 0; i < npx; ++i)
	{
		for (int j = 0; j < npy; ++j)
		{
			double x = cx - radius + i * dlx;
			double y = cy - radius + j * dly;
			double dx = x - cx, dy = y - cy;
			if (dx * dx + dy * dy > radius * radius) continue;
			if (x < x1 || x >= x2 || y < y1 || y >= y2 || cz < z1 || cz >= z2) continue;
			typename std::decay_t<decltype(cell->data_)>::value_type p;
			p.x = x; p.y = y; p.z = cz;
			p.u = 0.0; p.v = 0.0; p.w = 0.0;
			cell->data_.emplace_back(p);
		}
	}
}

// Staggered trilinear gather of (fx,fy,fz), matches particle_force(): the
// x-component is gathered at face-based (x/hx) positions in x and
// centre-based (x/hx-0.5) positions in y,z, and likewise (cyclically)
// for fy, fz.
template <typename ElementT>
void gather_force(double px, double py, double pz, double hx, double hy, double hz,
                   int ii, int jj, int kk, int bn, int bm, int bl,
                   int n, int m, int l, const double* fx, const double* fy, const double* fz,
                   double* out_fx, double* out_fy, double* out_fz)
{
	// n,m,l describe fx's shape; fy/fz share it (push_particles_block passes
	// each its own, identical, auto-injected triple)
	auto clampl = [&](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };

	double xb = px / hx;           int ib = (int)std::floor(xb);  double fxb = xb - ib;
	double yb = py / hy;           int jb = (int)std::floor(yb);  double fyb = yb - jb;
	double zb = pz / hz;           int kb = (int)std::floor(zb);  double fzb = zb - kb;
	double xa = px / hx - 0.5;     int ia = (int)std::floor(xa);  double fxa = xa - ia;
	double ya = py / hy - 0.5;     int ja = (int)std::floor(ya);  double fya = ya - ja;
	double za = pz / hz - 0.5;     int ka = (int)std::floor(za);  double fza = za - ka;

	int lib = clampl(ib - ii * bn + 1, 0, n - 2), lia = clampl(ia - ii * bn + 1, 0, n - 2);
	int ljb = clampl(jb - jj * bm + 1, 0, m - 2), lja = clampl(ja - jj * bm + 1, 0, m - 2);
	int lkb = clampl(kb - kk * bl + 1, 0, l - 2), lka = clampl(ka - kk * bl + 1, 0, l - 2);

	auto trilerp = [&](const double* f, int li, int lj, int lk, double di, double dj, double dk) {
		return f[flat3(li,     lj,     lk,     m, l)] * (1 - di) * (1 - dj) * (1 - dk) +
		       f[flat3(li + 1, lj,     lk,     m, l)] * (di)     * (1 - dj) * (1 - dk) +
		       f[flat3(li + 1, lj + 1, lk,     m, l)] * (di)     * (dj)     * (1 - dk) +
		       f[flat3(li,     lj + 1, lk,     m, l)] * (1 - di) * (dj)     * (1 - dk) +
		       f[flat3(li,     lj,     lk + 1, m, l)] * (1 - di) * (1 - dj) * (dk) +
		       f[flat3(li + 1, lj,     lk + 1, m, l)] * (di)     * (1 - dj) * (dk) +
		       f[flat3(li + 1, lj + 1, lk + 1, m, l)] * (di)     * (dj)     * (dk) +
		       f[flat3(li,     lj + 1, lk + 1, m, l)] * (1 - di) * (dj)     * (dk);
	};

	*out_fx = trilerp(fx, lib, lja, lka, fxb, fya, fza);
	*out_fy = trilerp(fy, lia, ljb, lka, fxa, fyb, fza);
	*out_fz = trilerp(fz, lia, lja, lkb, fxa, fya, fzb);
}

// Push (kick-drift) + reflect off the domain walls (position-based, so
// it is a correct no-op for any fragment not actually sitting on that
// wall) + route into one of 27 mailboxes based on which side of THIS
// fragment's own owned box [x1,x2)x[y1,y2)x[z1,z2) the new position ends
// up on, independently per axis - unlike an axis-priority scheme, this
// reproduces the reference's combined direction code exactly, so a
// particle that crosses two or three faces in the same step (a diagonal
// move) is still routed correctly in a single hop.
template <typename ElementTIn, typename ElementTOut>
void push_particles_block(double tau, double xn, double yn, double zn,
                           double hx, double hy, double hz,
                           int ii, int jj, int kk, int bn, int bm, int bl,
                           int fn, int fm, int fl, const double* fx,
                           int fn2, int fm2, int fl2, const double* fy,
                           int fn3, int fm3, int fl3, const double* fz,
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
	(void)fn2; (void)fm2; (void)fl2; (void)fn3; (void)fm3; (void)fl3;
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

	double x1 = (ii * bn) * hx,       x2 = (ii * bn + bn) * hx;
	double y1 = (jj * bm) * hy,       y2 = (jj * bm + bm) * hy;
	double z1 = (kk * bl) * hz,       z2 = (kk * bl + bl) * hz;

	for (const auto& old_p : cell_old->data_)
	{
		P p = old_p;

		double afx, afy, afz;
		gather_force<ElementTIn>(p.x, p.y, p.z, hx, hy, hz, ii, jj, kk, bn, bm, bl, fn, fm, fl, fx, fy, fz,
		                         &afx, &afy, &afz);

		p.u += tau * afx;
		p.v += tau * afy;
		p.w += tau * afz;

		// CFL clamp, matches particles_move_sort() in pic3d_common/common/
		// pic3d.cpp exactly: caps displacement at one cell per step. The
		// reference's own TODO notes this isn't a full fix ("make too fast
		// particles move cell by cell"), but without it a particle very
		// close to the point-mass singularity can gain enough velocity to
		// jump clear across a fragment in one step - more than this port's
		// single-hop-per-timestep mailbox scheme can route, so it's load-
		// bearing here, not just an optimization.
		double umax = hx / tau, vmax = hy / tau, wmax = hz / tau;
		if (p.u > umax) p.u = umax; else if (p.u < -umax) p.u = -umax;
		if (p.v > vmax) p.v = vmax; else if (p.v < -vmax) p.v = -vmax;
		if (p.w > wmax) p.w = wmax; else if (p.w < -wmax) p.w = -wmax;

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

// Each fragment's sole producer of its own next-version particle list:
// its own survivors (mail_self, written by ITS OWN push call) plus
// whatever each of its (up to 26) neighbors routed towards it. Direction
// naming is from the WRITER's own perspective, so what fragment F reads
// as "coming from direction d" is neighbor (F+d)'s OWN mailbox for the
// REVERSED direction (2-d per axis, 1 stays 1) - e.g. receiving "from
// low-x,low-y,self-z" (001) means reading the low-x-low-y neighbor's OWN
// "high-x,high-y,self-z" (221) mailbox, exactly mirroring the reference's
// own bi001 = bo221 convention. At a fragment-grid edge, where a given
// direction's neighbor doesn't exist, pic3d.py passes a harmless
// self-reference and an exists=0 flag - the loop below is skipped
// entirely in that case, so that reference is never dereferenced.
template <typename ElementT>
void merge_particles_block(
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

/*=========================================================================
 * Diagnostics (reduced: particle count + total kinetic energy proxy -
 * not the reference's full 13-field properties vector, see pic3d.py's
 * module docstring) and output
 *=========================================================================*/

template <typename ElementT>
void count_particles_k(const ElementT* cell, int& count)
{
	count += (int)cell->data_.size();
}

template <typename ElementT>
void sum_kinetic_energy_k(const ElementT* cell, double& ke_sum)
{
	double s = 0.0;
	for (const auto& p : cell->data_) s += p.u * p.u + p.v * p.v + p.w * p.w;
	ke_sum += s;
}

void print_step(int ts, int n_particles, double ke)
{
	std::printf("ts=%d n_particles=%d KE=%.6le\n", ts, n_particles, ke);
	std::fflush(stdout);
}

// Dumps this fragment's own density block (interior only, no ghost) to a
// small binary file under results/ every nts_dump steps (matches
// pic3d.par's own "nts" - the gate lives in C++, not as a pysol-level
// if, the same way pysol's other pic3d port gates dump_vti_block on
// "ts % 100" - a plain Python `if ts % nts_dump == 0:` can't express this,
// since pysol's `if` only accepts atom-to-atom comparisons, and "ts %
// nts_dump" is a BinOp). No in-program gather, so this works unchanged
// under MPI (mirrors pysol's other pic3d port's dump_vti_block/
// write_vti_files pattern). write_density_files() (run after pysol_run,
// plain Python, not a kernel) reads every fragment's file back together.
void dump_density_block(int ts, int nts_dump, int ii, int jj, int kk, int bn, int bm, int bl,
                         int n, int m, int l, const double* ro)
{
	if (ts % nts_dump != 0) return;

	char name[256];
	std::snprintf(name, sizeof(name), "results/ro_%05d_%d_%d_%d.bin", ts, ii, jj, kk);
	FILE* f = std::fopen(name, "wb");
	if (f == NULL) { std::fprintf(stderr, "Could not open %s\n", name); return; }
	for (int li = 1; li <= bn; ++li)
	for (int lj = 1; lj <= bm; ++lj)
	for (int lk = 1; lk <= bl; ++lk)
	{
		double v = ro[flat3(li, lj, lk, m, l)];
		std::fwrite(&v, sizeof(double), 1, f);
	}
	std::fclose(f);
}
