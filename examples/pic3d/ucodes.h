#pragma once

#include "CgGate/rts.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <mpi.h>

/*=========================================================================
 * pysol reimplementation of pic3d.cpp (3D electrostatic PIC solver).
 *
 * EVERYTHING here is fragmented into an nb x mb x lb block grid - phi/rho/
 * resid2/ef (1-node ghost layer per side, ghost-exchanged via
 * pysol_update_array), node_vol/den_ion/den_elec (no ghost - see
 * reduce_density_block), and particles (one GrowingArray cell per block,
 * no ghost - particles that cross a block boundary during a push are
 * migrated to the owning block's cell by the mailbox scheme described at
 * push_particles/merge_particles_mail below). None of it needs a GLOBAL,
 * non-fragmented array or a static C++ buffer anywhere, which is what
 * makes this run correctly distributed across more than one MPI process:
 * every array element has exactly one rank that owns and produces it, and
 * every cross-block dependency goes through a real pysol Array/
 * GrowingArray read, which the RTS fetches from whichever rank owns it
 * automatically.
 *
 * Every kernel argument that is a pysol Array gets its own block-size
 * integers (n, m, l) inserted immediately before it by the .sol generator
 * (one triple per array argument, even when two arrays share the same
 * shape) - every signature below is written to match that calling
 * convention exactly. GrowingArray<Particle> arguments get no such
 * prefix (n_dims=0 for them).
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

// Zero-fills a fragmented block (interior + ghost, or no-ghost - just
// however many elements n*m*l actually is).
void zero_block3d(int n, int m, int l, double* a)
{
	for (int i = 0; i < n * m * l; ++i)
	{
		a[i] = 0.0;
	}
}

// Node volumes for block (ii,jj,kk): full cell volume on interior nodes,
// halved per boundary face it's on - same formula as the original's
// computeNodeVolumes(), evaluated at each node's GLOBAL position, so no
// ghost/exchange is needed: unlike density, nothing here depends on a
// neighboring block's data. No ghost layer either (node_vol_block is
// [bn,bm,bl]) - see reduce_density_block for why it doesn't need one.
void compute_node_volumes_block(int ii, int jj, int kk,
                                 double dhx, double dhy, double dhz,
                                 int gn, int gm, int gl,
                                 int bn, int bm, int bl, double* node_vol_block)
{
	for (int li = 0; li < bn; ++li)
	{
		for (int lj = 0; lj < bm; ++lj)
		{
			for (int lk = 0; lk < bl; ++lk)
			{
				int gi = ii * bn + li;
				int gj = jj * bm + lj;
				int gk = kk * bl + lk;
				double V = dhx * dhy * dhz;
				if (gi == 0 || gi == gn - 1) V *= 0.5;
				if (gj == 0 || gj == gm - 1) V *= 0.5;
				if (gk == 0 || gk == gl - 1) V *= 0.5;
				node_vol_block[flat3(li, lj, lk, bm, bl)] = V;
			}
		}
	}
}

/*=========================================================================
 * Particle loading / push / inter-block migration
 *=========================================================================*/

// Quiet start: loads a uniform npx*npy*npz macroparticle lattice over box
// [x1, x2) representing number density num_den, with zero initial
// velocity (the electric field is exactly zero before the first Poisson
// solve, so addParticle()'s half-step velocity rewind in the original is
// a no-op here and is omitted). Mirrors loadParticlesBoxQS()+addParticle().
//
// Always called on block (0,0,0) only (see pic3d.py): every particle
// starts out sitting in the wrong cell for its actual position, except
// whichever happen to fall in block (0,0,0)'s own territory. That's fine
// - the very first push's mailbox migration (see push_particles /
// merge_particles_mail below) sorts every particle into its real owning
// block before anything else reads particles_ion/particles_elec, so
// load_particles_box_qs itself never needs to know about block
// boundaries at all.
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

// Particle push for block (ii,jj,kk): v += (q/m)E dt, x += v dt, specular
// reflection on the domain walls - same physics as the original's
// pushParticles() + XtoL() + gatherEF() + gather(), but E is gathered
// (trilinear) from THIS block's own ghost-exchanged ef_x/ef_y/ef_z
// fragment instead of a global array, since every particle in cell_old is
// guaranteed (by the previous timestep's merge_particles_mail) to be
// owned by this block, i.e. within its local range - the trilinear
// gather's "+1" neighbor access is always covered by this block's own
// 1-node ghost layer.
//
// ParSolGen arrays (and GrowingArray cells) are single-assignment: a
// kernel that modifies one must take the OLD and NEW array elements
// separately, like hello.py's calc_main(prev, curr). A particle's pushed
// position may now belong to a DIFFERENT block, though, so there is no
// single "new cell" to rebuild into the way a single-rank global particle
// population could: each particle is routed into one of 7 output cells -
// "mail_self" (stayed in this block) or one of the 6 face-direction
// mailboxes (mail_xm/xp/ym/yp/zm/zp) - instead of being written directly
// into a neighbor's cell, which would be a second, independent producer
// of that neighbor's array element and violate single-assignment exactly
// like the old global ef_x/ef_y/ef_z did. merge_particles_mail (below)
// is each block's single, sole producer of its own next-version cell,
// combining its own mail_self with whatever its 6 neighbors routed
// towards it.
//
// This assumes a particle never crosses more than one block in a single
// timestep (a real constraint on dt relative to block size, same as any
// CFL-style stability bound - if it's violated the simulation is already
// numerically suspect for other reasons). A particle whose push would
// need a genuinely diagonal (2- or 3-axis) hop is routed by its first
// nonzero axis instead of being dropped - a documented approximation, not
// a particle loss or a crash.
//
// "Fresh" NEW version does not mean "empty": a GrowingArray's backing
// store only has a finite number of physical slots (nversions), so a
// mailbox cell can be a slot last used several iterations ago and still
// holding old data - every mailbox is explicitly cleared before being
// rebuilt so leftover data from a previous reuse of that slot is never
// appended to.
template <typename ElementTIn, typename ElementTOut>
void push_particles(double dt, double qm_dt,
                     double x0x, double x0y, double x0z,
                     double xmx, double xmy, double xmz,
                     double dhx, double dhy, double dhz,
                     int ii, int jj, int kk, int nb, int mb, int lb,
                     int efnx, int efmx, int eflx, const double* ef_x_block,
                     int efny, int efmy, int efly, const double* ef_y_block,
                     int efnz, int efmz, int eflz, const double* ef_z_block,
                     const ElementTIn* cell_old,
                     ElementTOut* mail_self, ElementTOut* mail_xm, ElementTOut* mail_xp,
                     ElementTOut* mail_ym, ElementTOut* mail_yp,
                     ElementTOut* mail_zm, ElementTOut* mail_zp)
{
	(void)efny; (void)efmy; (void)efly; (void)efnz; (void)efmz; (void)eflz;
	const int pm = efmx, pl = eflx;
	const int bn = efnx - 2, bm = efmx - 2, bl = eflx - 2;

	using P = typename std::decay_t<decltype(mail_self->data_)>::value_type;

	mail_self->data_.clear();
	mail_xm->data_.clear();
	mail_xp->data_.clear();
	mail_ym->data_.clear();
	mail_yp->data_.clear();
	mail_zm->data_.clear();
	mail_zp->data_.clear();

	auto gather = [&](const double* f, int li, int lj, int lk, double di, double dj, double dk) -> double {
		return f[flat3(li,     lj,     lk,     pm, pl)] * (1 - di) * (1 - dj) * (1 - dk) +
		       f[flat3(li + 1, lj,     lk,     pm, pl)] * (di)     * (1 - dj) * (1 - dk) +
		       f[flat3(li + 1, lj + 1, lk,     pm, pl)] * (di)     * (dj)     * (1 - dk) +
		       f[flat3(li,     lj + 1, lk,     pm, pl)] * (1 - di) * (dj)     * (1 - dk) +
		       f[flat3(li,     lj,     lk + 1, pm, pl)] * (1 - di) * (1 - dj) * (dk) +
		       f[flat3(li + 1, lj,     lk + 1, pm, pl)] * (di)     * (1 - dj) * (dk) +
		       f[flat3(li + 1, lj + 1, lk + 1, pm, pl)] * (di)     * (dj)     * (dk) +
		       f[flat3(li,     lj + 1, lk + 1, pm, pl)] * (1 - di) * (dj)     * (dk);
	};

	for (const auto& old_p : cell_old->data_)
	{
		P p = old_p;

		double lcx = (p.pos_x - x0x) / dhx;
		double lcy = (p.pos_y - x0y) / dhy;
		double lcz = (p.pos_z - x0z) / dhz;
		int gi = (int)lcx; double di = lcx - gi;
		int gj = (int)lcy; double dj = lcy - gj;
		int gk = (int)lcz; double dk = lcz - gk;

		// Local (block+ghost) indices for the OLD position: the particle
		// was placed in this block's cell by the previous merge, so its
		// old position is guaranteed within this block's owned range
		// (li in [1,bn]) - clamped defensively anyway so a violation of
		// the single-block-hop assumption degrades to a slightly wrong
		// gather rather than an out-of-bounds read.
		int li = gi - ii * bn + 1;
		int lj = gj - jj * bm + 1;
		int lk = gk - kk * bl + 1;
		if (li < 1) li = 1; else if (li > bn) li = bn;
		if (lj < 1) lj = 1; else if (lj > bm) lj = bm;
		if (lk < 1) lk = 1; else if (lk > bl) lk = bl;

		double efx = gather(ef_x_block, li, lj, lk, di, dj, dk);
		double efy = gather(ef_y_block, li, lj, lk, di, dj, dk);
		double efz = gather(ef_z_block, li, lj, lk, di, dj, dk);

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

		// Reflection guarantees the NEW position is within [x0, xm), so
		// the destination block index is always within [0, nb) etc - no
		// clamping needed here (unlike the defensive clamp above).
		int ngi = (int)((p.pos_x - x0x) / dhx);
		int ngj = (int)((p.pos_y - x0y) / dhy);
		int ngk = (int)((p.pos_z - x0z) / dhz);
		int dii = ngi / bn;
		int djj = ngj / bm;
		int dkk = ngk / bl;

		int ddi = dii - ii, ddj = djj - jj, ddk = dkk - kk;
		ElementTOut* dst = mail_self;
		if (ddi < 0) dst = mail_xm;
		else if (ddi > 0) dst = mail_xp;
		else if (ddj < 0) dst = mail_ym;
		else if (ddj > 0) dst = mail_yp;
		else if (ddk < 0) dst = mail_zm;
		else if (ddk > 0) dst = mail_zp;

		dst->data_.emplace_back(p);
	}
	(void)nb; (void)mb; (void)lb;
}

// Block (ii,jj,kk)'s sole producer of its own next-version particle cell:
// combines its own survivors (from_self, written by ITS OWN push call's
// mail_self) with whatever its 6 face neighbors routed towards it (each
// neighbor's push wrote into ONE of ITS OWN 6 mailboxes, which this block
// reads as from_xm/xp/ym/yp/zm/zp) - see push_particles above for the
// mailbox scheme and why a particle can't just be written directly into a
// neighbor's cell. At a domain edge, where a face neighbor doesn't exist,
// pic3d.py passes THIS block's own, guaranteed-empty mailbox for that
// direction instead (no particle is ever routed off the edge of the
// domain, since push_particles reflects positions before computing a
// destination block) - so every one of the 7 inputs here is always a
// valid read, no boundary special-casing needed in C++ at all.
template <typename ElementTIn, typename ElementTOut>
void merge_particles_mail(const ElementTIn* from_self,
                           const ElementTIn* from_xm, const ElementTIn* from_xp,
                           const ElementTIn* from_ym, const ElementTIn* from_yp,
                           const ElementTIn* from_zm, const ElementTIn* from_zp,
                           ElementTOut* cell_new)
{
	cell_new->data_.clear();
	for (const auto& p : from_self->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_xm->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_xp->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_ym->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_yp->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_zm->data_) cell_new->data_.emplace_back(p);
	for (const auto& p : from_zp->data_) cell_new->data_.emplace_back(p);
}

/*=========================================================================
 * Number density (fragmented scatter + additive ghost reduction)
 *=========================================================================*/

// Scatters block (ii,jj,kk)'s own particles' macroparticle weight
// (trilinear) onto a RAW, UNNORMALIZED charge buffer with the same
// 1-node-ghost shape as phi: a particle near this block's edge
// contributes partly to a node actually owned by the neighboring block,
// landing in this block's ghost layer exactly the way sor_sweep_block's
// stencil reads phi's ghost. Mirrors computeNumberDensity()'s scatter
// step, except NOT yet divided by node volume, and NOT yet reduced across
// blocks - see reduce_density_block below for both of those, done once
// per node rather than once per contributing block (same reason
// compute_density's node_vol division in the original PIC3D port happened
// once at the end, not per particle).
template <typename ElementT>
void compute_density_raw_block(double x0x, double x0y, double x0z,
                                double dhx, double dhy, double dhz,
                                int ii, int jj, int kk,
                                const ElementT* cell,
                                int dn, int dm, int dl, double* den_raw)
{
	for (int n = 0; n < dn * dm * dl; ++n) den_raw[n] = 0.0;

	const int bn = dn - 2, bm = dm - 2, bl = dl - 2;

	for (const auto& p : cell->data_)
	{
		double lcx = (p.pos_x - x0x) / dhx;
		double lcy = (p.pos_y - x0y) / dhy;
		double lcz = (p.pos_z - x0z) / dhz;
		int gi = (int)lcx; double di = lcx - gi;
		int gj = (int)lcy; double dj = lcy - gj;
		int gk = (int)lcz; double dk = lcz - gk;

		int li = gi - ii * bn + 1;
		int lj = gj - jj * bm + 1;
		int lk = gk - kk * bl + 1;
		if (li < 1) li = 1; else if (li > bn) li = bn;
		if (lj < 1) lj = 1; else if (lj > bm) lj = bm;
		if (lk < 1) lk = 1; else if (lk > bl) lk = bl;

		double v = p.mpw;
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

// Reduces block (ii,jj,kk)'s raw scatter (see compute_density_raw_block)
// into the final, normalized density: its own interior plus, on each of
// the 6 faces it owns, the matching ghost-layer spillover computed BY
// each neighbor's own raw scatter (xm_raw/xp_raw/ym_raw/yp_raw/zm_raw/
// zp_raw) - this is the additive counterpart of phi's copy-only
// pysol_update_array ghost exchange: pysol_update_array (generated from
// exchange_3d.sol's _write_ghost3d_*) overwrites a ghost cell with a
// neighbor's value, which is correct for phi/ef (a node's potential/field
// doesn't change depending on which block looks at it) but wrong for
// density (a particle's scatter weight must be counted once, not
// replace whatever the owning block already had there) - so this reduces
// by hand instead of calling pysol_update_array.
//
// fxm/fxp/fym/fyp/fzm/fzp select, per direction, which local index of the
// corresponding raw block to read: the neighbor's matching ghost face
// (e.g. xm_raw's "+i" ghost, index bn+1) when a real neighbor exists, or
// this block's OWN same-direction ghost (always zero - no particle can be
// scattered beyond the domain edge) at a domain boundary, where
// xm_raw/xp_raw/etc is simply this block's own raw buffer again, passed
// by pic3d.py exactly like merge_particles_mail's boundary handling.
void reduce_density_block(int fxm, int fxp, int fym, int fyp, int fzm, int fzp,
                           int sn, int sm, int sl, const double* self_raw,
                           int n_xm, int m_xm, int l_xm, const double* xm_raw,
                           int n_xp, int m_xp, int l_xp, const double* xp_raw,
                           int n_ym, int m_ym, int l_ym, const double* ym_raw,
                           int n_yp, int m_yp, int l_yp, const double* yp_raw,
                           int n_zm, int m_zm, int l_zm, const double* zm_raw,
                           int n_zp, int m_zp, int l_zp, const double* zp_raw,
                           int vn, int vm, int vl, const double* node_vol,
                           int dn, int dm, int dl, double* den)
{
	(void)n_xm; (void)m_xm; (void)l_xm; (void)n_xp; (void)m_xp; (void)l_xp;
	(void)n_ym; (void)m_ym; (void)l_ym; (void)n_yp; (void)m_yp; (void)l_yp;
	(void)n_zm; (void)m_zm; (void)l_zm; (void)n_zp; (void)m_zp; (void)l_zp;
	(void)vn; (void)vm;

	const int bn = sn - 2, bm = sm - 2, bl = sl - 2;

	for (int li = 1; li <= bn; ++li)
	{
		for (int lj = 1; lj <= bm; ++lj)
		{
			for (int lk = 1; lk <= bl; ++lk)
			{
				int n = flat3(li, lj, lk, sm, sl);
				double total = self_raw[n];
				if (li == 1)  total += xm_raw[flat3(fxm, lj, lk, sm, sl)];
				if (li == bn) total += xp_raw[flat3(fxp, lj, lk, sm, sl)];
				if (lj == 1)  total += ym_raw[flat3(li, fym, lk, sm, sl)];
				if (lj == bm) total += yp_raw[flat3(li, fyp, lk, sm, sl)];
				if (lk == 1)  total += zm_raw[flat3(li, lj, fzm, sm, sl)];
				if (lk == bl) total += zp_raw[flat3(li, lj, fzp, sm, sl)];

				int vidx = flat3(li - 1, lj - 1, lk - 1, vm, vl);
				int didx = flat3(li - 1, lj - 1, lk - 1, dm, dl);
				den[didx] = (node_vol[vidx] != 0.0) ? (total / node_vol[vidx]) : 0.0;
			}
		}
	}
}

// Writes the current particle count of a species' block into `count`, an
// "out" scalar kernel argument - ACCUMULATES rather than overwrites, so
// pic3d.py can call it once per block and add up a whole-domain total the
// same way it already summed den_ion/den_elec (see sum_block below).
template <typename ElementT>
void count_particles(const ElementT* cell, int& count)
{
	count += (int)cell->data_.size();
}

// Sums every cell of block (ii,jj,kk)'s den_ion/den_elec fragment into
// `result`, an "out" scalar kernel argument - ACCUMULATES (+= , not =)
// across repeated per-block calls, since den_ion/den_elec are now
// fragmented (nb*mb*lb blocks) rather than one single-block global array,
// so sum()'s whole-array reduce no longer applies the way it does for
// resid2/phi (whose block grid matches the program's main distributor);
// pic3d.py resets the accumulator to 0 before the per-block loop, exactly
// like n_ion/n_elec above.
void sum_block(int gn, int gm, int gl, const double* block, double& result)
{
	double total = 0.0;
	for (int n = 0; n < gn * gm * gl; ++n) total += block[n];
	result += total;
}

/*=========================================================================
 * Charge density (per block, no bridging needed anymore)
 *=========================================================================*/

// rho = q_ion*den_ion + q_elec*den_elec, for block (ii,jj,kk)'s own
// (already fully reduced) den_ion/den_elec fragments, written directly
// into rho_frag's interior - single assignment, one block, one producer,
// no GLOBAL rho array or distribute_rho_block bridge needed at all now
// that density itself is fragmented. Mirrors computeChargeDensity().
void compute_charge_density_block(double q_ion, double q_elec,
                                   int dnI, int dmI, int dlI, const double* den_ion,
                                   int dnE, int dmE, int dlE, const double* den_elec,
                                   int rn, int rm, int rl, double* rho_block)
{
	(void)dnE; (void)dmE; (void)dlE;
	const int bn = rn - 2, bm = rm - 2, bl = rl - 2;
	for (int li = 0; li < bn; ++li)
	{
		for (int lj = 0; lj < bm; ++lj)
		{
			for (int lk = 0; lk < bl; ++lk)
			{
				int didx = flat3(li, lj, lk, dmI, dlI);
				int ridx = flat3(li + 1, lj + 1, lk + 1, rm, rl);
				rho_block[ridx] = den_ion[didx] * q_ion + den_elec[didx] * q_elec;
			}
		}
	}
}

/*=========================================================================
 * VTI output - per block, no gather of any kind (see write_vti_files() in
 * pic3d.py for the Python-side post-processing that assembles the final
 * .vti from these).
 *=========================================================================*/

// Writes block (ii,jj,kk)'s own fields - phi, ef (now real, ghost-
// exchanged pysol Arrays, so no finite-difference recompute needed here
// any more), rho, den_ion, den_elec, node_vol - to
// "results/vti_block_<ts>_<ii>_<jj>_<kk>.bin": bn*bm*bl records of 8
// doubles in li/lj/lk nested order, no header - node (gi,gj,gk)/record
// index is recoverable from the filename plus bn/bm/bl alone, since
// gi = ii*bn + (li-1) etc (see distribute_rho_block's old convention,
// still used here). Every rank only ever owns and writes the blocks
// assigned to it, so (given a shared output directory) every block's
// file exists once the run finishes regardless of how many ranks it ran
// across - no cross-rank gather needed in C++ at all. write_vti_files()
// in pic3d.py (run after the simulation, as a plain Python post-
// processing step) reads every block's file and assembles the final
// .vti, exactly like examples/simple/update_array_3d_loop_test.py's
// verify() reads per-block check files after the run instead of
// gathering them inside the .sol-compiled program.
void dump_vti_block(int ts, int ii, int jj, int kk,
                     int pn, int pm, int pl, const double* phi_block,
                     int enx, int emx, int elx, const double* ef_x_block,
                     int eny, int emy, int ely, const double* ef_y_block,
                     int enz, int emz, int elz, const double* ef_z_block,
                     int rn, int rm, int rl, const double* rho_block,
                     int dnI, int dmI, int dlI, const double* den_ion_block,
                     int dnE, int dmE, int dlE, const double* den_elec_block,
                     int vn, int vm, int vl, const double* node_vol_block)
{
	(void)enx; (void)emx; (void)elx; (void)eny; (void)emy; (void)ely;
	(void)enz; (void)emz; (void)elz; (void)rn; (void)rm; (void)rl;
	(void)dnE; (void)dmE; (void)dlE; (void)vn; (void)vm; (void)vl;

	if (ts % 100 != 0) {
		return;
	}

	const int bn = pn - 2, bm = pm - 2, bl = pl - 2;

	std::filesystem::create_directories("results");
	std::ostringstream name;
	name << "results/vti_block_" << std::setfill('0') << std::setw(5) << ts
	     << "_" << ii << "_" << jj << "_" << kk << ".bin";
	std::ofstream out(name.str(), std::ios::binary);
	if (!out.is_open())
	{
		std::cerr << "Could not open " << name.str() << "\n";
		return;
	}

	for (int li = 1; li <= bn; ++li)
	{
		for (int lj = 1; lj <= bm; ++lj)
		{
			for (int lk = 1; lk <= bl; ++lk)
			{
				int n_gh = flat3(li, lj, lk, pm, pl);
				int n_pl = flat3(li - 1, lj - 1, lk - 1, dmI, dlI);
				double rec[8] = {
					phi_block[n_gh], ef_x_block[n_gh], ef_y_block[n_gh], ef_z_block[n_gh],
					rho_block[n_gh], den_ion_block[n_pl], den_elec_block[n_pl], node_vol_block[n_pl]
				};
				out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
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
 * Electric field (fragmented phi -> fragmented ef, ghost-exchanged by
 * pysol_update_array exactly like phi itself)
 *=========================================================================*/

// E = -grad(phi): central differences using the (ghost-exchanged)
// neighboring block where available, 2nd-order one-sided differences at
// the true global domain edges. Writes block (ii,jj,kk)'s own interior
// into ef_x_new/ef_y_new/ef_z_new (single assignment - one block, one
// producer) and zeroes their ghost layer, which pysol_update_array then
// fills from the neighboring blocks' interiors, exactly the way phi's own
// ghost gets filled after each SOR sweep. Mirrors computeElectricField().
void compute_efield_block(int ii, int jj, int kk,
                           double dhx, double dhy, double dhz,
                           int pn, int pm, int pl, const double* phi_block,
                           int gn, int gm, int gl,
                           int enx, int emx, int elx, double* ef_x_new,
                           int eny, int emy, int ely, double* ef_y_new,
                           int enz, int emz, int elz, double* ef_z_new)
{
	(void)eny; (void)emy; (void)ely; (void)enz; (void)emz; (void)elz;
	const int bn = pn - 2, bm = pm - 2, bl = pl - 2;
	const int si = pm * pl, sj = pl, sk = 1;

	for (int n = 0; n < enx * emx * elx; ++n)
	{
		ef_x_new[n] = 0.0;
		ef_y_new[n] = 0.0;
		ef_z_new[n] = 0.0;
	}

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

				ef_x_new[n] = Ex;
				ef_y_new[n] = Ey;
				ef_z_new[n] = Ez;
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
