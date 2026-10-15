#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

/*=========================================================================
 * pysol reimplementation of lattice-gas.cpp's fluid solver (D2Q9 BGK
 * lattice-Boltzmann), as a lid-driven cavity instead of the reference's
 * channel + immersed-boundary particle (dropped - see lattice_gas.py's
 * module docstring for why).
 *
 * Fragmented into an nb x mb block grid, the same way pic3d/ucodes.h
 * fragments its own fields: f0..f8 (pre-collision populations, no ghost -
 * collision only ever reads a node's own populations) and f2_0..f2_8
 * (post-collision/post-bounce-back populations, 1-node ghost layer,
 * ghost-exchanged via pysol_update_array once per timestep before
 * streaming reads a neighboring block's edge). rho/ux/uy are fragmented,
 * no-ghost diagnostic/output fields, recomputed by collide_block every
 * timestep (so re-declaring them each call, the same safe pattern
 * pic3d's node_vol/den_ion/den_elec already use, is fine).
 *
 * None of this needs a GLOBAL, non-fragmented array or a static C++
 * buffer anywhere - no g_f/g_f2/g_rho/... of the kind the pic3d port
 * eliminated from pysol's own earlier draft.
 *=========================================================================*/

inline int flat2(int li, int lj, int m) { return li * m + lj; }

inline void lbm_weights(double w[9])
{
	static const double ws[9] = {
		4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
		1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
	for (int k = 0; k < 9; ++k) w[k] = ws[k];
}

inline void lbm_cx(int c[9])
{
	static const int cs[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
	for (int k = 0; k < 9; ++k) c[k] = cs[k];
}

inline void lbm_cy(int c[9])
{
	static const int cs[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
	for (int k = 0; k < 9; ++k) c[k] = cs[k];
}

// D2Q9 opposite-direction map (k -> -c[k]): 0 is its own opposite, the
// other 8 pair up (1<->3, 2<->4, 5<->7, 6<->8).
inline int lbm_opposite(int k)
{
	static const int op[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
	return op[k];
}

// Rest-state (rho=1, u=0) equilibrium everywhere - mirrors init()'s
// feq formula in the reference specialized to the quiescent initial
// condition init_hydro() sets up (dropping the reference's inlet
// ux=vel_center bulk initialization, since this port starts the cavity
// at rest and drives it purely from the lid).
void init_lbm_block(int bn, int bm,
                     int f0n, int f0m, double* f0, int f1n, int f1m, double* f1,
                     int f2n, int f2m, double* f2, int f3n, int f3m, double* f3,
                     int f4n, int f4m, double* f4, int f5n, int f5m, double* f5,
                     int f6n, int f6m, double* f6, int f7n, int f7m, double* f7,
                     int f8n, int f8m, double* f8)
{
	(void)f0n; (void)f0m; (void)f1n; (void)f1m; (void)f2n; (void)f2m;
	(void)f3n; (void)f3m; (void)f4n; (void)f4m; (void)f5n; (void)f5m;
	(void)f6n; (void)f6m; (void)f7n; (void)f7m; (void)f8n; (void)f8m;

	double w[9];
	lbm_weights(w);
	double* fout[9] = {f0, f1, f2, f3, f4, f5, f6, f7, f8};
	for (int li = 0; li < bn; ++li)
		for (int lj = 0; lj < bm; ++lj)
			for (int k = 0; k < 9; ++k)
				fout[k][flat2(li, lj, bm)] = w[k];
}

// Block (ii,jj)'s sole producer of its own next-version f2/rho/ux/uy:
// standard BGK collision at interior (non-wall) nodes, and a
// self-contained bounce-back (no neighbor read needed, unlike the
// reference's neighbor-based update_bounce_back - see lattice_gas.py's
// module docstring) at the 4 domain edges, combined into one pass so f2
// still has exactly one producer per block (collide_bulk +
// update_bounce_back being two separate passes over the SAME array in
// the reference would be two producers here, which pysol's
// single-assignment rule forbids - the same reason push_particles routes
// through mailboxes instead of writing a neighbor's cell directly in the
// pic3d port). Only the top edge (gj == gny - 1) carries a nonzero
// prescribed wall velocity (the lid); the other three edges are
// stationary no-slip walls - this works uniformly, including at the
// corners, since the bounce-back formula is the same algebraic
// expression for every k, just with uwx forced to 0 off the top edge.
void collide_block(int ii, int jj, int bn, int bm, int gnx, int gny,
                    double omega, double vel_lid,
                    int f0n, int f0m, const double* f0, int f1n, int f1m, const double* f1,
                    int f2n, int f2m, const double* f2, int f3n, int f3m, const double* f3,
                    int f4n, int f4m, const double* f4, int f5n, int f5m, const double* f5,
                    int f6n, int f6m, const double* f6, int f7n, int f7m, const double* f7,
                    int f8n, int f8m, const double* f8,
                    int g0n, int g0m, double* f2_0, int g1n, int g1m, double* f2_1,
                    int g2n, int g2m, double* f2_2, int g3n, int g3m, double* f2_3,
                    int g4n, int g4m, double* f2_4, int g5n, int g5m, double* f2_5,
                    int g6n, int g6m, double* f2_6, int g7n, int g7m, double* f2_7,
                    int g8n, int g8m, double* f2_8,
                    int rn, int rm, double* rho_out,
                    int un, int um, double* ux_out,
                    int vn, int vm, double* uy_out)
{
	(void)f0n; (void)f0m; (void)f1n; (void)f1m; (void)f2n; (void)f2m;
	(void)f3n; (void)f3m; (void)f4n; (void)f4m; (void)f5n; (void)f5m;
	(void)f6n; (void)f6m; (void)f7n; (void)f7m; (void)f8n; (void)f8m;
	(void)rn; (void)rm; (void)un; (void)um; (void)vn; (void)vm;
	// All 9 f2_* ghosted blocks share one shape (bn+2 x bm+2); g0m is
	// used below as that shared stride since every g*m is identical.
	const int gm_shared = g0m;

	double w[9];
	int cx[9], cy[9];
	lbm_weights(w);
	lbm_cx(cx);
	lbm_cy(cy);
	const double* fin[9] = {f0, f1, f2, f3, f4, f5, f6, f7, f8};
	double* fout[9] = {f2_0, f2_1, f2_2, f2_3, f2_4, f2_5, f2_6, f2_7, f2_8};
	const int g_n[9] = {g0n, g1n, g2n, g3n, g4n, g5n, g6n, g7n, g8n};
	const int g_m[9] = {g0m, g1m, g2m, g3m, g4m, g5m, g6m, g7m, g8m};

	for (int k = 0; k < 9; ++k)
		for (int n = 0; n < g_n[k] * g_m[k]; ++n)
			fout[k][n] = 0.0;

	for (int li = 0; li < bn; ++li)
	{
		for (int lj = 0; lj < bm; ++lj)
		{
			const int gi = ii * bn + li, gj = jj * bm + lj;
			const int in_idx = flat2(li, lj, bm);
			const int out_idx = flat2(li + 1, lj + 1, gm_shared);

			double fl[9];
			for (int k = 0; k < 9; ++k) fl[k] = fin[k][in_idx];

			const bool is_wall = (gi == 0) || (gi == gnx - 1) || (gj == 0) || (gj == gny - 1);
			if (!is_wall)
			{
				double rho = 0.0, mx = 0.0, my = 0.0;
				for (int k = 0; k < 9; ++k) { rho += fl[k]; mx += fl[k] * cx[k]; my += fl[k] * cy[k]; }
				const double uxv = mx / rho, uyv = my / rho;
				for (int k = 0; k < 9; ++k)
				{
					const double feq = w[k] * (rho + 3.0 * rho * (cx[k] * uxv + cy[k] * uyv)
					                            + 4.5 * rho * ((cx[k] * cx[k] - 1.0 / 3.0) * uxv * uxv
					                                           + (cy[k] * cy[k] - 1.0 / 3.0) * uyv * uyv
					                                           + 2.0 * uxv * uyv * cx[k] * cy[k]));
					fout[k][out_idx] = fl[k] * (1.0 - omega) + omega * feq;
				}
				rho_out[in_idx] = rho;
				ux_out[in_idx] = uxv;
				uy_out[in_idx] = uyv;
			}
			else
			{
				const double rho_wall = 1.0;
				const double uwx = (gj == gny - 1) ? vel_lid : 0.0;
				const double uwy = 0.0;
				for (int k = 0; k < 9; ++k)
				{
					const int o = lbm_opposite(k);
					fout[k][out_idx] = fl[o] + 2.0 * w[k] * 3.0 * rho_wall * (cx[k] * uwx + cy[k] * uwy);
				}
				rho_out[in_idx] = rho_wall;
				ux_out[in_idx] = uwx;
				uy_out[in_idx] = uwy;
			}
		}
	}
}

// Streams block (ii,jj)'s next-timestep populations from the
// ghost-exchanged f2: f_k(local) = f2_k(local - c[k]), reading into the
// neighboring block's edge when (local - c[k]) falls in the 1-node
// ghost layer pysol_update_array filled right before this call - the
// same role compute_efield_block's post-exchange read of ef_x/ef_y/ef_z
// plays in the pic3d port.
void stream_block(int bn, int bm,
                   int g0n, int g0m, const double* f2_0, int g1n, int g1m, const double* f2_1,
                   int g2n, int g2m, const double* f2_2, int g3n, int g3m, const double* f2_3,
                   int g4n, int g4m, const double* f2_4, int g5n, int g5m, const double* f2_5,
                   int g6n, int g6m, const double* f2_6, int g7n, int g7m, const double* f2_7,
                   int g8n, int g8m, const double* f2_8,
                   int f0n, int f0m, double* f0_out, int f1n, int f1m, double* f1_out,
                   int f2n, int f2m, double* f2_out, int f3n, int f3m, double* f3_out,
                   int f4n, int f4m, double* f4_out, int f5n, int f5m, double* f5_out,
                   int f6n, int f6m, double* f6_out, int f7n, int f7m, double* f7_out,
                   int f8n, int f8m, double* f8_out)
{
	(void)g0n;
	(void)g1n; (void)g1m; (void)g2n; (void)g2m; (void)g3n; (void)g3m;
	(void)g4n; (void)g4m; (void)g5n; (void)g5m; (void)g6n; (void)g6m;
	(void)g7n; (void)g7m; (void)g8n; (void)g8m;
	(void)f0n; (void)f0m; (void)f1n; (void)f1m; (void)f2n; (void)f2m;
	(void)f3n; (void)f3m; (void)f4n; (void)f4m; (void)f5n; (void)f5m;
	(void)f6n; (void)f6m; (void)f7n; (void)f7m; (void)f8n; (void)f8m;
	// All 9 f2_* ghosted blocks share one shape (bn+2 x bm+2).
	const int gm_shared = g0m;

	int cx[9], cy[9];
	lbm_cx(cx);
	lbm_cy(cy);
	const double* fin[9] = {f2_0, f2_1, f2_2, f2_3, f2_4, f2_5, f2_6, f2_7, f2_8};
	double* fout[9] = {f0_out, f1_out, f2_out, f3_out, f4_out, f5_out, f6_out, f7_out, f8_out};

	for (int li = 0; li < bn; ++li)
	{
		for (int lj = 0; lj < bm; ++lj)
		{
			const int out_idx = flat2(li, lj, bm);
			for (int k = 0; k < 9; ++k)
			{
				const int si = (li + 1) - cx[k];
				const int sj = (lj + 1) - cy[k];
				fout[k][out_idx] = fin[k][flat2(si, sj, gm_shared)];
			}
		}
	}
}

// Dumps block (ii,jj)'s own rho/ux/uy straight to its own file under
// results/ - no in-program gather, mirroring pic3d's dump_vti_block;
// lattice_gas.py's write_vtk_file() (run after the simulation, plain
// Python) reads every block's file back and assembles a single VTK
// STRUCTURED_POINTS file.
void dump_lbm_block(int ts, int ii, int jj, int bn, int bm,
                     int rn, int rm, const double* rho,
                     int un, int um, const double* ux,
                     int vn, int vm, const double* uy)
{
	(void)rn; (void)rm; (void)un; (void)um; (void)vn; (void)vm;

	// Every 1000th step only (matches the reference's NOUTPUT=1000) - same
	// "gate inside the kernel, not a pysol `if` around the call" pattern
	// pic3d's dump_vti_block uses, so the call site stays unconditional.
	if (ts % 1000 != 0) {
		return;
	}

	std::filesystem::create_directories("results");
	std::ostringstream name;
	name << "results/lbm_block_" << std::setfill('0') << std::setw(5) << ts
	     << "_" << ii << "_" << jj << ".bin";
	std::ofstream out(name.str(), std::ios::binary);
	if (!out.is_open())
	{
		std::cerr << "Could not open " << name.str() << "\n";
		return;
	}

	for (int n = 0; n < bn * bm; ++n)
	{
		double rec[3] = {rho[n], ux[n], uy[n]};
		out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
	}
}

/*=========================================================================
 * Live visualization: streams velocity blocks to a separate Python
 * process (visualizer.py) over a plain TCP socket as the simulation
 * runs, instead of waiting for a file dump - visualizer.py redraws the
 * instant a block arrives, so the flow is visibly developing in real
 * time rather than only inspectable after the fact from results/*.vtk.
 *
 * All configuration is via environment variables, read once (not pysol
 * kernel arguments - adding a string argument to every call site for a
 * host/port that's the same for the whole run would be needless
 * plumbing), so the IP/port are easy to change without touching this
 * file or lattice_gas.py:
 *   LATTICE_GAS_VIZ_ENABLE  - must be "1" to turn this on at all. Unset
 *                             (the default) means a normal run never
 *                             even attempts a connection, so it never
 *                             pays a connect-timeout cost when nobody's
 *                             listening.
 *   LATTICE_GAS_VIZ_HOST    - default "127.0.0.1".
 *   LATTICE_GAS_VIZ_PORT    - default 8765.
 *   LATTICE_GAS_VIZ_EVERY   - send every Nth step (default 50) - separate
 *                             from dump_lbm_block's "every 1000th step"
 *                             file-dump cadence, since a live view wants
 *                             much finer granularity than a final-
 *                             analysis file dump does.
 *=========================================================================*/
namespace lattice_gas_viz_detail
{

inline int& socket_fd()
{
	static int fd = -1;
	return fd;
}

inline bool& connect_attempted()
{
	static bool attempted = false;
	return attempted;
}

// Lazily connects on the first call (and only the first call - every
// later call just reuses the same open socket, or does nothing once a
// connection attempt has failed/been disabled, so a missing visualizer
// costs exactly one failed connect() for the whole run, not one per
// send). Returns whether there's now a live, usable connection.
inline bool ensure_connected()
{
	if (connect_attempted())
	{
		return socket_fd() >= 0;
	}
	connect_attempted() = true;

	const char* enable_env = std::getenv("LATTICE_GAS_VIZ_ENABLE");
	if (!enable_env || std::string(enable_env) != "1")
	{
		return false;
	}

	const char* host_env = std::getenv("LATTICE_GAS_VIZ_HOST");
	const char* port_env = std::getenv("LATTICE_GAS_VIZ_PORT");
	const std::string host = host_env ? host_env : "127.0.0.1";
	const int port = port_env ? std::atoi(port_env) : 8765;

	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
	{
		std::cerr << "lattice-gas viz: socket() failed, live visualization disabled\n";
		return false;
	}

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(static_cast<uint16_t>(port));
	if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
	{
		std::cerr << "lattice-gas viz: invalid LATTICE_GAS_VIZ_HOST '" << host << "'\n";
		close(fd);
		return false;
	}

	if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
	{
		std::cerr << "lattice-gas viz: could not connect to " << host << ":" << port
		          << " - is visualizer.py running? Live visualization disabled for this run.\n";
		close(fd);
		return false;
	}

	std::cerr << "lattice-gas viz: connected to " << host << ":" << port << "\n";
	socket_fd() = fd;
	return true;
}

inline bool send_all(const void* data, std::size_t len)
{
	const char* p = static_cast<const char*>(data);
	std::size_t sent = 0;
	while (sent < len)
	{
		const ssize_t n = send(socket_fd(), p + sent, len - sent, 0);
		if (n <= 0)
		{
			return false;
		}
		sent += static_cast<std::size_t>(n);
	}
	return true;
}

inline void disconnect()
{
	close(socket_fd());
	socket_fd() = -1;
}

} // namespace lattice_gas_viz_detail

// Sends block (ii,jj)'s velocity (ux,uy) for global grid (gnx,gny) to the
// live visualizer - see the module comment above for the wire format and
// the LATTICE_GAS_VIZ_* environment variables. A 32-byte header (a 4-byte
// "LGVP" magic plus 7 little-endian int32 fields: step, ii, jj, bn, bm,
// gnx, gny) is followed by bn*bm raw float64 ux values then bn*bm raw
// float64 uy values - visualizer.py's HEADER_FORMAT/struct.unpack must
// stay in sync with this layout if either side changes.
void send_velocity_block(int ts, int ii, int jj, int bn, int bm, int gnx, int gny,
                          int un, int um, const double* ux,
                          int vn, int vm, const double* uy)
{
	(void)un; (void)um; (void)vn; (void)vm;

	const char* every_env = std::getenv("LATTICE_GAS_VIZ_EVERY");
	const int every = every_env ? std::atoi(every_env) : 50;
	if (every <= 0 || ts % every != 0)
	{
		return;
	}

	if (!lattice_gas_viz_detail::ensure_connected())
	{
		return;
	}

	char header[32];
	std::memcpy(header, "LGVP", 4);
	const int32_t fields[7] = {ts, ii, jj, bn, bm, gnx, gny};
	std::memcpy(header + 4, fields, sizeof(fields));

	const bool ok = lattice_gas_viz_detail::send_all(header, sizeof(header))
	                && lattice_gas_viz_detail::send_all(ux, static_cast<std::size_t>(bn) * bm * sizeof(double))
	                && lattice_gas_viz_detail::send_all(uy, static_cast<std::size_t>(bn) * bm * sizeof(double));

	if (!ok)
	{
		std::cerr << "lattice-gas viz: send failed, disabling live visualization for the rest of this run\n";
		lattice_gas_viz_detail::disconnect();
	}
}

void print_step(int ts)
{
	std::cout << "ts=" << ts << std::endl;
}
