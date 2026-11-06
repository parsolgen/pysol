#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <mutex>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

/*=============================================================================
 * pysol port of examples/water/shadel.glsl - a procedural raymarched water
 * surface (Gerstner-ish layered wave field, sun/sky/cloud background,
 * fresnel reflection + refraction glint), transliterated line-for-line from
 * GLSL into plain C++ doubles - same approach as examples/raytracing and
 * examples/raytracing_2 (Vec2/Vec3 stand in for GLSL's vec2/vec3).
 *
 * The shader's camera only moves in response to iMouse, which this headless
 * port has no equivalent of - "click" is therefore always false, so the
 * mouse-driven ro.yz/ro.zx rotations in render() are dead code here and are
 * not ported. With no mouse, the camera is fully static; all of the motion
 * in the image comes from wave() / wavedx()'s dependence on iTime, which
 * this port threads through explicitly as the `t` parameter instead of a
 * GLSL global (same approach as examples/raytracing_2's cam/spheres).
 *
 * #define AA 1.0 in the original means mainImage's multi-sample loop is
 * always skipped (the `if(AA==1.0)` early-return branch) - so this port
 * only implements the AA==1 path (one render() call per pixel) and does not
 * port the supersampling loop.
 *
 * waterCol in render() is assigned twice in the original, the first time
 * with a plain `=` that's immediately and fully overwritten by the second
 * (not `+=`) - only the second, live assignment is ported here.
 *
 * calc_block's only inputs are a handful of scalars (block/image geometry,
 * animation time) plus its own block's pixel range - every pixel is an
 * independent computation (one raymarch), so there is no pysol_update_array
 * anywhere in this example either.
 *
 * r/g/b are SingleVersionArray (is_single_version=True), same reasoning as
 * examples/raytracing and examples/raytracing_2: meant to run continuously,
 * so only the latest frame's pixels need to stay resident.
 *
 * send_image_block streams each block's RGB tile to visualizer.py (its own
 * WTR_VIZ_* env vars / "WTIM" magic / default port 8769, so it can run
 * alongside the other two examples' visualizers without cross-wiring) -
 * same mutex-guarded-send architecture as those examples.
 *============================================================================*/

inline int flat2(int li, int lj, int m) { return li * m + lj; }

// -- Vec2/Vec3: stand-ins for GLSL's vec2/vec3 -----------------------------

struct Vec2
{
	double x, y;
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return Vec2{a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return Vec2{a.x - b.x, a.y - b.y}; }
inline Vec2 operator-(Vec2 a) { return Vec2{-a.x, -a.y}; }
inline Vec2 operator*(Vec2 a, double s) { return Vec2{a.x * s, a.y * s}; }
inline Vec2 operator*(double s, Vec2 a) { return a * s; }
// Component-wise multiply - GLSL's vec2*vec2 (used by render()'s vignette:
// uv*vec2(0.8,1.)).
inline Vec2 operator*(Vec2 a, Vec2 b) { return Vec2{a.x * b.x, a.y * b.y}; }
inline Vec2 operator/(Vec2 a, double s) { return Vec2{a.x / s, a.y / s}; }

struct Vec3
{
	double x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return Vec3{-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, double s) { return Vec3{a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
// Component-wise multiply - GLSL's vec3*vec3.
inline Vec3 operator*(Vec3 a, Vec3 b) { return Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }
inline Vec3 operator/(Vec3 a, double s) { return Vec3{a.x / s, a.y / s, a.z / s}; }
// Scalar broadcast add/sub - GLSL's vec3+float / float-vec3 (e.g. sky()'s
// "p+=s;" and "1.0-sun*3.75").
inline Vec3 operator+(Vec3 a, double s) { return Vec3{a.x + s, a.y + s, a.z + s}; }
inline Vec3 operator-(double s, Vec3 a) { return Vec3{s - a.x, s - a.y, s - a.z}; }

inline double wtr_dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline double wtr_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 wtr_cross(Vec3 a, Vec3 b)
{
	return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double wtr_length(Vec2 a) { return std::sqrt(wtr_dot(a, a)); }
inline double wtr_length(Vec3 a) { return std::sqrt(wtr_dot(a, a)); }
inline Vec3 wtr_normalize(Vec3 a)
{
	const double l = wtr_length(a);
	return l > 0.0 ? a * (1.0 / l) : a;
}
inline Vec3 wtr_sin(Vec3 a) { return Vec3{std::sin(a.x), std::sin(a.y), std::sin(a.z)}; }
inline Vec3 wtr_cos(Vec3 a) { return Vec3{std::cos(a.x), std::cos(a.y), std::cos(a.z)}; }
inline Vec3 wtr_pow(Vec3 a, double e) { return Vec3{std::pow(a.x, e), std::pow(a.y, e), std::pow(a.z, e)}; }

// GLSL's reflect(I, N) / refract(I, N, eta).
inline Vec3 wtr_reflect(Vec3 i, Vec3 n) { return i - n * (2.0 * wtr_dot(n, i)); }
inline Vec3 wtr_refract(Vec3 i, Vec3 n, double eta)
{
	const double n_dot_i = wtr_dot(n, i);
	const double k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
	if (k < 0.0)
	{
		return Vec3{0.0, 0.0, 0.0};
	}
	return i * eta - n * (eta * n_dot_i + std::sqrt(k));
}

inline double wtr_clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
inline Vec3 wtr_clamp(Vec3 v, double a, double b)
{
	return Vec3{wtr_clamp(v.x, a, b), wtr_clamp(v.y, a, b), wtr_clamp(v.z, a, b)};
}
inline Vec3 wtr_mix(Vec3 a, Vec3 b, double t) { return a * (1.0 - t) + b * t; }
inline double wtr_smoothstep(double edge0, double edge1, double x)
{
	const double t = wtr_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
	return t * t * (3.0 - 2.0 * t);
}

// GLSL's rot(a) = mat2(cos a, sin a, -sin a, cos a), applied as "v *= rot(a)"
// to a swizzled pair of components (e.g. rd.yz, p.xz) - works out to the
// ordinary 2D rotation formula regardless of GLSL's matrix/vector
// multiplication order for this particular (column0=(cos,sin),
// column1=(-sin,cos)) matrix. wtr_rotate_pair rotates the pair (a, b) in
// place, standing in for e.g. "rd.yz *= rot(theta);" as
// wtr_rotate_pair(rd.y, rd.z, theta).
inline Vec2 wtr_rotate(Vec2 v, double theta)
{
	const double c = std::cos(theta), s = std::sin(theta);
	return Vec2{v.x * c - v.y * s, v.x * s + v.y * c};
}
inline void wtr_rotate_pair(double& a, double& b, double theta)
{
	const double c = std::cos(theta), s = std::sin(theta);
	const double na = a * c - b * s;
	const double nb = a * s + b * c;
	a = na;
	b = nb;
}

// -- shader constants (#define's and global vec2/float initializers) -----

const double WTR_STEPS = 80.0;
const double WTR_MDIST = 35.0;
const double WTR_PI = 3.1415926535;

const int WTR_ITERS_TRACE = 9;
const int WTR_ITERS_NORM = 20;

const double WTR_HOR_SCALE = 1.1;
const double WTR_OCC_SPEED = 1.4;
const double WTR_DX_DET = 0.65;

const double WTR_FREQ = 0.6;
const double WTR_HEIGHT_DIV = 2.5;
const double WTR_WEIGHT_SCL = 0.8;
const double WTR_FREQ_SCL = 1.2;
const double WTR_TIME_SCL = 1.095;
const double WTR_WAV_ROT = 1.21;
const double WTR_DRAG = 0.9;
const double WTR_SCRL_SPEED = 1.5;
const Vec2 WTR_SCROLL_DIR{1.0, 1.0};

const double WTR_SUNROT_X = -0.3;
const double WTR_SUNROT_Y = -0.25;
const double WTR_SPEC = 0.13;

// -- wavedx/wave: the layered procedural wave field ------------------------

inline Vec2 wtr_wavedx(Vec2 wav_pos, int iters, double t)
{
	Vec2 dx{0.0, 0.0};
	Vec2 wav_dir{1.0, 0.0};
	double wav_weight = 1.0;
	wav_pos = wav_pos + t * WTR_SCRL_SPEED * WTR_SCROLL_DIR;
	wav_pos = wav_pos * WTR_HOR_SCALE;
	double wav_freq = WTR_FREQ;
	double wav_time = WTR_OCC_SPEED * t;
	for (int i = 0; i < iters; ++i)
	{
		wav_dir = wtr_rotate(wav_dir, WTR_WAV_ROT);
		const double x = wtr_dot(wav_dir, wav_pos) * wav_freq + wav_time;
		double result = std::exp(std::sin(x) - 1.0) * std::cos(x);
		result *= wav_weight;
		dx = dx + (wav_dir * result) / std::pow(wav_weight, WTR_DX_DET);
		wav_freq *= WTR_FREQ_SCL;
		wav_time *= WTR_TIME_SCL;
		wav_pos = wav_pos - wav_dir * (result * WTR_DRAG);
		wav_weight *= WTR_WEIGHT_SCL;
	}
	const double wav_sum = -(std::pow(WTR_WEIGHT_SCL, static_cast<double>(iters)) - 1.0) * WTR_HEIGHT_DIV;
	return dx / std::pow(wav_sum, 1.0 - WTR_DX_DET);
}

inline double wtr_wave(Vec2 wav_pos, int iters, double t)
{
	double wav = 0.0;
	Vec2 wav_dir{1.0, 0.0};
	double wav_weight = 1.0;
	wav_pos = wav_pos + t * WTR_SCRL_SPEED * WTR_SCROLL_DIR;
	wav_pos = wav_pos * WTR_HOR_SCALE;
	double wav_freq = WTR_FREQ;
	double wav_time = WTR_OCC_SPEED * t;
	for (int i = 0; i < iters; ++i)
	{
		wav_dir = wtr_rotate(wav_dir, WTR_WAV_ROT);
		const double x = wtr_dot(wav_dir, wav_pos) * wav_freq + wav_time;
		const double wave_i = std::exp(std::sin(x) - 1.0) * wav_weight;
		wav += wave_i;
		wav_freq *= WTR_FREQ_SCL;
		wav_time *= WTR_TIME_SCL;
		wav_pos = wav_pos - wav_dir * (wave_i * WTR_DRAG * std::cos(x));
		wav_weight *= WTR_WEIGHT_SCL;
	}
	const double wav_sum = -(std::pow(WTR_WEIGHT_SCL, static_cast<double>(iters)) - 1.0) * WTR_HEIGHT_DIV;
	return wav / wav_sum;
}

inline Vec3 wtr_norm(Vec3 p, double t)
{
	const Vec2 wav = -wtr_wavedx(Vec2{p.x, p.z}, WTR_ITERS_NORM, t);
	return wtr_normalize(Vec3{wav.x, 1.0, wav.y});
}

inline double wtr_map(Vec3 p, double t)
{
	p.y -= wtr_wave(Vec2{p.x, p.z}, WTR_ITERS_TRACE, t);
	return p.y;
}

// -- palette / sky ----------------------------------------------------------

inline Vec3 wtr_pal(double t, Vec3 a, Vec3 b, Vec3 c, Vec3 d)
{
	return a + b * wtr_cos((c * t + d) * (2.0 * WTR_PI));
}
inline Vec3 wtr_spc(double n, double bright)
{
	return wtr_pal(n, Vec3{bright, bright, bright}, Vec3{0.5, 0.5, 0.5}, Vec3{1.0, 1.0, 1.0},
	                Vec3{0.0, 0.33, 0.67});
}

inline Vec3 wtr_sky(Vec3 rd, double gnx_d, double gny_d, double t)
{
	const double min_res = std::min(gnx_d, gny_d);
	const double px = 1.5 / min_res;
	const Vec3 rdo = rd;
	const double rad = 0.015;
	Vec3 col{0.0, 0.0, 0.0};

	// Sun.
	wtr_rotate_pair(rd.y, rd.z, WTR_SUNROT_Y);
	wtr_rotate_pair(rd.x, rd.z, WTR_SUNROT_X);
	const double s_fade = 2.5 / min_res;
	const double z_fade = rd.z * 0.5 + 1.;

	const Vec3 sc = wtr_spc(WTR_SPEC - 0.1, 0.6) * 0.85;
	const double a = wtr_length(Vec2{rd.x, rd.y});
	const Vec3 sun = sc * (wtr_smoothstep(a - px - s_fade, a + px + s_fade, rad) * z_fade * 2.0);
	col = col + sun;
	col = col + sc * (rad / (rad + std::pow(a, 1.7)) * z_fade);
	col = col + wtr_mix(col, wtr_spc(WTR_SPEC + 0.1, 0.8), wtr_clamp(1.0 - wtr_length(col), 0.0, 1.0)) * 0.2;

	// The implication of clouds.
	double e = 0.0;
	Vec3 p = rdo;
	p.x *= 0.4;
	p.z *= 0.4;
	p.x += t * 0.007;
	for (double s = 200.0; s > 10.0; s *= 0.8)
	{
		wtr_rotate_pair(p.x, p.z, s);
		p = p + s;
		const Vec3 term = wtr_sin(p * s + t * 0.02) / s;
		e += std::fabs(wtr_dot(term, Vec3{1.65, 1.65, 1.65}));
	}
	e *= wtr_smoothstep(0.5, 0.4, e - 0.095);

	col = col + wtr_mix(sc, Vec3{1.0, 1.0, 1.0}, 0.4) *
	                (e * wtr_smoothstep(-0.02, 0.3, rdo.y) * 0.8 * (1.0 - sun * 3.75));

	return col;
}

// -- render: one raymarch per pixel, the direct analogue of the shader's
// render() (mainImage with AA==1 just forwards straight into it) ----------

inline Vec3 wtr_render(double fx, double fy, double gnx_d, double gny_d, double t)
{
	const double min_res = std::min(gnx_d, gny_d);
	const Vec2 uv{(fx - 0.5 * gnx_d) / min_res, (fy - 0.5 * gny_d) / min_res};
	Vec3 col{0.0, 0.0, 0.0};
	// No mouse support -> "click" is always false, so this is simply the
	// shader's fixed ro (the click-driven ro.yz/ro.zx rotations are dead
	// code here and are not ported).
	const Vec3 ro = Vec3{0.0, 2.25, -3.0} * 1.1;
	const Vec3 lk{0.0, 2.0, 0.0};
	const Vec3 f = wtr_normalize(lk - ro);
	const Vec3 r = wtr_normalize(wtr_cross(Vec3{0.0, 1.0, 0.0}, f));
	const Vec3 rd = wtr_normalize(f * 0.9 + r * uv.x + wtr_cross(f, r) * uv.y);

	double dO = 0.0;
	bool hit = false;
	Vec3 p = ro;

	const double t_pln = -(ro.y - 1.86) / rd.y;
	if (t_pln > 0.0)
	{
		dO += t_pln;
		for (double i = 0.0; i < WTR_STEPS; i += 1.0)
		{
			p = ro + rd * dO;
			const double d = wtr_map(p, t);
			dO += d;
			if (std::fabs(d) < 0.005 || i > WTR_STEPS - 2.0)
			{
				hit = true;
				break;
			}
			if (dO > WTR_MDIST)
			{
				dO = WTR_MDIST;
				break;
			}
		}
	}

	const Vec3 skyrd = wtr_sky(rd, gnx_d, gny_d, t);
	if (hit)
	{
		Vec3 n = wtr_norm(p, t);
		Vec3 rfl = wtr_reflect(rd, n);
		rfl.y = std::fabs(rfl.y);
		Vec3 rf = wtr_refract(rd, n, 1.0 / 1.33);
		const double fres = wtr_clamp(std::pow(1.0 - std::max(0.0, wtr_dot(-n, rd)), 5.0), 0.0, 1.0);

		Vec3 sun_dir{0.0, 0.15, 1.0};
		wtr_rotate_pair(sun_dir.x, sun_dir.z, -WTR_SUNROT_X);

		col = col + wtr_sky(rfl, gnx_d, gny_d, t) * (fres * 0.9);
		const double sub_refract = std::pow(std::max(0.0, wtr_dot(rf, sun_dir)), 35.0);
		col = col + wtr_pow(wtr_spc(WTR_SPEC - 0.1, 0.5), 2.2) * (sub_refract * 2.5);

		Vec3 rd2 = rd;
		wtr_rotate_pair(rd2.x, rd2.z, WTR_SUNROT_X);

		// (The shader's first waterCol assignment is fully overwritten by
		// this one - see the module comment - so only this is ported.)
		const Vec3 water_col =
		    wtr_clamp(wtr_spc(WTR_SPEC - 0.1, 0.4), 0.0, 1.0) *
		    (0.4 * std::pow(std::min(p.y * 0.7 + 0.9, 1.8), 4.0) * wtr_length(skyrd) * (rd2.z * 0.15 + 0.85));
		col = col + water_col * 0.17;

		col = wtr_mix(col, skyrd, dO / WTR_MDIST);
	}
	else
	{
		col = col + skyrd;
	}

	col = wtr_clamp(col, 0.0, 1.0);
	col = wtr_pow(col, 0.87);
	col = col * (1.0 - 0.8 * std::pow(wtr_length(uv * Vec2{0.8, 1.0}), 2.7));

	return col;
}

// Block (ii,jj)'s sole producer of its own r/g/b pixel tile - every pixel is
// an independent raymarch (no neighbor reads), same as examples/raytracing
// and examples/raytracing_2's calc_block.
void calc_block(int ii, int jj, int bn, int bm, int gnx, int gny, double t, int rn, int rm, double* r_out, int gcn,
                 int gcm, double* g_out, int bcn, int bcm, double* b_out)
{
	(void)rn;
	(void)gcn;
	(void)bcn;

	const double gnx_d = static_cast<double>(gnx);
	const double gny_d = static_cast<double>(gny);

	for (int li = 0; li < bn; ++li)
	{
		const int gi = ii * bn + li;
		for (int lj = 0; lj < bm; ++lj)
		{
			const int gj = jj * bm + lj;
			// fragCoord pixel centers; gj=0 is the BOTTOM row, matching the
			// shader's own convention (see visualizer.py's vertical flip).
			const double fx = gi + 0.5;
			const double fy = gj + 0.5;

			const Vec3 col = wtr_render(fx, fy, gnx_d, gny_d, t);

			const int idx = flat2(li, lj, rm);
			r_out[idx] = wtr_clamp(col.x, 0.0, 1.0);
			g_out[idx] = wtr_clamp(col.y, 0.0, 1.0);
			b_out[idx] = wtr_clamp(col.z, 0.0, 1.0);
		}
	}
}

void print_frame(int frame)
{
	std::printf("frame=%d\n", frame);
	std::fflush(stdout);
}

/*=========================================================================
 * Live-image streaming to visualizer.py - same architecture as examples/
 * raytracing and examples/raytracing_2, with its own env var names / magic
 * bytes / default port so all three can run side by side without
 * cross-wiring.
 *========================================================================*/
namespace water_viz_detail
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

inline std::mutex& send_mutex()
{
	static std::mutex m;
	return m;
}

inline bool ensure_connected()
{
	if (connect_attempted())
	{
		return socket_fd() >= 0;
	}
	connect_attempted() = true;

	const char* enable_env = std::getenv("WTR_VIZ_ENABLE");
	if (!enable_env || std::string(enable_env) != "1")
	{
		return false;
	}

	const char* host_env = std::getenv("WTR_VIZ_HOST");
	const char* port_env = std::getenv("WTR_VIZ_PORT");
	const std::string host = host_env ? host_env : "127.0.0.1";
	const int port = port_env ? std::atoi(port_env) : 8769;

	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
	{
		std::fprintf(stderr, "water viz: socket() failed, live visualization disabled\n");
		return false;
	}

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(static_cast<uint16_t>(port));
	if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
	{
		std::fprintf(stderr, "water viz: invalid WTR_VIZ_HOST '%s'\n", host.c_str());
		close(fd);
		return false;
	}

	if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
	{
		std::fprintf(stderr,
		              "water viz: could not connect to %s:%d - is visualizer.py "
		              "running? Live visualization disabled for this run.\n",
		              host.c_str(), port);
		close(fd);
		return false;
	}

	std::fprintf(stderr, "water viz: connected to %s:%d\n", host.c_str(), port);
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

}  // namespace water_viz_detail

// Wire format (must stay in sync with visualizer.py's HEADER_FORMAT/MAGIC):
// a 4-byte "WTIM" magic + 7x little-endian int32 (frame, ii, jj, bn, bm,
// gnx, gny), followed by bn*bm float64 r, then bn*bm float64 g, then bn*bm
// float64 b - all row-major [li][lj], same layout as calc_block writes
// them.
void send_image_block(int frame, int ii, int jj, int bn, int bm, int gnx, int gny, int rn, int rm, const double* r,
                       int gcn, int gcm, const double* g, int bcn, int bcm, const double* b)
{
	(void)rn;
	(void)rm;
	(void)gcn;
	(void)gcm;
	(void)bcn;
	(void)bcm;

	std::lock_guard<std::mutex> lock(water_viz_detail::send_mutex());

	if (!water_viz_detail::ensure_connected())
	{
		return;
	}

	char header[32];
	std::memcpy(header, "WTIM", 4);
	const int32_t fields[7] = {frame, ii, jj, bn, bm, gnx, gny};
	std::memcpy(header + 4, fields, sizeof(fields));

	const std::size_t block_bytes = static_cast<std::size_t>(bn) * bm * sizeof(double);
	const bool ok = water_viz_detail::send_all(header, sizeof(header)) &&
	                water_viz_detail::send_all(r, block_bytes) && water_viz_detail::send_all(g, block_bytes) &&
	                water_viz_detail::send_all(b, block_bytes);
	if (!ok)
	{
		std::fprintf(stderr, "water viz: send failed, dropping connection\n");
		close(water_viz_detail::socket_fd());
		water_viz_detail::socket_fd() = -1;
	}
}
