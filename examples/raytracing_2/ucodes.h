#pragma once

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
 * pysol port of examples/raytracing_2/shader.glsl - a Phong-lighting ray
 * tracer (2 planes, 2 animated spheres, 2 point lights, reflection bounces
 * via NUM_ITERATIONS=2, i.e. 3 total CastRay evaluations per pixel),
 * transliterated line-for-line from GLSL into plain C++ doubles - same
 * approach as examples/raytracing's ucodes.h (Vec3 stands in for GLSL's
 * vec3; "out" parameters become C++ references).
 *
 * Like examples/raytracing, calc_block's only inputs are a handful of
 * scalars (image resolution, animation time) plus its own block's pixel
 * range - every pixel is independent of every other pixel, so there is no
 * pysol_update_array anywhere in raytracer_2.py.
 *
 * The camera itself is static in the original shader (cam_pos/cam_lookat
 * are fixed); all of the motion comes from the two spheres' positions,
 * which already depend on iTime (InitScene's abs(sin(2*iTime)) bounce and
 * cos(iTime)/sin(iTime) circling). raytracer_2.py's main loop advances a
 * single `t` scalar by a fixed dt every frame and passes it straight into
 * calc_block in place of iTime.
 *
 * r/g/b are SingleVersionArray (is_single_version=True) for the same
 * reason as examples/raytracing's: this runs continuously, and a plain
 * Array would otherwise keep every historical frame's image resident.
 *
 * send_image_block streams each block's RGB tile to visualizer.py (copied
 * into this directory, with its own RT2_VIZ_* env vars / "RTI2" magic /
 * default port 8768 so it can run alongside examples/raytracing's without
 * cross-wiring) - same mutex-guarded-send architecture as examples/
 * raytracing and examples/gas-ca.
 *============================================================================*/

inline int flat2(int li, int lj, int m) { return li * m + lj; }

// -- Vec3: a stand-in for GLSL's vec3 --------------------------------------

struct Vec3
{
	double x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator-(Vec3 a) { return Vec3{-a.x, -a.y, -a.z}; }
inline Vec3 operator*(Vec3 a, double s) { return Vec3{a.x * s, a.y * s, a.z * s}; }
inline Vec3 operator*(double s, Vec3 a) { return a * s; }
// Component-wise multiply - GLSL's vec3*vec3 (used throughout PhongLighting
// for k_a/k_d/k_r/I_a/I_source), unlike examples/raytracing which never
// needed it.
inline Vec3 operator*(Vec3 a, Vec3 b) { return Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }

inline double rt2_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 rt2_cross(Vec3 a, Vec3 b)
{
	return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double rt2_length(Vec3 a) { return std::sqrt(rt2_dot(a, a)); }
inline Vec3 rt2_normalize(Vec3 a)
{
	const double l = rt2_length(a);
	return l > 0.0 ? a * (1.0 / l) : a;
}
// GLSL's reflect(I, N): the incident vector I reflected about unit normal N.
inline Vec3 rt2_reflect(Vec3 i, Vec3 n) { return i - n * (2.0 * rt2_dot(n, i)); }
inline double rt2_clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }

const double RT2_PI = 3.1415926536;
const Vec3 RT2_BACKGROUND_COLOR{0.1, 0.2, 0.6};
const double RT2_FOVY = 50.0 * RT2_PI / 180.0;
const double RT2_DEFAULT_TMIN = 10.0e-4;
const double RT2_DEFAULT_TMAX = 10.0e6;
const int RT2_NUM_ITERATIONS = 2;

struct Plane
{
	double A, B, C, D;
	int material_id;
};

struct Sphere
{
	Vec3 center;
	double radius;
	int material_id;
};

struct Light
{
	Vec3 position;
	Vec3 I_a;
	Vec3 I_source;
};

struct Material
{
	Vec3 k_a;
	Vec3 k_d;
	Vec3 k_r;
	Vec3 k_rg;
	double n;
};

struct Ray
{
	Vec3 o;
	Vec3 d;
};

inline bool rt2_intersect_plane(const Plane& pln, const Ray& ray, double tmin, double tmax,
                                 double& t, Vec3& hit_pos, Vec3& hit_normal)
{
	const Vec3 N{pln.A, pln.B, pln.C};
	const double NRd = rt2_dot(N, ray.d);
	const double NRo = rt2_dot(N, ray.o);
	const double t0 = (-pln.D - NRo) / NRd;
	if (t0 < tmin || t0 > tmax)
	{
		return false;
	}
	t = t0;
	hit_pos = ray.o + ray.d * t0;
	hit_normal = rt2_normalize(N);
	return true;
}

inline bool rt2_intersect_sphere(const Sphere& sph, const Ray& ray, double tmin, double tmax,
                                  double& t, Vec3& hit_pos, Vec3& hit_normal)
{
	const Vec3 oc = ray.o - sph.center;
	const double a = rt2_dot(ray.d, ray.d);
	const double b = 2.0 * rt2_dot(oc, ray.d);
	const double c = rt2_dot(oc, oc) - sph.radius * sph.radius;
	const double discriminant = b * b - 4.0 * a * c;
	if (discriminant < 0.0)
	{
		return false;
	}

	const double sq = std::sqrt(discriminant);
	const double t0 = (-b - sq) / (2.0 * a);
	const double t1 = (-b + sq) / (2.0 * a);

	if ((t0 < tmin && t1 < tmin) || (t0 > tmax && t1 > tmax))
	{
		return false;
	}

	t = (t0 > tmin && t0 < tmax) ? t0 : t1;
	hit_pos = ray.o + ray.d * t;
	hit_normal = rt2_normalize(hit_pos - sph.center);
	return true;
}

// Input vectors L, N and V point AWAY from the surface point, all unit.
inline Vec3 rt2_phong_lighting(Vec3 L, Vec3 N, Vec3 V, bool in_shadow, const Material& mat,
                                const Light& light)
{
	if (in_shadow)
	{
		return light.I_a * mat.k_a;
	}
	const Vec3 R = rt2_reflect(-L, N);
	const double N_dot_L = std::max(0.0, rt2_dot(N, L));
	const double R_dot_V = std::max(0.0, rt2_dot(R, V));
	const double R_dot_V_pow_n = (R_dot_V == 0.0) ? 0.0 : std::pow(R_dot_V, mat.n);

	return light.I_a * mat.k_a + light.I_source * (mat.k_d * N_dot_L + mat.k_r * R_dot_V_pow_n);
}

// Casts one ray into the (2-plane, 2-sphere, 2-light) scene and returns the
// Phong-lit color at the nearest hit (or the background color on a miss).
// has_hit/hit_pos/hit_normal/k_rg are "out" parameters, exactly like the
// original GLSL CastRay.
inline Vec3 rt2_cast_ray(const Ray& ray, const Plane planes[2], const Sphere spheres[2],
                          const Material materials[3], const Light lights[2], bool& has_hit,
                          Vec3& hit_pos, Vec3& hit_normal, Vec3& k_rg)
{
	bool has_hit_something = false;
	double nearest_t = RT2_DEFAULT_TMAX;
	Vec3 nearest_hit_pos{0.0, 0.0, 0.0};
	Vec3 nearest_hit_normal{0.0, 0.0, 0.0};
	int nearest_mat_id = 0;

	double temp_t;
	Vec3 temp_hit_pos, temp_hit_normal;

	for (int i = 0; i < 2; ++i)
	{
		if (rt2_intersect_plane(planes[i], ray, RT2_DEFAULT_TMIN, RT2_DEFAULT_TMAX, temp_t, temp_hit_pos,
		                         temp_hit_normal) &&
		    temp_t < nearest_t)
		{
			has_hit_something = true;
			nearest_t = temp_t;
			nearest_hit_pos = temp_hit_pos;
			nearest_hit_normal = temp_hit_normal;
			nearest_mat_id = planes[i].material_id;
		}
	}
	for (int i = 0; i < 2; ++i)
	{
		if (rt2_intersect_sphere(spheres[i], ray, RT2_DEFAULT_TMIN, RT2_DEFAULT_TMAX, temp_t, temp_hit_pos,
		                          temp_hit_normal) &&
		    temp_t < nearest_t)
		{
			has_hit_something = true;
			nearest_t = temp_t;
			nearest_hit_pos = temp_hit_pos;
			nearest_hit_normal = temp_hit_normal;
			nearest_mat_id = spheres[i].material_id;
		}
	}

	has_hit = has_hit_something;
	if (!has_hit_something)
	{
		return RT2_BACKGROUND_COLOR;
	}

	Vec3 i_local{0.0, 0.0, 0.0};
	for (int i = 0; i < 2; ++i)
	{
		const Vec3 shadow_dir = lights[i].position - nearest_hit_pos;
		const double shadow_dist = rt2_length(shadow_dir);
		const Vec3 shadow_dir_n = rt2_normalize(shadow_dir);

		bool in_shadow = false;
		for (int j = 0; j < 2 && !in_shadow; ++j)
		{
			if (rt2_intersect_plane(planes[j], Ray{nearest_hit_pos, shadow_dir_n}, RT2_DEFAULT_TMIN,
			                         shadow_dist, temp_t, temp_hit_pos, temp_hit_normal))
			{
				in_shadow = true;
			}
		}
		for (int j = 0; j < 2 && !in_shadow; ++j)
		{
			if (rt2_intersect_sphere(spheres[j], Ray{nearest_hit_pos, shadow_dir_n}, RT2_DEFAULT_TMIN,
			                          shadow_dist, temp_t, temp_hit_pos, temp_hit_normal))
			{
				in_shadow = true;
			}
		}

		i_local = i_local + rt2_phong_lighting(shadow_dir_n, nearest_hit_normal, -ray.d, in_shadow,
		                                        materials[nearest_mat_id], lights[i]);
	}

	hit_pos = nearest_hit_pos;
	hit_normal = nearest_hit_normal;
	k_rg = materials[nearest_mat_id].k_rg;
	return i_local;
}

// Block (ii,jj)'s sole producer of its own r/g/b pixel tile - every pixel is
// an independent computation (no neighbor reads), same as examples/
// raytracing's calc_block.
void calc_block(int ii, int jj, int bn, int bm, int gnx, int gny, double t,
                 int rn, int rm, double* r_out,
                 int gcn, int gcm, double* g_out,
                 int bcn, int bcm, double* b_out)
{
	(void)rn; (void)gcn; (void)bcn;

	// -- scene (same values as InitScene(), spheres driven by t) --
	const Plane planes[2] = {
		Plane{0.0, 1.0, 0.0, 0.0, 0},
		Plane{0.0, 0.0, 1.0, 3.5, 0},
	};
	const Sphere spheres[2] = {
		Sphere{Vec3{0.0, std::fabs(std::sin(2.0 * t)) + 0.7, 0.0}, 0.7, 1},
		Sphere{Vec3{1.5 * std::cos(t), 0.5, 1.5 * std::sin(t)}, 0.5, 2},
	};
	Material materials[3];
	materials[0].k_d = Vec3{0.5, 0.5, 0.5};
	materials[0].k_a = materials[0].k_d * 0.2;
	materials[0].k_r = materials[0].k_d * 2.0;
	materials[0].k_rg = materials[0].k_r * 0.5;
	materials[0].n = 64.0;
	materials[1].k_d = Vec3{0.8, 0.7, 0.1};
	materials[1].k_a = materials[1].k_d * 0.2;
	materials[1].k_r = materials[1].k_d * 2.0;
	materials[1].k_rg = materials[1].k_r * 0.5;
	materials[1].n = 64.0;
	materials[2].k_d = Vec3{0.0, 0.8, 0.0};
	materials[2].k_a = materials[2].k_d * 0.2;
	materials[2].k_r = Vec3{1.0, 1.0, 1.0};
	materials[2].k_rg = materials[2].k_r * 0.5;
	materials[2].n = 128.0;
	const Light lights[2] = {
		Light{Vec3{4.0, 8.0, -3.0}, Vec3{0.1, 0.1, 0.1}, Vec3{1.0, 1.0, 1.0}},
		Light{Vec3{-4.0, 8.0, 0.0}, Vec3{0.1, 0.1, 0.1}, Vec3{1.0, 1.0, 1.0}},
	};

	// -- camera (static - all motion comes from the spheres above) --
	const Vec3 cam_pos{2.5, 1.0, 2.5};
	const Vec3 cam_lookat{0.25, 1.0, 0.0};
	const Vec3 cam_up{0.0, 1.0, 0.0};
	const Vec3 cam_z = rt2_normalize(cam_pos - cam_lookat);
	const Vec3 cam_x = rt2_normalize(rt2_cross(cam_up, cam_z));
	const Vec3 cam_y = rt2_normalize(rt2_cross(cam_z, cam_x));
	const double pixel_pos_z = -1.0 / std::tan(RT2_FOVY / 2.0);

	for (int li = 0; li < bn; ++li)
	{
		const int gi = ii * bn + li;
		for (int lj = 0; lj < bm; ++lj)
		{
			const int gj = jj * bm + lj;

			const double px = (2.0 * (gi + 0.5) - gnx) / gny;
			const double py = (2.0 * (gj + 0.5) - gny) / gny;

			const Ray primary_ray{cam_pos, rt2_normalize(cam_x * px + cam_y * py + cam_z * pixel_pos_z)};

			Vec3 i_result{0.0, 0.0, 0.0};
			Vec3 compounded_k_rg{1.0, 1.0, 1.0};
			Ray next_ray = primary_ray;

			for (int level = 0; level <= RT2_NUM_ITERATIONS; ++level)
			{
				bool has_hit;
				Vec3 hit_pos, hit_normal, k_rg;
				const Vec3 i_local =
					rt2_cast_ray(next_ray, planes, spheres, materials, lights, has_hit, hit_pos, hit_normal, k_rg);

				i_result = i_result + compounded_k_rg * i_local;

				if (!has_hit)
				{
					break;
				}

				compounded_k_rg = compounded_k_rg * k_rg;
				next_ray = Ray{hit_pos, rt2_normalize(rt2_reflect(next_ray.d, hit_normal))};
			}

			const int idx = flat2(li, lj, rm);
			r_out[idx] = rt2_clamp(i_result.x, 0.0, 1.0);
			g_out[idx] = rt2_clamp(i_result.y, 0.0, 1.0);
			b_out[idx] = rt2_clamp(i_result.z, 0.0, 1.0);
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
 * raytracing, with its own env var names / magic bytes / default port so
 * both examples can run side by side without cross-wiring.
 *========================================================================*/
namespace raytracing2_viz_detail
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

	const char* enable_env = std::getenv("RT2_VIZ_ENABLE");
	if (!enable_env || std::string(enable_env) != "1")
	{
		return false;
	}

	const char* host_env = std::getenv("RT2_VIZ_HOST");
	const char* port_env = std::getenv("RT2_VIZ_PORT");
	const std::string host = host_env ? host_env : "127.0.0.1";
	const int port = port_env ? std::atoi(port_env) : 8768;

	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
	{
		std::fprintf(stderr, "raytracing_2 viz: socket() failed, live visualization disabled\n");
		return false;
	}

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(static_cast<uint16_t>(port));
	if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
	{
		std::fprintf(stderr, "raytracing_2 viz: invalid RT2_VIZ_HOST '%s'\n", host.c_str());
		close(fd);
		return false;
	}

	if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
	{
		std::fprintf(stderr,
		              "raytracing_2 viz: could not connect to %s:%d - is visualizer.py "
		              "running? Live visualization disabled for this run.\n",
		              host.c_str(), port);
		close(fd);
		return false;
	}

	std::fprintf(stderr, "raytracing_2 viz: connected to %s:%d\n", host.c_str(), port);
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

}  // namespace raytracing2_viz_detail

// Wire format (must stay in sync with visualizer.py's HEADER_FORMAT/MAGIC):
// a 4-byte "RTI2" magic + 7x little-endian int32 (frame, ii, jj, bn, bm,
// gnx, gny), followed by bn*bm float64 r, then bn*bm float64 g, then
// bn*bm float64 b - all row-major [li][lj], same layout as calc_block
// writes them.
void send_image_block(int frame, int ii, int jj, int bn, int bm, int gnx, int gny,
                       int rn, int rm, const double* r,
                       int gcn, int gcm, const double* g,
                       int bcn, int bcm, const double* b)
{
	(void)rn; (void)rm; (void)gcn; (void)gcm; (void)bcn; (void)bcm;

	std::lock_guard<std::mutex> lock(raytracing2_viz_detail::send_mutex());

	if (!raytracing2_viz_detail::ensure_connected())
	{
		return;
	}

	char header[32];
	std::memcpy(header, "RTI2", 4);
	const int32_t fields[7] = {frame, ii, jj, bn, bm, gnx, gny};
	std::memcpy(header + 4, fields, sizeof(fields));

	const std::size_t block_bytes = static_cast<std::size_t>(bn) * bm * sizeof(double);
	const bool ok = raytracing2_viz_detail::send_all(header, sizeof(header)) &&
	                raytracing2_viz_detail::send_all(r, block_bytes) &&
	                raytracing2_viz_detail::send_all(g, block_bytes) &&
	                raytracing2_viz_detail::send_all(b, block_bytes);
	if (!ok)
	{
		std::fprintf(stderr, "raytracing_2 viz: send failed, dropping connection\n");
		close(raytracing2_viz_detail::socket_fd());
		raytracing2_viz_detail::socket_fd() = -1;
	}
}
