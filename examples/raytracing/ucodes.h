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
 * pysol port of examples/raytracing/shader.gl - a Shadertoy-style analytic
 * ray tracer (one ground plane with a checker pattern + 3 animated
 * spheres, one reflection bounce, two refraction bounces, Fresnel mixing
 * between them, and a hard shadow ray), transliterated line-for-line from
 * GLSL into plain C++ doubles (Vec3 below stands in for GLSL's vec3).
 *
 * Unlike every other example here, the "simulation" has no cell-to-cell
 * dependency at all: calc_block's only inputs are a handful of scalars
 * (image resolution, camera angle, animation time) and its own block's
 * pixel range - every pixel's color is a pure function of those, computed
 * independently of every other pixel and every other block. So there is
 * no pysol_update_array anywhere in raytracer.py - there is no ghost
 * margin to exchange, because there is no neighbor read to begin with.
 *
 * The "camera orbits the scene" animation from the original shader
 * (ro/rd both built from `rk*iTime`) is driven here by raytracer.py's
 * main loop instead of a free-running shader clock: each frame it
 * advances a single `t` scalar by a fixed dt and passes `t` (for the
 * spheres' vertical bobbing, cos(t) - same as the original's cos(iTime))
 * and `rk*t` (the camera's orbit angle) into calc_block. That outer loop
 * is exactly "wrap the algorithm in a for-loop that rotates the camera a
 * little bit each time" - see raytracer.py.
 *
 * r/g/b are declared as SingleVersionArray (Array(..., is_single_version=
 * True) on the Python side) rather than a plain Array: this is meant to
 * run continuously at an exhibition (num_frames is huge, like examples/
 * gas-ca's), and a plain Array would keep every historical frame's full
 * image resident forever - SingleVersionArray keeps only the latest.
 *
 * send_image_block streams each block's freshly computed RGB tile to
 * visualizer.py (copied into this directory) over a TCP socket the
 * instant it's computed - the exact same pygame+socket architecture as
 * examples/gas-ca and examples/pic3d_v2's live_vis.py, including the
 * mutex around the whole connect-then-send critical section (needed
 * because OMP-parallel block kernel calls can otherwise race on the
 * shared socket and interleave bytes on the wire - a bug previously hit
 * and fixed in examples/gas-ca).
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

inline double rt_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double rt_length(Vec3 a) { return std::sqrt(rt_dot(a, a)); }
inline Vec3 rt_normalize(Vec3 a)
{
	const double l = rt_length(a);
	return l > 0.0 ? a * (1.0 / l) : a;
}
inline double rt_clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
// GLSL's mod(a,b) (always in [0,b), unlike C's fmod for negative a).
inline double rt_glsl_mod(double a, double b) { return a - b * std::floor(a / b); }

const double RT_MAX_DIST = 100.0;

struct Material
{
	double refr_koef;
	Vec3 color;
};

struct Sphere
{
	double radius;
	Vec3 center;
	Material material;
};

struct Plane
{
	Vec3 normal;
	double y;
	Material material;
};

struct Ray
{
	Vec3 origin;
	Vec3 direction;
};

struct Intersect
{
	double dist;
	Vec3 normal;
	Material material;
};

// A miss sentinel: dist == RT_MAX_DIST, exactly like the original shader's
// "Intersect miss" global - every "hit != miss" check below becomes a
// "hit.dist < RT_MAX_DIST" check instead (equivalent, since no real hit in
// this scene lands exactly on - or past - the far clip distance).
inline Intersect rt_miss()
{
	return Intersect{RT_MAX_DIST, Vec3{0.0, 0.0, 0.0}, Material{0.0, Vec3{1.0, 1.0, 1.0}}};
}

inline Intersect rt_intersect_plane(const Ray& ray, const Plane& plane)
{
	if (ray.direction.y < 0.0)
	{
		Intersect hit;
		const double t = std::fabs(plane.y / ray.direction.y);
		const Vec3 OP = ray.direction * t;
		hit.dist = rt_length(OP);
		hit.normal = plane.normal;
		hit.material = plane.material;
		hit.material.color = Vec3{0.0, 1.0, 1.0};

		const double cell_size = 5.0;
		const Vec3 OP_proj = OP - Vec3{0.0, -plane.y, 0.0};
		const bool is_z_even = rt_glsl_mod(std::floor(OP_proj.z / cell_size), 2.0) == 0.0;
		const bool is_x_even = rt_glsl_mod(std::floor(OP_proj.x / cell_size), 2.0) == 0.0;
		if ((is_z_even && is_x_even) || (!is_z_even && !is_x_even))
		{
			hit.material.color = Vec3{1.0, 0.0, 1.0};
		}
		return hit;
	}
	return rt_miss();
}

inline Intersect rt_intersect_sphere(const Ray& ray, const Sphere& sphere)
{
	const Vec3 oc = sphere.center - ray.origin;
	const double l = rt_dot(ray.direction, oc);
	const double det = l * l - rt_dot(oc, oc) + sphere.radius * sphere.radius;
	if (det < 0.0)
	{
		return rt_miss();
	}
	double len = l - std::sqrt(det);
	if (len < 0.0)
	{
		len = l + std::sqrt(det);
	}
	if (len < 0.0)
	{
		return rt_miss();
	}
	Intersect hit;
	hit.dist = len;
	hit.normal = rt_normalize(ray.origin + ray.direction * len - sphere.center);
	hit.material = sphere.material;
	return hit;
}

inline Intersect rt_trace(const Ray& ray, const Plane& plane, const Sphere spheres[3])
{
	Intersect hit = rt_intersect_plane(ray, plane);
	for (int i = 0; i < 3; ++i)
	{
		const Intersect sphere_hit = rt_intersect_sphere(ray, spheres[i]);
		if (sphere_hit.dist < hit.dist)
		{
			hit = sphere_hit;
		}
	}
	return hit;
}

inline double rt_fresnel(Vec3 ray_dir, Vec3 normal, double refr_koef)
{
	double cosi = rt_clamp(rt_dot(ray_dir, normal), -1.0, 1.0);
	double etai = 1.0;
	double etat = refr_koef;
	if (cosi > 0.0)
	{
		etai = etat;
		etat = 1.0;
	}
	const double sint = etai / etat * std::sqrt(std::max(0.0, 1.0 - cosi * cosi));
	if (sint >= 1.0)
	{
		return 1.0;
	}
	const double cost = std::sqrt(std::max(0.0, 1.0 - sint * sint));
	cosi = std::fabs(cosi);
	const double Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	const double Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rs * Rs + Rp * Rp) / 2.0;
}

inline Vec3 rt_refract(const Ray& ray, Vec3 normal, double ior)
{
	double cosi = rt_clamp(rt_dot(ray.direction, normal), -1.0, 1.0);
	double etai = 1.0;
	double etat = ior;
	if (cosi < 0.0)
	{
		cosi = -cosi;
	}
	else
	{
		etai = etat;
		etat = 1.0;
		normal = -normal;
	}
	const double eta = etai / etat;
	const double k = 1.0 - eta * eta * (1.0 - cosi * cosi);
	if (k < 0.0)
	{
		return Vec3{0.0, 0.0, 0.0};
	}
	return ray.direction * eta + normal * (eta * cosi - std::sqrt(k));
}

// Block (ii,jj)'s sole producer of its own r/g/b pixel tile - every pixel
// is an independent, local computation (no neighbor reads), so this is
// trivially safe to call from every block in parallel with no exchange
// step at all.
void calc_block(int ii, int jj, int bn, int bm, int gnx, int gny, double t, double angle,
                 int rn, int rm, double* r_out,
                 int gcn, int gcm, double* g_out,
                 int bcn, int bcm, double* b_out)
{
	(void)rn; (void)gcn; (void)bcn;

	const double r_radius = 17.0;
	const Vec3 sun_dir = rt_normalize(Vec3{2.0, -2.0, 2.0});

	const Plane plane{Vec3{0.0, 1.0, 0.0}, -5.0, Material{0.6, Vec3{0.0, 0.0, 0.0}}};
	const Sphere spheres[3] = {
		Sphere{2.0, Vec3{-7.0, -2.0 - std::cos(t), 0.0}, Material{20.7, Vec3{1.0, 0.0, 0.0}}},
		Sphere{3.0, Vec3{1.0, -1.0 + std::cos(t), 0.0}, Material{22.4, Vec3{0.0, 1.0, 0.0}}},
		Sphere{2.0, Vec3{8.0, -2.0 - std::cos(t), 0.0}, Material{25.9, Vec3{0.0, 0.0, 1.0}}},
	};

	const Vec3 ro{(1.0 + r_radius) * std::sin(angle), 0.0, (1.0 + r_radius) * std::cos(angle)};

	for (int li = 0; li < bn; ++li)
	{
		const int gi = ii * bn + li;
		for (int lj = 0; lj < bm; ++lj)
		{
			const int gj = jj * bm + lj;

			double u = (gi + 0.5) / gnx - 0.5;
			const double v = (gj + 0.5) / gny - 0.5;
			u *= static_cast<double>(gnx) / static_cast<double>(gny);

			const Ray ray{ro, rt_normalize(Vec3{u * std::cos(angle) + r_radius * std::sin(angle), v,
			                                     -u * std::sin(angle) + r_radius * std::cos(angle)} -
			                                ro)};

			const Intersect hit = rt_trace(ray, plane, spheres);

			Vec3 color{0.05, 0.05, 0.05};
			if (hit.dist < RT_MAX_DIST)
			{
				color = hit.material.color *
				        (rt_dot(hit.normal, -sun_dir) * (1.0 - std::pow(hit.dist / RT_MAX_DIST, 2.0)));
			}
			const Vec3 hit_point = ray.direction * (hit.dist * 0.9999) + ray.origin;

			const double kr = rt_fresnel(ray.direction, hit.normal, hit.material.refr_koef);

			// -- reflection (a single bounce, like the original's numOfIters=1) --
			Vec3 reflection_color{0.0, 0.0, 0.0};
			const Vec3 reflected_dir =
				hit.normal * (2.0 * std::fabs(rt_dot(-ray.direction, hit.normal))) + ray.direction;
			const Ray reflected_ray{hit_point, reflected_dir};
			const Intersect reflected_hit = rt_trace(reflected_ray, plane, spheres);
			if (reflected_hit.dist < RT_MAX_DIST)
			{
				reflection_color = reflected_hit.material.color *
				                   (0.8 * rt_dot(reflected_hit.normal, -sun_dir) *
				                    (1.0 - std::pow(reflected_hit.dist / RT_MAX_DIST, 2.0)));
			}

			// -- refraction (two bounces, like the original's numOfIters=2) --
			Vec3 refraction_color{0.0, 0.0, 0.0};
			if (kr < 1.0)
			{
				Intersect refr_hit = hit;
				Ray refr_ray = ray;
				for (int i = 0; i < 2; ++i)
				{
					const double bias_dir = (i % 2 == 0) ? -1.0 : 1.0;
					const Vec3 refr_point =
						refr_ray.direction * refr_hit.dist + refr_ray.origin + refr_hit.normal * (0.01 * bias_dir);
					const Vec3 refr_dir = rt_refract(refr_ray, refr_hit.normal, refr_hit.material.refr_koef);
					const Ray ray_to_refract{refr_point, refr_dir};

					refr_hit = rt_trace(ray_to_refract, plane, spheres);
					if (refr_hit.dist >= RT_MAX_DIST)
					{
						break;
					}
					refr_ray = Ray{refr_point, ray_to_refract.direction};
				}
				if (refr_hit.dist < RT_MAX_DIST)
				{
					refraction_color =
						refraction_color + refr_hit.material.color * (0.8 * rt_dot(refr_hit.normal, -sun_dir));
				}
			}

			color = color + (reflection_color * kr + refraction_color * (1.0 - kr)) * 0.8;

			// -- hard shadow --
			const Ray ray_to_sun{hit_point, -sun_dir};
			const Intersect hit_to_sun = rt_trace(ray_to_sun, plane, spheres);
			if (hit_to_sun.dist < RT_MAX_DIST)
			{
				color = color * 0.25;
			}

			const int idx = flat2(li, lj, rm);
			r_out[idx] = rt_clamp(color.x, 0.0, 1.0);
			g_out[idx] = rt_clamp(color.y, 0.0, 1.0);
			b_out[idx] = rt_clamp(color.z, 0.0, 1.0);
		}
	}
}

void print_frame(int frame)
{
	std::printf("frame=%d\n", frame);
	std::fflush(stdout);
}

/*=========================================================================
 * Live-image streaming to visualizer.py - same architecture (and the same
 * connect-once / mutex-guarded-send pattern) as examples/gas-ca and
 * examples/pic3d_v2's live_vis.py.
 *========================================================================*/
namespace raytracing_viz_detail
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

	const char* enable_env = std::getenv("RT_VIZ_ENABLE");
	if (!enable_env || std::string(enable_env) != "1")
	{
		return false;
	}

	const char* host_env = std::getenv("RT_VIZ_HOST");
	const char* port_env = std::getenv("RT_VIZ_PORT");
	const std::string host = host_env ? host_env : "127.0.0.1";
	const int port = port_env ? std::atoi(port_env) : 8767;

	const int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
	{
		std::fprintf(stderr, "raytracing viz: socket() failed, live visualization disabled\n");
		return false;
	}

	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(static_cast<uint16_t>(port));
	if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
	{
		std::fprintf(stderr, "raytracing viz: invalid RT_VIZ_HOST '%s'\n", host.c_str());
		close(fd);
		return false;
	}

	if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
	{
		std::fprintf(stderr,
		              "raytracing viz: could not connect to %s:%d - is visualizer.py "
		              "running? Live visualization disabled for this run.\n",
		              host.c_str(), port);
		close(fd);
		return false;
	}

	std::fprintf(stderr, "raytracing viz: connected to %s:%d\n", host.c_str(), port);
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

}  // namespace raytracing_viz_detail

// Wire format (must stay in sync with visualizer.py's HEADER_FORMAT/MAGIC):
// a 4-byte "RTIM" magic + 7x little-endian int32 (frame, ii, jj, bn, bm,
// gnx, gny), followed by bn*bm float64 r, then bn*bm float64 g, then
// bn*bm float64 b - all row-major [li][lj], same layout as calc_block
// writes them.
void send_image_block(int frame, int ii, int jj, int bn, int bm, int gnx, int gny,
                       int rn, int rm, const double* r,
                       int gcn, int gcm, const double* g,
                       int bcn, int bcm, const double* b)
{
	(void)rn; (void)rm; (void)gcn; (void)gcm; (void)bcn; (void)bcm;

	std::lock_guard<std::mutex> lock(raytracing_viz_detail::send_mutex());

	if (!raytracing_viz_detail::ensure_connected())
	{
		return;
	}

	char header[32];
	std::memcpy(header, "RTIM", 4);
	const int32_t fields[7] = {frame, ii, jj, bn, bm, gnx, gny};
	std::memcpy(header + 4, fields, sizeof(fields));

	const std::size_t block_bytes = static_cast<std::size_t>(bn) * bm * sizeof(double);
	const bool ok = raytracing_viz_detail::send_all(header, sizeof(header)) &&
	                raytracing_viz_detail::send_all(r, block_bytes) &&
	                raytracing_viz_detail::send_all(g, block_bytes) &&
	                raytracing_viz_detail::send_all(b, block_bytes);
	if (!ok)
	{
		std::fprintf(stderr, "raytracing viz: send failed, dropping connection\n");
		close(raytracing_viz_detail::socket_fd());
		raytracing_viz_detail::socket_fd() = -1;
	}
}
