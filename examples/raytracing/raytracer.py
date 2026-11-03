from pysol import (
    Array, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config,
)

# =============================================================================
# pysol port of examples/raytracing/shader.gl - a Shadertoy-style analytic
# ray tracer (ground plane + 3 animated spheres, one reflection bounce, two
# refraction bounces mixed by Fresnel, a hard shadow ray). See ucodes.h's
# module comment for the full port notes.
#
# calc_block is the only kernel doing real work here - it is the direct
# analogue of the shader's mainImage, one call per block, each pixel
# computed independently of every other pixel (no stencil, so no
# pysol_update_array anywhere in this example - there is nothing to
# exchange between blocks).
#
# The animation itself - the shader's free-running `iTime` - is replaced by
# this file's own main loop: each frame advances `t` by a fixed `dt` and
# passes `t` (sphere bobbing) and `rk * t` (camera orbit angle) into
# calc_block, i.e. the camera rotates a little further around the scene
# every frame.
# =============================================================================


@pysol
def raytracer():
    calc_k = import_kernel("calc_block",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("bn", True, Type.Int),
                             KernelArgument("bm", True, Type.Int),
                             KernelArgument("gnx", True, Type.Int),
                             KernelArgument("gny", True, Type.Int),
                             KernelArgument("t", True, Type.Double),
                             KernelArgument("angle", True, Type.Double),
                             KernelArgument("r", False, Type.Double, 2),
                             KernelArgument("g", False, Type.Double, 2),
                             KernelArgument("b", False, Type.Double, 2)])

    print_frame_k = import_kernel("print_frame", [KernelArgument("frame", True, Type.Int)])

    # Streams block (ii,jj)'s freshly computed RGB tile to visualizer.py
    # over a TCP socket the instant it's produced - see send_image_block's
    # comment in ucodes.h for the wire format and the RT_VIZ_* environment
    # variables.
    send_viz_k = import_kernel("send_image_block",
                                [KernelArgument("frame", True, Type.Int),
                                 KernelArgument("ii", True, Type.Int),
                                 KernelArgument("jj", True, Type.Int),
                                 KernelArgument("bn", True, Type.Int),
                                 KernelArgument("bm", True, Type.Int),
                                 KernelArgument("gnx", True, Type.Int),
                                 KernelArgument("gny", True, Type.Int),
                                 KernelArgument("r", True, Type.Double, 2),
                                 KernelArgument("g", True, Type.Double, 2),
                                 KernelArgument("b", True, Type.Double, 2)])

    # -- image/block parameters --
    # 800x450 (16:9), a 4x3 block grid - 12 fragments for OMP parallelism,
    # same idea as examples/gas-ca's 3x3.
    nb : Type.Int = 4
    mb : Type.Int = 3
    bn : Type.Int = 200
    bm : Type.Int = 150
    gnx : Type.Int = nb * bn
    gny : Type.Int = mb * bm

    # Meant to run continuously at an exhibition, like examples/gas-ca -
    # frame only grows by 1 per iteration (not 2, unlike gas_ca's ping-
    # ponged array version), so this is nowhere near int32 overflow.
    num_frames : Type.Int = 500000000

    dt : Type.Double = 0.02
    rk : Type.Double = 0.4

    # -- arrays --
    # is_single_version=True: only the latest frame's pixels are kept
    # resident (this runs for a very long time, and a plain Array would
    # otherwise retain every historical frame's full image forever).
    r = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)
    g = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)
    b = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)

    # -- main render loop --
    # A `while` loop, not "for frame in range(num_frames):" - matches
    # examples/gas-ca's and examples/lattice-gas's own main loops (a `for`
    # loop wrapping a kernel call on a 2D array hangs after the first
    # outer iteration; `while` does not).
    frame : Type.Int = 0
    t : Type.Double = 0.0
    while frame <= num_frames:
        for i in range(nb):
            for j in range(mb):
                calc_k(i, j, bn, bm, gnx, gny, t, rk * t,
                       r[frame, i][j], g[frame, i][j], b[frame, i][j])

        print_frame_k(frame)

        for i in range(nb):
            for j in range(mb):
                send_viz_k(frame, i, j, bn, bm, gnx, gny,
                           r[frame, i][j], g[frame, i][j], b[frame, i][j])

        t = t + dt
        frame = frame + 1


if __name__ == "__main__":
    load_config("config.json")

    print("==========RAYTRACING (pysol port of shader.gl)============")
    pysol_compile("raytracer")
    pysol_run("raytracer")
