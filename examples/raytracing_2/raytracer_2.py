from pysol import (
    Array, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config,
)

# =============================================================================
# pysol port of examples/raytracing_2/shader.glsl - a Phong-lighting ray
# tracer (2 planes, 2 animated spheres, 2 point lights, reflection bounces).
# See ucodes.h's module comment for the full port notes; this mirrors
# examples/raytracing's raytracer.py structure.
#
# calc_block is the direct analogue of the shader's mainImage, one call per
# block, every pixel independent of every other pixel - no stencil, so no
# pysol_update_array anywhere in this example either.
#
# The camera is static in the original shader; all motion comes from the
# two spheres' positions (already functions of iTime in InitScene). This
# file's main loop advances `t` by a fixed `dt` every frame and passes it
# into calc_block in place of iTime.
# =============================================================================


@pysol
def raytracer_2():
    calc_k = import_kernel("calc_block",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("bn", True, Type.Int),
                             KernelArgument("bm", True, Type.Int),
                             KernelArgument("gnx", True, Type.Int),
                             KernelArgument("gny", True, Type.Int),
                             KernelArgument("t", True, Type.Double),
                             KernelArgument("r", False, Type.Double, 2),
                             KernelArgument("g", False, Type.Double, 2),
                             KernelArgument("b", False, Type.Double, 2)])

    print_frame_k = import_kernel("print_frame", [KernelArgument("frame", True, Type.Int)])

    # Streams block (ii,jj)'s freshly computed RGB tile to visualizer.py
    # over a TCP socket the instant it's produced - see send_image_block's
    # comment in ucodes.h for the wire format and the RT2_VIZ_* environment
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
    # 800x450 (16:9), a 4x3 block grid - same layout as examples/raytracing.
    nb : Type.Int = 4
    mb : Type.Int = 3
    bn : Type.Int = 200
    bm : Type.Int = 150
    gnx : Type.Int = nb * bn
    gny : Type.Int = mb * bm

    # Meant to run continuously at an exhibition, like examples/raytracing -
    # frame only grows by 1 per iteration, nowhere near int32 overflow.
    num_frames : Type.Int = 500000000

    dt : Type.Double = 0.02

    # -- arrays --
    # is_single_version=True: only the latest frame's pixels are kept
    # resident (this runs for a very long time, and a plain Array would
    # otherwise retain every historical frame's full image forever).
    r = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)
    g = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)
    b = Array(2, 2, [nb, mb], [bn, bm], Type.Double, True)

    # -- main render loop --
    # A `while` loop, not "for frame in range(num_frames):" - matches
    # examples/raytracing's and examples/gas-ca's own main loops (a `for`
    # loop wrapping a kernel call on a 2D array hangs after the first
    # outer iteration; `while` does not).
    frame : Type.Int = 0
    t : Type.Double = 0.0
    while frame <= num_frames:
        for i in range(nb):
            for j in range(mb):
                calc_k(i, j, bn, bm, gnx, gny, t, r[frame, i][j], g[frame, i][j], b[frame, i][j])

        print_frame_k(frame)

        for i in range(nb):
            for j in range(mb):
                send_viz_k(frame, i, j, bn, bm, gnx, gny,
                           r[frame, i][j], g[frame, i][j], b[frame, i][j])

        t = t + dt
        frame = frame + 1


if __name__ == "__main__":
    load_config("config.json")

    print("==========RAYTRACING_2 (pysol port of shader.glsl)============")
    pysol_compile("raytracer_2")
    pysol_run("raytracer_2")
