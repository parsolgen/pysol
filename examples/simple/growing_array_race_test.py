import os
import sys

import numpy as np

from pysol import (
    GrowingArray, KernelArgument, Type, import_kernel, pysol,
    pysol_compile, pysol_run, load_config, pysol_struct,
)

# =============================================================================
# Minimal, standalone reproduction of pic3d_v2's 27-direction GrowingArray
# particle-mailbox migration pattern (push_particles_block/
# merge_particles_block in examples/pic3d_v2/ucodes.h, driven from
# examples/pic3d_v2/pic3d.py), with NO physics beyond constant-velocity
# ballistic motion + wall reflection - just enough to drive the SAME
# cross-fragment GrowingArray read/write pattern that pic3d_v2's module
# docstring documents as a confirmed-but-unfixed RTS-level race (see
# examples/pic3d_v2/pic3d.py's "Known limitation" note, bug (2)).
#
# Reproduces the real bug's triggering conditions as closely as practical at
# a much smaller scale:
#   - Same 10x10x10 = 1000 fragment grid as pic3d_v2's full-scale
#     configuration - the exact grid size bug (2) was confirmed at; fewer
#     fragments were NOT observed to reproduce it there.
#   - Same severe initial load imbalance: every token starts at z=CZ
#     exactly, so the entire population sits in one single z-layer of
#     fragments (k=5) at ts=0, just like pic3d_v2's generate_particles_disk_k
#     (every particle starts at z=cz there too).
#   - Same 27-mailbox GrowingArray push+merge pattern, including the
#     default-then-override im1/jm1/km1 neighbor-index idiom and the
#     has_lo_*/has_hi_*/exists_* gating pic3d_v2 (and pysol's other pic3d
#     port) use for every cross-fragment exchange.
#
# Deliberately NOT reproduced (irrelevant to the race, dropped to keep this
# test fast and independently checkable):
#   - Any force field / density / potential solve - tokens move at constant,
#     per-token velocity assigned once at init from a simple closed-form
#     function of each token's id (see ucodes.h's init_tokens_block), so
#     every token's exact trajectory is a plain, bit-reproducible double-
#     precision calculation that a pure-Python "ground truth" (see
#     compute_ground_truth()/verify() below) can replay exactly, step for
#     step, and compare against what the compiled, multithreaded pysol
#     program actually produced. Any mismatch can only be the GrowingArray
#     exchange itself - never a physics or floating-point difference between
#     two different implementations of the same formula, since both sides
#     use the exact same sequence of +, -, *, / operations in the exact same
#     order (bit-identical under IEEE-754 on the same machine).
#
# Usage: this is a standalone reproduction/debugging tool, not a pass/fail
# gate in the usual CI sense. Run it once at OMP_NUM_THREADS=1 (must PASS -
# a single thread cannot race with itself) and once at a higher thread count
# (e.g. 10, matching the user's original bug report) IN A SEPARATE PROCESS:
# OpenMP/libgomp reads OMP_NUM_THREADS once per process, lazily, at the
# first parallel region it executes, so comparing thread counts needs two
# separate `python3 growing_array_race_test.py` invocations, not two calls
# in one process. The race is intermittent even at OMP_NUM_THREADS=10 - a
# single PASS there does not clear it; run it several times.
#
#   OMP_NUM_THREADS=1  python3 growing_array_race_test.py   # always PASS
#   OMP_NUM_THREADS=10 python3 growing_array_race_test.py   # run a few times - intermittent FAIL
#
# Confirmed with these exact constants (NFX=NFY=NFZ=10, NPX_TOKENS=NPY_
# TOKENS=1130 -> 1,000,996 tokens, matching pic3d_v2's own full production
# scale exactly): OMP_NUM_THREADS=1 passed every time across repeated runs;
# OMP_NUM_THREADS=10 passed on some runs and failed on others, losing
# anywhere from ~800 to ~17,500 tokens (always "missing", never duplicated)
# spread across dozens of fragments that differ from run to run - genuine,
# timing-dependent nondeterminism, not a fixed logic error.
#
# VMAX is deliberately small (0.15, not something larger like 3.0) - see
# ucodes.h's comment on merge_tokens_block for why: a LARGER VMAX lets
# tokens reach the outermost fragment-grid edge (ii/jj/kk == 0 or ==
# NFX_LAST) within NUM_TS steps, which triggers a SEPARATE, 100%
# deterministic (reproducible even at OMP_NUM_THREADS=1) token-loss bug
# unrelated to the threaded race this file targets. VMAX=0.15 keeps every
# token's worst-case displacement (VMAX*TAU*NUM_TS = 0.15*0.01*100 = 0.15)
# safely under the 0.2 margin between the token disk's edge (radius 0.2
# from the domain center) and the nearest fragment-grid edge, so that
# bug never triggers here and this test isolates only the real race.
# =============================================================================

NFX, NFY, NFZ = 10, 10, 10
XN, YN, ZN = 1.0, 1.0, 1.0
HX, HY, HZ = XN / NFX, YN / NFY, ZN / NFZ
CX, CY, CZ = 0.5, 0.5, 0.5
RADIUS = 0.2
NPX_TOKENS, NPY_TOKENS = 1130, 1130
TAU = 0.01
VMAX = 0.15
NUM_TS = 100


@pysol_struct
class Token:
    id: Type.Int
    x: Type.Double
    y: Type.Double
    z: Type.Double
    u: Type.Double
    v: Type.Double
    w: Type.Double


@pysol
def growing_array_race_test():
    # ── kernel imports ───────────────────────────────────────────────────
    hello_world_k = import_kernel("hello_world_k", [])

    init_k = import_kernel("init_tokens_block",
                            [KernelArgument("cx", True, Type.Double),
                             KernelArgument("cy", True, Type.Double),
                             KernelArgument("cz", True, Type.Double),
                             KernelArgument("radius", True, Type.Double),
                             KernelArgument("hx", True, Type.Double),
                             KernelArgument("hy", True, Type.Double),
                             KernelArgument("hz", True, Type.Double),
                             KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("npx", True, Type.Int),
                             KernelArgument("npy", True, Type.Int),
                             KernelArgument("vmax", True, Type.Double),
                             KernelArgument("cell", False, Token)])

    push_k = import_kernel("push_tokens_block",
                            [KernelArgument("tau", True, Type.Double),
                             KernelArgument("xn", True, Type.Double),
                             KernelArgument("yn", True, Type.Double),
                             KernelArgument("zn", True, Type.Double),
                             KernelArgument("hx", True, Type.Double),
                             KernelArgument("hy", True, Type.Double),
                             KernelArgument("hz", True, Type.Double),
                             KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("cell_old", True, Token),
                             KernelArgument("mail_self", False, Token),
                             KernelArgument("mail_000", False, Token),
                             KernelArgument("mail_001", False, Token),
                             KernelArgument("mail_002", False, Token),
                             KernelArgument("mail_010", False, Token),
                             KernelArgument("mail_011", False, Token),
                             KernelArgument("mail_012", False, Token),
                             KernelArgument("mail_020", False, Token),
                             KernelArgument("mail_021", False, Token),
                             KernelArgument("mail_022", False, Token),
                             KernelArgument("mail_100", False, Token),
                             KernelArgument("mail_101", False, Token),
                             KernelArgument("mail_102", False, Token),
                             KernelArgument("mail_110", False, Token),
                             KernelArgument("mail_112", False, Token),
                             KernelArgument("mail_120", False, Token),
                             KernelArgument("mail_121", False, Token),
                             KernelArgument("mail_122", False, Token),
                             KernelArgument("mail_200", False, Token),
                             KernelArgument("mail_201", False, Token),
                             KernelArgument("mail_202", False, Token),
                             KernelArgument("mail_210", False, Token),
                             KernelArgument("mail_211", False, Token),
                             KernelArgument("mail_212", False, Token),
                             KernelArgument("mail_220", False, Token),
                             KernelArgument("mail_221", False, Token),
                             KernelArgument("mail_222", False, Token)])

    merge_k = import_kernel("merge_tokens_block",
                             [KernelArgument("from_self", True, Token),
                              KernelArgument("mail000", True, Token), KernelArgument("exists000", True, Type.Int),
                              KernelArgument("mail001", True, Token), KernelArgument("exists001", True, Type.Int),
                              KernelArgument("mail002", True, Token), KernelArgument("exists002", True, Type.Int),
                              KernelArgument("mail010", True, Token), KernelArgument("exists010", True, Type.Int),
                              KernelArgument("mail011", True, Token), KernelArgument("exists011", True, Type.Int),
                              KernelArgument("mail012", True, Token), KernelArgument("exists012", True, Type.Int),
                              KernelArgument("mail020", True, Token), KernelArgument("exists020", True, Type.Int),
                              KernelArgument("mail021", True, Token), KernelArgument("exists021", True, Type.Int),
                              KernelArgument("mail022", True, Token), KernelArgument("exists022", True, Type.Int),
                              KernelArgument("mail100", True, Token), KernelArgument("exists100", True, Type.Int),
                              KernelArgument("mail101", True, Token), KernelArgument("exists101", True, Type.Int),
                              KernelArgument("mail102", True, Token), KernelArgument("exists102", True, Type.Int),
                              KernelArgument("mail110", True, Token), KernelArgument("exists110", True, Type.Int),
                              KernelArgument("mail112", True, Token), KernelArgument("exists112", True, Type.Int),
                              KernelArgument("mail120", True, Token), KernelArgument("exists120", True, Type.Int),
                              KernelArgument("mail121", True, Token), KernelArgument("exists121", True, Type.Int),
                              KernelArgument("mail122", True, Token), KernelArgument("exists122", True, Type.Int),
                              KernelArgument("mail200", True, Token), KernelArgument("exists200", True, Type.Int),
                              KernelArgument("mail201", True, Token), KernelArgument("exists201", True, Type.Int),
                              KernelArgument("mail202", True, Token), KernelArgument("exists202", True, Type.Int),
                              KernelArgument("mail210", True, Token), KernelArgument("exists210", True, Type.Int),
                              KernelArgument("mail211", True, Token), KernelArgument("exists211", True, Type.Int),
                              KernelArgument("mail212", True, Token), KernelArgument("exists212", True, Type.Int),
                              KernelArgument("mail220", True, Token), KernelArgument("exists220", True, Type.Int),
                              KernelArgument("mail221", True, Token), KernelArgument("exists221", True, Type.Int),
                              KernelArgument("mail222", True, Token), KernelArgument("exists222", True, Type.Int),
                              KernelArgument("cell_new", False, Token)])

    dump_k = import_kernel("dump_token_ids_block",
                            [KernelArgument("ii", True, Type.Int),
                             KernelArgument("jj", True, Type.Int),
                             KernelArgument("kk", True, Type.Int),
                             KernelArgument("cell", True, Token)])

    # ── body ─────────────────────────────────────────────────────────────
    hello_world_k()

    nfx : Type.Int = 10
    nfy : Type.Int = 10
    nfz : Type.Int = 10
    nfx_last : Type.Int = nfx - 1
    nfy_last : Type.Int = nfy - 1
    nfz_last : Type.Int = nfz - 1

    xn : Type.Double = 1.0
    yn : Type.Double = 1.0
    zn : Type.Double = 1.0
    hx : Type.Double = 0.1
    hy : Type.Double = 0.1
    hz : Type.Double = 0.1
    cx : Type.Double = 0.5
    cy : Type.Double = 0.5
    cz : Type.Double = 0.5
    radius : Type.Double = 0.2
    npx_tokens : Type.Int = 1130
    npy_tokens : Type.Int = 1130
    tau : Type.Double = 0.01
    vmax : Type.Double = 0.15
    num_ts : Type.Int = 100

    tokens = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_self = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_000 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_001 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_002 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_010 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_011 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_012 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_020 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_021 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_022 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_100 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_101 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_102 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_110 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_112 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_120 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_121 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_122 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_200 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_201 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_202 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_210 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_211 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_212 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_220 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_221 = GrowingArray(3, [nfx, nfy, nfz], Token)
    mail_222 = GrowingArray(3, [nfx, nfy, nfz], Token)

    # -- initial tokens: one lattice-in-disk, every fragment filtering the
    # SAME global candidate lattice down to its own owned box, all at
    # z=cz exactly - the entire population starts in the single k=5
    # fragment layer, mirroring pic3d_v2's own severe load imbalance --
    for i in range(nfx):
        for j in range(nfy):
            for k in range(nfz):
                init_k(cx, cy, cz, radius, hx, hy, hz, i, j, k,
                       npx_tokens, npy_tokens, vmax, tokens[0, i][j][k])

    ts : Type.Int = 0
    pv : Type.Int = 0
    while ts < num_ts:
        for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    push_k(tau, xn, yn, zn, hx, hy, hz, i, j, k,
                           tokens[pv, i][j][k],
                           mail_self[pv + 1, i][j][k],
                           mail_000[pv + 1, i][j][k],
                           mail_001[pv + 1, i][j][k],
                           mail_002[pv + 1, i][j][k],
                           mail_010[pv + 1, i][j][k],
                           mail_011[pv + 1, i][j][k],
                           mail_012[pv + 1, i][j][k],
                           mail_020[pv + 1, i][j][k],
                           mail_021[pv + 1, i][j][k],
                           mail_022[pv + 1, i][j][k],
                           mail_100[pv + 1, i][j][k],
                           mail_101[pv + 1, i][j][k],
                           mail_102[pv + 1, i][j][k],
                           mail_110[pv + 1, i][j][k],
                           mail_112[pv + 1, i][j][k],
                           mail_120[pv + 1, i][j][k],
                           mail_121[pv + 1, i][j][k],
                           mail_122[pv + 1, i][j][k],
                           mail_200[pv + 1, i][j][k],
                           mail_201[pv + 1, i][j][k],
                           mail_202[pv + 1, i][j][k],
                           mail_210[pv + 1, i][j][k],
                           mail_211[pv + 1, i][j][k],
                           mail_212[pv + 1, i][j][k],
                           mail_220[pv + 1, i][j][k],
                           mail_221[pv + 1, i][j][k],
                           mail_222[pv + 1, i][j][k])

        for i in range(nfx):
            for j in range(nfy):
                for k in range(nfz):
                    has_lo_x : Type.Int = 1
                    has_hi_x : Type.Int = 1
                    has_lo_y : Type.Int = 1
                    has_hi_y : Type.Int = 1
                    has_lo_z : Type.Int = 1
                    has_hi_z : Type.Int = 1
                    if i == 0:
                        has_lo_x = 0
                    if i == nfx_last:
                        has_hi_x = 0
                    if j == 0:
                        has_lo_y = 0
                    if j == nfy_last:
                        has_hi_y = 0
                    if k == 0:
                        has_lo_z = 0
                    if k == nfz_last:
                        has_hi_z = 0
                    im1 : Type.Int = i - 1
                    ip1 : Type.Int = i + 1
                    jm1 : Type.Int = j - 1
                    jp1 : Type.Int = j + 1
                    km1 : Type.Int = k - 1
                    kp1 : Type.Int = k + 1
                    if i == 0:
                        im1 = i
                    if i == nfx_last:
                        ip1 = i
                    if j == 0:
                        jm1 = j
                    if j == nfy_last:
                        jp1 = j
                    if k == 0:
                        km1 = k
                    if k == nfz_last:
                        kp1 = k
                    exists_000 : Type.Int = has_lo_x * has_lo_y * has_lo_z
                    exists_001 : Type.Int = has_lo_x * has_lo_y
                    exists_002 : Type.Int = has_lo_x * has_lo_y * has_hi_z
                    exists_010 : Type.Int = has_lo_x * has_lo_z
                    exists_011 : Type.Int = has_lo_x
                    exists_012 : Type.Int = has_lo_x * has_hi_z
                    exists_020 : Type.Int = has_lo_x * has_hi_y * has_lo_z
                    exists_021 : Type.Int = has_lo_x * has_hi_y
                    exists_022 : Type.Int = has_lo_x * has_hi_y * has_hi_z
                    exists_100 : Type.Int = has_lo_y * has_lo_z
                    exists_101 : Type.Int = has_lo_y
                    exists_102 : Type.Int = has_lo_y * has_hi_z
                    exists_110 : Type.Int = has_lo_z
                    exists_112 : Type.Int = has_hi_z
                    exists_120 : Type.Int = has_hi_y * has_lo_z
                    exists_121 : Type.Int = has_hi_y
                    exists_122 : Type.Int = has_hi_y * has_hi_z
                    exists_200 : Type.Int = has_hi_x * has_lo_y * has_lo_z
                    exists_201 : Type.Int = has_hi_x * has_lo_y
                    exists_202 : Type.Int = has_hi_x * has_lo_y * has_hi_z
                    exists_210 : Type.Int = has_hi_x * has_lo_z
                    exists_211 : Type.Int = has_hi_x
                    exists_212 : Type.Int = has_hi_x * has_hi_z
                    exists_220 : Type.Int = has_hi_x * has_hi_y * has_lo_z
                    exists_221 : Type.Int = has_hi_x * has_hi_y
                    exists_222 : Type.Int = has_hi_x * has_hi_y * has_hi_z

                    merge_k(mail_self[pv + 1, i][j][k],
                            mail_222[pv + 1, im1][jm1][km1], exists_000,
                            mail_221[pv + 1, im1][jm1][k], exists_001,
                            mail_220[pv + 1, im1][jm1][kp1], exists_002,
                            mail_212[pv + 1, im1][j][km1], exists_010,
                            mail_211[pv + 1, im1][j][k], exists_011,
                            mail_210[pv + 1, im1][j][kp1], exists_012,
                            mail_202[pv + 1, im1][jp1][km1], exists_020,
                            mail_201[pv + 1, im1][jp1][k], exists_021,
                            mail_200[pv + 1, im1][jp1][kp1], exists_022,
                            mail_122[pv + 1, i][jm1][km1], exists_100,
                            mail_121[pv + 1, i][jm1][k], exists_101,
                            mail_120[pv + 1, i][jm1][kp1], exists_102,
                            mail_112[pv + 1, i][j][km1], exists_110,
                            mail_110[pv + 1, i][j][kp1], exists_112,
                            mail_102[pv + 1, i][jp1][km1], exists_120,
                            mail_101[pv + 1, i][jp1][k], exists_121,
                            mail_100[pv + 1, i][jp1][kp1], exists_122,
                            mail_022[pv + 1, ip1][jm1][km1], exists_200,
                            mail_021[pv + 1, ip1][jm1][k], exists_201,
                            mail_020[pv + 1, ip1][jm1][kp1], exists_202,
                            mail_012[pv + 1, ip1][j][km1], exists_210,
                            mail_011[pv + 1, ip1][j][k], exists_211,
                            mail_010[pv + 1, ip1][j][kp1], exists_212,
                            mail_002[pv + 1, ip1][jp1][km1], exists_220,
                            mail_001[pv + 1, ip1][jp1][k], exists_221,
                            mail_000[pv + 1, ip1][jp1][kp1], exists_222,
                            tokens[pv + 2, i][j][k])
        ts = ts + 1
        pv = pv + 2

    for i in range(nfx):
        for j in range(nfy):
            for k in range(nfz):
                dump_k(i, j, k, tokens[pv, i][j][k])


# ── ground truth (plain Python/numpy, no pysol) ─────────────────────────────

def compute_ground_truth():
    """Replays every token's exact trajectory (constant velocity + wall
    reflection + current-fragment-relative routing) using the SAME sequence
    of double-precision operations init_tokens_block/push_tokens_block use,
    so the result is bit-for-bit what a race-free run must produce."""
    npx, npy = NPX_TOKENS, NPY_TOKENS
    dlx = (2.0 * RADIUS) / (npx - 1)
    dly = (2.0 * RADIUS) / (npy - 1)

    ii_idx, jj_idx = np.meshgrid(np.arange(npx), np.arange(npy), indexing="ij")
    x = CX - RADIUS + ii_idx.astype(np.float64) * dlx
    y = CY - RADIUS + jj_idx.astype(np.float64) * dly
    dx = x - CX
    dy = y - CY
    mask = (dx * dx + dy * dy) <= RADIUS * RADIUS

    ids = (ii_idx * npy + jj_idx)[mask].astype(np.int64)
    x = x[mask].copy()
    y = y[mask].copy()
    z = np.full_like(x, CZ)

    mod7 = (ids % 7).astype(np.float64)
    mod11 = (ids % 11).astype(np.float64)
    mod13 = (ids % 13).astype(np.float64)
    u = VMAX * (2.0 * (mod7 / 6.0) - 1.0)
    v = VMAX * (2.0 * (mod11 / 10.0) - 1.0)
    w = VMAX * (2.0 * (mod13 / 12.0) - 1.0)

    def locate(coord, h, n):
        fi = np.clip((coord / h).astype(np.int64), 0, n - 1)
        # Self-correct any division/multiplication rounding mismatch so the
        # result always satisfies fi*h <= coord < (fi+1)*h - the same
        # multiplication-based boundary comparison init_tokens_block/
        # push_tokens_block use, never a floor-division shortcut.
        for _ in range(2):
            fi = np.where((fi > 0) & (coord < fi * h), fi - 1, fi)
            fi = np.where((fi < n - 1) & (coord >= (fi + 1) * h), fi + 1, fi)
        return fi

    fi = locate(x, HX, NFX)
    fj = locate(y, HY, NFY)
    fk = locate(z, HZ, NFZ)

    for _ in range(NUM_TS):
        x = x + TAU * u
        y = y + TAU * v
        z = z + TAU * w

        neg = x < 0.0
        x = np.where(neg, -x, x)
        u = np.where(neg, -u, u)
        hi = (~neg) & (x >= XN)
        x = np.where(hi, 2.0 * XN - x, x)
        u = np.where(hi, -u, u)

        neg = y < 0.0
        y = np.where(neg, -y, y)
        v = np.where(neg, -v, v)
        hi = (~neg) & (y >= YN)
        y = np.where(hi, 2.0 * YN - y, y)
        v = np.where(hi, -v, v)

        neg = z < 0.0
        z = np.where(neg, -z, z)
        w = np.where(neg, -w, w)
        hi = (~neg) & (z >= ZN)
        z = np.where(hi, 2.0 * ZN - z, z)
        w = np.where(hi, -w, w)

        x1, x2 = fi * HX, fi * HX + HX
        y1, y2 = fj * HY, fj * HY + HY
        z1, z2 = fk * HZ, fk * HZ + HZ

        ddx = np.where(x < x1, 0, np.where(x >= x2, 2, 1))
        ddy = np.where(y < y1, 0, np.where(y >= y2, 2, 1))
        ddz = np.where(z < z1, 0, np.where(z >= z2, 2, 1))

        fi = fi + (ddx - 1)
        fj = fj + (ddy - 1)
        fk = fk + (ddz - 1)

    return ids, fi, fj, fk


def verify():
    """Compares every fragment's actual dumped token-id set (written by
    dump_token_ids_block after the pysol run) against compute_ground_truth()'s
    independently-computed expectation. Any fragment where the two sets
    differ is a token the GrowingArray exchange lost, duplicated, or
    misrouted under multithreading - i.e. the race, reproduced."""
    ids, fi, fj, fk = compute_ground_truth()

    expected = {}
    for n in range(len(ids)):
        key = (int(fi[n]), int(fj[n]), int(fk[n]))
        expected.setdefault(key, set()).add(int(ids[n]))

    actual = {}
    total_actual = 0
    for i in range(NFX):
        for j in range(NFY):
            for k in range(NFZ):
                path = f"race_ids_{i}_{j}_{k}.bin"
                if not os.path.exists(path):
                    continue
                data = np.fromfile(path, dtype=np.int32)
                if data.size:
                    actual[(i, j, k)] = set(int(v) for v in data)
                    total_actual += data.size

    total_expected = len(ids)
    all_keys = set(expected) | set(actual)
    mismatched = []
    for key in sorted(all_keys):
        exp = expected.get(key, set())
        act = actual.get(key, set())
        if exp != act:
            mismatched.append((key, exp - act, act - exp))

    if mismatched:
        n_missing = sum(len(m) for _, m, _ in mismatched)
        n_extra = sum(len(e) for _, _, e in mismatched)
        print(f"RACE REPRODUCED: {len(mismatched)}/{len(all_keys)} fragment(s) "
              f"disagree with the deterministic ground truth "
              f"(expected {total_expected} tokens total, actually dumped "
              f"{total_actual}; {n_missing} missing, {n_extra} extra/duplicated):")
        for key, missing, extra in mismatched[:20]:
            print(f"  fragment {key}: missing={sorted(missing)[:5]}"
                  f"{'...' if len(missing) > 5 else ''} "
                  f"extra={sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
        if len(mismatched) > 20:
            print(f"  ... and {len(mismatched) - 20} more mismatched fragment(s)")
        sys.exit(1)

    print(f"PASS: all {total_expected} tokens landed in their deterministically "
          f"expected fragment - no GrowingArray race detected at this thread count "
          f"(OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')})")


if __name__ == "__main__":
    load_config("config.json")

    print("==========GROWING ARRAY RACE REPRODUCTION TEST============")
    pysol_compile("growing_array_race_test")
    pysol_run("growing_array_race_test")
    verify()
