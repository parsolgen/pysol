"""Live velocity-field visualizer for lattice_gas.py - plain pygame +
sockets (same libraries as examples/pic3d_v2/live_vis.py), no matplotlib.

Run this first, then run lattice_gas.py with:

    LATTICE_GAS_VIZ_ENABLE=1 python3 lattice_gas.py

lattice_gas.py's send_velocity_block kernel (see ucodes.h) opens ONE
persistent connection for the whole run and streams every block's
velocity (ux, uy) over it the instant it's computed. This script keeps
a shared ux/uy grid updated from that stream on a background thread,
and redraws from whatever's currently in it on a plain 30fps loop - the
flow is visibly developing while the simulation is still running.

Usage:
    python3 visualizer.py [port]        # default 8765, matches
                                         # lattice_gas.py's
                                         # LATTICE_GAS_VIZ_PORT default

Wire format (must stay in sync with send_velocity_block in ucodes.h):
    header   = 4 bytes b"LGVP" + 7x little-endian int32:
               step, ii, jj, bn, bm, gnx, gny
    payload  = bn*bm little-endian float64 (ux block, row-major [li][lj])
             + bn*bm little-endian float64 (uy block, same layout)
"""
import socket
import struct
import sys
import threading

import numpy as np
import pygame

HOST = "0.0.0.0"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

HEADER_FORMAT = "<4siiiiiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"LGVP"

# Global ux/uy grids - (re)allocated lazily from the first header's
# (gnx, gny), or reallocated if a later one reports a different size
# (a fresh simulation run may use different parameters than the last).
# Stored as (gny, gnx), matching the wire layout's [jj][ii] block tiling;
# transposed to pygame's (x, y) order only at render time.
ux_grid = None
uy_grid = None
grid_lock = threading.Lock()


def recv_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


# Diverging blue -> white -> red map (same idea as the original
# matplotlib "RdBu_r"), scaled symmetrically around zero by this
# field's own current largest magnitude.
NEG = np.array([30, 60, 200], dtype=np.float32)
MID = np.array([245, 245, 245], dtype=np.float32)
POS = np.array([200, 30, 30], dtype=np.float32)


def velocity_to_rgb(values):
    vmax = max(float(np.max(np.abs(values))), 1e-12)
    t = np.clip(values / vmax, -1, 1)
    neg_t = np.clip(-t, 0, 1)[..., None]
    pos_t = np.clip(t, 0, 1)[..., None]
    rgb = MID + neg_t * (NEG - MID) + pos_t * (POS - MID)
    return rgb.astype(np.uint8)


def process_connection(conn):
    global ux_grid, uy_grid
    print(f"visualizer.py: connected from {conn.getpeername()}")
    try:
        while True:
            header = recv_exact(conn, HEADER_SIZE)
            if header is None:
                print("visualizer.py: sender disconnected, waiting for a new connection...")
                return
            magic, step, ii, jj, bn, bm, gnx, gny = struct.unpack(HEADER_FORMAT, header)
            if magic != MAGIC:
                print("visualizer.py: bad magic bytes, dropping this connection")
                return

            count = bn * bm
            payload = recv_exact(conn, count * 8 * 2)
            if payload is None:
                print("visualizer.py: sender disconnected mid-block")
                return

            ux_block = np.frombuffer(payload, dtype="<f8", count=count).reshape(bn, bm)
            uy_block = np.frombuffer(payload, dtype="<f8", count=count, offset=count * 8).reshape(bn, bm)

            with grid_lock:
                if ux_grid is None or ux_grid.shape != (gny, gnx):
                    ux_grid = np.zeros((gny, gnx), dtype=np.float64)
                    uy_grid = np.zeros((gny, gnx), dtype=np.float64)
                ux_grid[jj * bm:(jj + 1) * bm, ii * bn:(ii + 1) * bn] = ux_block.T
                uy_grid[jj * bm:(jj + 1) * bm, ii * bn:(ii + 1) * bn] = uy_block.T
    finally:
        conn.close()


def network_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    print(f"Listening on {HOST}:{PORT}")

    while True:
        conn, _ = server.accept()
        process_connection(conn)  # one sender (one persistent connection) at a time


def render_scaled(screen, image, dest_rect):
    """image is (x, y, 3); dest_rect is (x0, y0, w, h) within screen."""
    x0, y0, w, h = dest_rect
    img_w, img_h = image.shape[:2]

    scale = min(w / img_w, h / img_h)
    draw_w, draw_h = int(img_w * scale), int(img_h * scale)
    x, y = x0 + (w - draw_w) // 2, y0 + (h - draw_h) // 2

    surface = pygame.surfarray.make_surface(image)
    scaled = pygame.transform.smoothscale(surface, (draw_w, draw_h))
    screen.blit(scaled, (x, y))


def main():
    threading.Thread(target=network_thread, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((1000, 500), pygame.RESIZABLE)
    pygame.display.set_caption("lattice_gas live velocity (ux | uy)")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        with grid_lock:
            ux = None if ux_grid is None else ux_grid.copy()
            uy = None if uy_grid is None else uy_grid.copy()

        screen.fill((0, 0, 0))
        if ux is not None:
            win_w, win_h = screen.get_size()
            half_w = win_w // 2
            render_scaled(screen, velocity_to_rgb(ux).transpose(1, 0, 2), (0, 0, half_w, win_h))
            render_scaled(screen, velocity_to_rgb(uy).transpose(1, 0, 2), (half_w, 0, win_w - half_w, win_h))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
