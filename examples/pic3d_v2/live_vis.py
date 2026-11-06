"""Simple live viewer for pic3d_v2's density field - XY projection only
(no 3D model). Plain pygame + sockets, modeled directly on
examples/simple/draw.py - no matplotlib.

Run this before (or at any point during) a pic3d_v2 run that has
PYSOL_VIS_PORT set in its environment - see send_density_xy_block in
ucodes.h for the sender side. Every fragment connects once per
timestep, sends its own local sum-over-z tile, and disconnects; tiles
sharing the same (ii, jj) column (i.e. every kk) are added together
here into one global XY projection, exactly like write_density_files()'s
own offline `arr.sum(axis=2)`.

Usage:
    python3 live_vis.py [port]            # default port 8765
    PYSOL_VIS_PORT=8765 python3 pic3d.py  # in another terminal
"""
import socket
import struct
import sys
import threading

import numpy as np
import pygame

HOST = "0.0.0.0"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

# Must match VisHeader in ucodes.h exactly (10 native-endian int32 fields) -
# loopback-only tool, so no network byte order conversion either side.
HEADER_FORMAT = "<10i"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = 0x50563344

# Global XY-projection grid, sized lazily from the first message's own
# (nfx, nfy, ni, nj) fields. Already in pygame's (x, y) axis order, so no
# transpose is needed before make_surface().
#
# `grid` accumulates the CURRENT, still-incomplete timestep (fragments
# finish their nfz-deep z-sum at different real times, so at any instant
# some (ii,jj) columns have fewer of their nfz contributions added in
# than others); `display_grid` is only replaced once a full set of
# nfx*nfy*nfz tiles has landed. Drawing straight from `grid` instead
# would show a fragment-aligned patchwork of differently-summed tiles
# every frame - a grid-pattern artifact, the same visual shape as the
# (now-fixed, see compute_density_raw_block) density double-count bug,
# but caused by this viewer racing ahead of still-arriving data rather
# than any double-counting in the simulation itself.
grid = None
display_grid = None
cur_ts = None
received = 0
expected = None
grid_lock = threading.Lock()


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


# Matches make_density_gif()'s "inferno"-ish look (low -> dark, high ->
# light yellow), just a plain two-color lerp instead of a real colormap.
DARK_BLUE = np.array([10, 10, 80], dtype=np.float32)
LIGHT_YELLOW = np.array([255, 245, 160], dtype=np.float32)


def density_to_rgb(values):
    vmin, vmax = values.min(), values.max()
    t = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = DARK_BLUE + t[..., None] * (LIGHT_YELLOW - DARK_BLUE)
    return rgb.astype(np.uint8)


def process_connection(conn):
    global grid, display_grid, cur_ts, received, expected
    try:
        header = recv_exact(conn, HEADER_SIZE)
        magic, ts, ii, jj, kk, nfx, nfy, nfz, ni, nj = struct.unpack(HEADER_FORMAT, header)
        if magic != MAGIC:
            return

        payload = recv_exact(conn, ni * nj * 8)
        tile = np.frombuffer(payload, dtype="<f8").reshape(ni, nj)

        with grid_lock:
            if grid is None or ts > cur_ts:
                # New timestep starting - drop whatever was accumulated
                # for the previous one and start a fresh frame.
                grid = np.zeros((nfx * ni, nfy * nj), dtype=np.float64)
                cur_ts = ts
                expected = nfx * nfy * nfz
                received = 0
            elif ts < cur_ts:
                return  # stale straggler from an already-superseded ts

            grid[ii * ni:(ii + 1) * ni, jj * nj:(jj + 1) * nj] += tile
            received += 1
            if received >= expected:
                # Every fragment's z-sum is in - safe to show as one frame.
                display_grid = grid.copy()

    except Exception as e:
        print("connection error:", e)

    finally:
        conn.close()


def network_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()

    print(f"Listening on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        threading.Thread(target=process_connection, args=(conn,), daemon=True).start()


def render_scaled(screen, image):
    win_w, win_h = screen.get_size()
    img_w, img_h = image.shape[:2]

    scale = min(win_w / img_w, win_h / img_h)
    draw_w, draw_h = int(img_w * scale), int(img_h * scale)
    x, y = (win_w - draw_w) // 2, (win_h - draw_h) // 2

    surface = pygame.surfarray.make_surface(image)
    scaled = pygame.transform.smoothscale(surface, (draw_w, draw_h))

    screen.fill((0, 0, 0))
    screen.blit(scaled, (x, y))


def main():
    threading.Thread(target=network_thread, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((800, 800), pygame.RESIZABLE)
    pygame.display.set_caption("pic3d_v2 live density (XY projection)")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        with grid_lock:
            frame = None if display_grid is None else display_grid.copy()

        if frame is not None:
            render_scaled(screen, density_to_rgb(frame))
            pygame.display.flip()

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
