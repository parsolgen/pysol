"""Live-image visualizer for raytracer_2.py's pysol port of
examples/raytracing_2/shader.glsl - plain pygame + sockets (same
architecture as examples/raytracing/visualizer.py), no matplotlib.

Run this first, then run raytracer_2.py with:

    RT2_VIZ_ENABLE=1 python3 raytracer_2.py

raytracer_2.py's send_image_block kernel (see ucodes.h) opens ONE
persistent connection for the whole run and streams every block's
freshly ray-traced RGB tile over it the instant it's computed. This
script accumulates a frame's tiles into one buffer and only swaps that
buffer into the one the render loop actually draws from once every
block for that frame has arrived - drawing straight from a
still-filling buffer shows a patchwork of tiles each one frame out of
sync with its neighbors, the same tearing bug previously found and
fixed in examples/raytracing's and examples/gas-ca's visualizers.

Usage:
    python3 visualizer.py [port]        # default 8768, matches
                                         # raytracer_2.py's RT2_VIZ_PORT
                                         # default (examples/raytracing
                                         # uses 8767, so both can run at
                                         # once without colliding)

Wire format (must stay in sync with send_image_block in ucodes.h):
    header  = 4 bytes b"RTI2" + 7x little-endian int32:
              frame, ii, jj, bn, bm, gnx, gny
    payload = bn*bm little-endian float64 (r block, row-major [li][lj])
            + bn*bm little-endian float64 (g block, same layout)
            + bn*bm little-endian float64 (b block, same layout)
"""
import socket
import struct
import sys
import threading

import numpy as np
import pygame

HOST = "0.0.0.0"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8768

HEADER_FORMAT = "<4siiiiiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"RTI2"

# (gny, gnx, 3) uint8 image grids - see the module docstring for why two
# buffers: `image_grid` accumulates the still-incomplete current frame's
# tiles (arriving one block at a time, in whatever order the sender's OMP
# threads get scheduled in), `image_display` is only replaced once every
# block for that frame has landed, and the render loop reads only
# `image_display`.
image_grid = None
image_display = None
cur_frame = None
received = 0
expected = None
grid_lock = threading.Lock()


def recv_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def process_connection(conn):
    global image_grid, image_display, cur_frame, received, expected
    print(f"visualizer.py: connected from {conn.getpeername()}")
    try:
        while True:
            header = recv_exact(conn, HEADER_SIZE)
            if header is None:
                print("visualizer.py: sender disconnected, waiting for a new connection...")
                return
            magic, frame, ii, jj, bn, bm, gnx, gny = struct.unpack(HEADER_FORMAT, header)
            if magic != MAGIC:
                print("visualizer.py: bad magic bytes, dropping this connection")
                return

            count = bn * bm
            payload = recv_exact(conn, count * 8 * 3)
            if payload is None:
                print("visualizer.py: sender disconnected mid-block")
                return

            r_block = np.frombuffer(payload, dtype="<f8", count=count).reshape(bn, bm)
            g_block = np.frombuffer(payload, dtype="<f8", count=count, offset=count * 8).reshape(bn, bm)
            b_block = np.frombuffer(payload, dtype="<f8", count=count, offset=count * 16).reshape(bn, bm)

            rgb_block = np.stack([r_block, g_block, b_block], axis=-1)
            rgb_block = (np.clip(rgb_block, 0.0, 1.0) * 255.0).astype(np.uint8)

            with grid_lock:
                if image_grid is None or image_grid.shape[:2] != (gny, gnx):
                    image_grid = np.zeros((gny, gnx, 3), dtype=np.uint8)
                    cur_frame = frame
                    expected = (gnx // bn) * (gny // bm)
                    received = 0
                elif frame > cur_frame:
                    # A new frame has started - whatever was still
                    # incomplete for the previous one is simply
                    # overwritten tile-by-tile below, same as
                    # examples/raytracing's visualizer.py.
                    cur_frame = frame
                    received = 0
                elif frame < cur_frame:
                    continue  # stale straggler from an already-superseded frame

                # rgb_block is (bn, bm, 3) in [li][lj] order; the grid is
                # (gny, gnx, 3) in [row][col] order, so transpose the
                # first two axes to match.
                image_grid[jj * bm:(jj + 1) * bm, ii * bn:(ii + 1) * bn] = rgb_block.transpose(1, 0, 2)
                received += 1
                if received >= expected:
                    # calc_block's pixel_pos.y follows the shader's GLSL
                    # convention (row gj=0 is the BOTTOM of the frame),
                    # but row 0 of a numpy/pygame image is the TOP - so
                    # flip vertically here, once per completed frame.
                    image_display = image_grid[::-1].copy()
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


def render_scaled(screen, image):
    win_w, win_h = screen.get_size()
    img_h, img_w = image.shape[:2]

    scale = min(win_w / img_w, win_h / img_h)
    draw_w, draw_h = int(img_w * scale), int(img_h * scale)
    x, y = (win_w - draw_w) // 2, (win_h - draw_h) // 2

    surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
    scaled = pygame.transform.smoothscale(surface, (draw_w, draw_h))
    screen.fill((0, 0, 0))
    screen.blit(scaled, (x, y))


def main():
    threading.Thread(target=network_thread, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((1000, 600), pygame.RESIZABLE)
    pygame.display.set_caption("pysol live ray-tracer 2")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        with grid_lock:
            image = None if image_display is None else image_display.copy()

        if image is not None:
            render_scaled(screen, image)
        else:
            screen.fill((0, 0, 0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
