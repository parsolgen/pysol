"""Live velocity-field visualizer for lattice_gas.py.

Run this first, then run lattice_gas.py (in the same process or on a
different machine) with:

    LATTICE_GAS_VIZ_ENABLE=1 python3 lattice_gas.py

lattice_gas.py's send_velocity_block kernel (see its long comment in
ucodes.h) connects to this script over a plain TCP socket and streams
each block's velocity (ux, uy) the instant it's computed; this script
redraws immediately on every block it receives, rather than waiting for
the run to finish or for a full timestep's worth of blocks to arrive -
"live" meaning exactly that: watch the lid-driven flow develop while the
simulation is still running.

IP address and port are both set on the COMMAND LINE here (--host/--port
default to 0.0.0.0:8765, i.e. listen on every interface) and via
environment variables on the simulation side (LATTICE_GAS_VIZ_HOST/
LATTICE_GAS_VIZ_PORT, default 127.0.0.1:8765) - change whichever side's
default doesn't fit (e.g. point LATTICE_GAS_VIZ_HOST at this machine's
real address if lattice_gas.py is running elsewhere on the network; this
script's --host only controls which local interface it listens on, not
where connections may come from).

Wire format (must stay in sync with send_velocity_block in ucodes.h):
    header   = 4 bytes b"LGVP" + 7x little-endian int32:
               step, ii, jj, bn, bm, gnx, gny
    payload  = bn*bm little-endian float64 (ux block, row-major [li][lj])
             + bn*bm little-endian float64 (uy block, same layout)
"""

import argparse
import socket
import struct
import sys

HEADER_FORMAT = "<4siiiiiii"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"LGVP"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0",
                         help="local interface to listen on (default: 0.0.0.0, all interfaces)")
    parser.add_argument("--port", type=int, default=8765,
                         help="TCP port to listen on (default: 8765, matches "
                              "lattice_gas.py's LATTICE_GAS_VIZ_PORT default)")
    parser.add_argument("--save-frames", metavar="DIR", default=None,
                         help="instead of an interactive window, save each received "
                              "frame as a PNG into DIR (headless-safe - useful for "
                              "testing without a display, or for building an "
                              "animation afterward)")
    return parser.parse_args()


def recv_exact(conn, n, on_idle):
    """Reads exactly n bytes, calling on_idle() (which processes GUI
    events and yields briefly) instead of blocking the whole process
    whenever the socket has nothing ready yet - this is what keeps the
    plot window responsive (resizable, closable) while waiting between
    blocks, not just while new data is actively arriving."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            on_idle()
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class FieldDisplay:
    """Owns the full-grid ux/uy arrays and the two live subplots, both
    (re)allocated lazily from the first header's (gnx, gny) - or
    reallocated if a later connection reports a different grid size,
    since each TCP connection is a fresh simulation run that may use
    different parameters than the last."""

    def __init__(self, plt, np, save_dir):
        self.plt = plt
        self.np = np
        self.save_dir = save_dir
        self.gnx = None
        self.gny = None
        self.ux = None
        self.uy = None
        self.fig = None
        self.ax_ux = None
        self.ax_uy = None
        self.im_ux = None
        self.im_uy = None
        self.frame_count = 0

    def _allocate(self, gnx, gny):
        np = self.np
        self.gnx, self.gny = gnx, gny
        self.ux = np.zeros((gny, gnx), dtype=np.float64)
        self.uy = np.zeros((gny, gnx), dtype=np.float64)

        if self.fig is not None:
            self.plt.close(self.fig)
        self.fig, (self.ax_ux, self.ax_uy) = self.plt.subplots(1, 2, figsize=(14, 4))
        self.im_ux = self.ax_ux.imshow(self.ux, origin="lower", cmap="RdBu_r", aspect="auto")
        self.im_uy = self.ax_uy.imshow(self.uy, origin="lower", cmap="RdBu_r", aspect="auto")
        self.ax_ux.set_title("ux")
        self.ax_uy.set_title("uy")
        self.fig.colorbar(self.im_ux, ax=self.ax_ux)
        self.fig.colorbar(self.im_uy, ax=self.ax_uy)
        self.fig.tight_layout()
        if self.save_dir is None:
            self.plt.show(block=False)

    def update(self, step, ii, jj, bn, bm, gnx, gny, ux_block, uy_block):
        np = self.np
        if gnx != self.gnx or gny != self.gny:
            self._allocate(gnx, gny)

        self.ux[jj * bm:(jj + 1) * bm, ii * bn:(ii + 1) * bn] = ux_block.T
        self.uy[jj * bm:(jj + 1) * bm, ii * bn:(ii + 1) * bn] = uy_block.T

        self.im_ux.set_data(self.ux)
        self.im_uy.set_data(self.uy)
        for im, field in ((self.im_ux, self.ux), (self.im_uy, self.uy)):
            lo, hi = float(field.min()), float(field.max())
            if lo == hi:
                lo, hi = lo - 1e-9, hi + 1e-9
            im.set_clim(lo, hi)

        self.fig.suptitle(f"step={step}  block=({ii},{jj})")

        if self.save_dir is not None:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            self.frame_count += 1
            path = os.path.join(self.save_dir, f"frame_{self.frame_count:06d}.png")
            self.fig.savefig(path)
        else:
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)

    def idle(self):
        if self.fig is not None and self.save_dir is None:
            self.plt.pause(0.01)
        elif self.save_dir is None:
            # No figure yet (nothing received this connection); still
            # need SOMETHING to keep any already-open window from an
            # earlier connection responsive.
            try:
                self.plt.pause(0.01)
            except Exception:
                pass


def main():
    args = parse_args()

    if args.save_frames is not None:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if args.save_frames is None:
        plt.ion()

    display = FieldDisplay(plt, np, args.save_frames)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(1)
    print(f"visualizer.py: listening on {args.host}:{args.port} "
          f"- waiting for lattice_gas.py to connect...", file=sys.stderr)

    try:
        while True:
            conn, addr = listener.accept()
            print(f"visualizer.py: connected from {addr}", file=sys.stderr)
            conn.settimeout(0.05)
            try:
                while True:
                    header_bytes = recv_exact(conn, HEADER_SIZE, display.idle)
                    if header_bytes is None:
                        print("visualizer.py: sender disconnected, waiting for a new connection...",
                              file=sys.stderr)
                        break

                    magic, step, ii, jj, bn, bm, gnx, gny = struct.unpack(HEADER_FORMAT, header_bytes)
                    if magic != MAGIC:
                        print("visualizer.py: bad magic bytes, dropping this connection", file=sys.stderr)
                        break

                    payload_size = bn * bm * 8 * 2
                    payload = recv_exact(conn, payload_size, display.idle)
                    if payload is None:
                        print("visualizer.py: sender disconnected mid-block, waiting for a new connection...",
                              file=sys.stderr)
                        break

                    block_doubles = bn * bm
                    ux_block = np.frombuffer(payload, dtype="<f8", count=block_doubles).reshape(bn, bm)
                    uy_block = np.frombuffer(payload, dtype="<f8", count=block_doubles,
                                              offset=block_doubles * 8).reshape(bn, bm)

                    display.update(step, ii, jj, bn, bm, gnx, gny, ux_block, uy_block)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nvisualizer.py: stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
