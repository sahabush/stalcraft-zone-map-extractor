#!/usr/bin/env python3
"""
Prototype: STALCRAFT location .mdat -> 3D voxel mesh (.glb).

Decodes the location to Minecraft blocks (via sc-file), builds a voxel volume,
emits only the EXPOSED block faces (hidden-face culling), colours them from the
real extracted palette, and exports a single .glb you can fly around in the
browser (see viewer.html).

This is a deliberately simple first pass: vertex-coloured faces, no textures, no
greedy meshing yet. Good enough to judge the look and weight.

Usage:
    python 3d/voxel_to_glb.py "<...>/map_cache/5.0/tournament_small_berdovka" -o 3d/berdovka.glb
"""
import argparse
import glob
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scmap

AIR = scmap.AIR | scmap.MARKER  # treat invisible marker as air


def load_volume(mca_dir, keep_depth=256, clip_pct=99.9, clip_margin=3, exclude=(),
                roof_window=5, roof_margin=2):
    """Return (vol, origin) where vol[y,z,x] = block id (0 = air).

    keep_depth : keep only this many blocks under each column's surface.
    clip_pct   : flat height cap = this percentile of column-roof heights (+margin),
                 which trims the few super-tall thin spires / invisible boundary
                 columns that rise far above the buildings.
    """
    solids = []  # (wx, wy, wz, id)
    for f in glob.glob(os.path.join(mca_dir, "*.mca")):
        for nbt in scmap.iter_chunks(f):
            lvl = nbt["Level"]
            cxp, czp = int(lvl["xPos"]), int(lvl["zPos"])
            for s in lvl["Sections"]:
                sy = int(s["Y"])
                B = np.frombuffer(bytes(s["Blocks"]), np.uint8).astype(np.uint16)
                A = s.get("Add")
                if A is not None:
                    a = np.frombuffer(bytes(A), np.uint8)
                    lo = (a & 0x0F).astype(np.uint16)
                    hi = (a >> 4).astype(np.uint16)
                    add = np.empty(4096, np.uint16)
                    add[0::2] = lo; add[1::2] = hi
                    B = B | (add << 8)
                B = B.reshape(16, 16, 16)  # (y, z, x)
                yy, zz, xx = np.where((B != 0))
                ids = B[yy, zz, xx]
                keep = ~np.isin(ids, list(AIR | set(exclude)))
                yy, zz, xx, ids = yy[keep], zz[keep], xx[keep], ids[keep]
                solids.append(np.stack([cxp * 16 + xx, sy * 16 + yy, czp * 16 + zz, ids], 1))
    if not solids:
        return None, None
    P = np.concatenate(solids).astype(np.int64)
    minx, miny, minz = P[:, 0].min(), P[:, 1].min(), P[:, 2].min()
    maxx, maxy, maxz = P[:, 0].max(), P[:, 1].max(), P[:, 2].max()
    W, Hh, D = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
    vol = np.zeros((Hh, D, W), np.uint16)
    vol[P[:, 1] - miny, P[:, 2] - minz, P[:, 0] - minx] = P[:, 3]
    solid = vol != 0
    has = solid.any(0)
    top = (Hh - 1) - np.argmax(solid[::-1], axis=0)   # top y index per (z,x)
    # report which block ids form the tall thin "spires" above the roofline
    h = top[has]
    spire_thr = int(np.percentile(h, 99)) + 6
    szz, sxx = np.where(has & (top > spire_thr))
    if len(szz):
        import collections
        tops = vol[top[szz, sxx], szz, sxx]
        hist = collections.Counter(int(i) for i in tops).most_common(6)
        print(f"      spires (>{spire_thr}, {len(szz)} cols) top-block ids: {hist}")
    # local roof clip via grayscale morphological OPENING of the height map:
    # removes thin things sticking up above the roofline (antennas, leftover walls)
    # that are narrower than the window, while keeping wide building roofs fully
    # intact — including their corners (median-filtering rounded them off).
    if roof_window and roof_window > 1:
        from scipy.ndimage import grey_opening
        th = np.where(has, top, 0).astype(np.int16)
        local = grey_opening(th, size=roof_window)
        cap_col = local + roof_margin                     # (z, x)
        yidx = np.arange(Hh)[:, None, None]
        vol = np.where(yidx <= cap_col[None], vol, 0).astype(np.uint16)
        solid = vol != 0
        top = np.minimum(top, cap_col)
    # global safety cap: trim any remaining extreme outliers
    cap = int(np.percentile(top[has], clip_pct)) + clip_margin
    if cap + 1 < Hh:
        vol[cap + 1:, :, :] = 0
        solid = vol != 0
        top = np.minimum(top, cap)
    # crop to a surface shell: keep only the top `keep_depth` blocks per column
    yidx = np.arange(Hh)[:, None, None]
    keep = solid & (yidx > (top[None] - keep_depth)) & has[None]
    vol = np.where(keep, vol, 0).astype(np.uint16)
    print(f"      volume {W}x{Hh}x{D}  height cap y<={cap}  "
          f"({int((vol!=0).sum()):,} voxels)")
    return vol, (minx, miny, minz)


# face templates: 4 corner offsets per +/- direction in (x,y,z)
FACES = {
    "x+": ([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], (1, 0, 0)),
    "x-": ([(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)], (-1, 0, 0)),
    "y+": ([(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)], (0, 1, 0)),
    "y-": ([(0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1)], (0, -1, 0)),
    "z+": ([(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)], (0, 0, 1)),
    "z-": ([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], (0, 0, -1)),
}


def downsample(vol, n):
    """Pool the volume into n^3 blocks (representative id = max in block).
    A quick lever to keep the prototype mesh light; the real tool will use
    greedy meshing instead."""
    if n <= 1:
        return vol
    Hh, D, W = vol.shape
    Hh2, D2, W2 = Hh // n, D // n, W // n
    v = vol[:Hh2 * n, :D2 * n, :W2 * n].reshape(Hh2, n, D2, n, W2, n)
    return v.max(axis=(1, 3, 5)).astype(np.uint16)


def _merge_plane(plane):
    """Greedy-merge a 2D int plane (0 = empty) into rectangles (i, j, h, w, val)."""
    H, W = plane.shape
    used = np.zeros((H, W), bool)
    out = []
    nz = plane != 0
    for i in range(H):
        row = plane[i]; rused = used[i]
        j = 0
        while j < W:
            c = row[j]
            if c == 0 or rused[j]:
                j += 1; continue
            w = 1
            while j + w < W and row[j + w] == c and not rused[j + w]:
                w += 1
            h = 1
            while i + h < H:
                seg = plane[i + h, j:j + w]
                if np.any(seg != c) or np.any(used[i + h, j:j + w]):
                    break
                h += 1
            used[i:i + h, j:j + w] = True
            out.append((i, j, h, w, int(c)))
            j += w
    return out


def _id_color(bid, debug):
    if debug:  # distinct hashed colour per block id (to identify blocks)
        import hashlib
        d = hashlib.md5(str(bid).encode()).digest()
        return (60 + d[0] * 3 // 4, 60 + d[1] * 3 // 4, 60 + d[2] * 3 // 4)
    return scmap.color_for(int(bid)) or scmap.FALLBACK


def _adjust(palette, saturation, brightness):
    p = palette.astype(np.float32)
    gray = p @ np.array([0.299, 0.587, 0.114], np.float32)
    p = gray[:, None] + saturation * (p - gray[:, None])   # saturation around luma
    p *= brightness
    return np.clip(p, 0, 255).astype(np.uint8)


def greedy_mesh(vol, debug=False, saturation=1.0, brightness=1.0):
    """Greedy voxel meshing: merge coplanar same-block faces into big quads.
    Drastically fewer triangles than per-voxel faces, at full resolution."""
    A = np.ascontiguousarray(vol.transpose(2, 0, 1))  # (x, y, z)
    dims = A.shape
    palette = np.zeros((65536, 3), np.uint8)
    for bid in np.unique(A[A != 0]):
        palette[bid] = _id_color(int(bid), debug)
    if not debug:
        palette = _adjust(palette, saturation, brightness)

    V, F, C = [], [], []
    base = 0
    for d in range(3):
        u, v = [a for a in (0, 1, 2) if a != d]
        Ad = np.moveaxis(A, d, 0)  # (dims[d], dims[u], dims[v])
        nd = dims[d]
        empty = np.zeros(Ad.shape[1:], Ad.dtype)
        for s in range(-1, nd):
            front = Ad[s] if s >= 0 else empty
            back = Ad[s + 1] if s + 1 < nd else empty
            fs, bs = front != 0, back != 0
            plane = np.zeros(front.shape, np.int32)
            m = fs & ~bs; plane[m] = front[m]
            m = bs & ~fs; plane[m] = -back[m].astype(np.int32)
            if not plane.any():
                continue
            for (i, j, h, w, c) in _merge_plane(plane):
                col = palette[abs(c)]
                quad = []
                for (du, dv) in ((i, j), (i + h, j), (i + h, j + w), (i, j + w)):
                    p = [0, 0, 0]
                    p[d] = s + 1; p[u] = du; p[v] = dv
                    quad.append(p)
                V.append(quad)
                F.append((base, base + 1, base + 2))
                F.append((base, base + 2, base + 3))
                C.append(col)
                base += 4
    V = np.array(V, np.float32).reshape(-1, 3)
    F = np.array(F, np.int64)
    C = np.repeat(np.array(C, np.uint8), 4, axis=0)
    return V, F, C


def build_mesh(vol):
    solid = vol != 0
    palette = np.zeros((65536, 3), np.uint8)
    for bid in np.unique(vol[solid]):
        palette[bid] = scmap.color_for(int(bid)) or scmap.FALLBACK

    V_parts, F_parts, C_parts = [], [], []
    base = 0
    for name, (corners, (dx, dy, dz)) in FACES.items():
        nb = np.zeros_like(solid)
        sl_dst = [slice(None)] * 3
        sl_src = [slice(None)] * 3
        for ax, d in zip((1, 0, 2), (dy, dz, dx)):  # vol axes order: y, z, x
            if d > 0:
                sl_dst[ax] = slice(0, vol.shape[ax] - d); sl_src[ax] = slice(d, None)
            elif d < 0:
                sl_dst[ax] = slice(-d, None); sl_src[ax] = slice(0, vol.shape[ax] + d)
        nb[tuple(sl_dst)] = solid[tuple(sl_src)]
        exposed = solid & ~nb
        ys, zs, xs = np.where(exposed)
        n = len(xs)
        if n == 0:
            continue
        # 4 vertices per face, in corner order -> shape (n,4,3)
        vblock = np.empty((n, 4, 3), np.float32)
        for k, (ox, oy, oz) in enumerate(corners):
            vblock[:, k, 0] = xs + ox
            vblock[:, k, 1] = ys + oy
            vblock[:, k, 2] = zs + oz
        V_parts.append(vblock.reshape(-1, 3))
        idx = base + np.arange(n) * 4
        F_parts.append(np.stack([idx, idx + 1, idx + 2], 1))
        F_parts.append(np.stack([idx, idx + 2, idx + 3], 1))
        C_parts.append(np.repeat(palette[vol[ys, zs, xs]], 4, axis=0))
        base += n * 4

    V = np.concatenate(V_parts)
    F = np.concatenate(F_parts).astype(np.int64)
    C = np.concatenate(C_parts).astype(np.uint8)
    return V, F, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("location")
    ap.add_argument("-o", "--output", default="3d/berdovka.glb")
    ap.add_argument("--keep-depth", type=int, default=256,
                    help="blocks kept below each column's surface (smaller = lighter)")
    ap.add_argument("--downsample", type=int, default=1,
                    help="voxel pooling factor (2-4 = chunkier; usually unneeded with greedy)")
    ap.add_argument("--no-greedy", dest="greedy", action="store_false",
                    help="disable greedy meshing (per-voxel faces, much heavier)")
    ap.add_argument("--clip-pct", type=float, default=99.9,
                    help="global safety height-cap percentile")
    ap.add_argument("--roof-window", type=int, default=5,
                    help="opening window (blocks): removes spires THINNER than this; "
                         "keeps wider building roofs and corners. 0 disables.")
    ap.add_argument("--roof-margin", type=int, default=2,
                    help="blocks allowed above the local roof before trimming")
    ap.add_argument("--exclude", default="172",
                    help="comma-separated block ids to drop (default 172 = arena's "
                         "invisible boundary walls)")
    ap.add_argument("--debug-colors", action="store_true",
                    help="give every block id a distinct colour (to identify blocks)")
    ap.add_argument("--saturation", type=float, default=1.45, help="colour intensity")
    ap.add_argument("--brightness", type=float, default=1.18, help="brightness multiplier")
    args = ap.parse_args()
    exclude = tuple(int(x) for x in args.exclude.split(",") if x.strip())

    mca = tempfile.mkdtemp()
    print("[1/3] decoding .mdat -> .mca")
    subprocess.run([sys.executable, "-m", "scfile", "mapcache", args.location, "-O", mca], check=True)

    print("[2/3] building voxel volume")
    vol, origin = load_volume(mca, keep_depth=args.keep_depth, clip_pct=args.clip_pct,
                              exclude=exclude, roof_window=args.roof_window,
                              roof_margin=args.roof_margin)
    if vol is None:
        print("no blocks"); return

    if args.downsample > 1:
        vol = downsample(vol, args.downsample)
        print(f"      downsampled x{args.downsample} -> {int((vol!=0).sum()):,} voxels")

    if args.greedy:
        print("[3/3] greedy meshing")
        V, F, C = greedy_mesh(vol, debug=args.debug_colors,
                              saturation=args.saturation, brightness=args.brightness)
    else:
        print("[3/3] meshing exposed faces (per-voxel)")
        V, F, C = build_mesh(vol)
    print(f"      {len(F):,} triangles, {len(V):,} vertices")
    mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=C, process=False)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mesh.export(args.output)
    mb = os.path.getsize(args.output) / 1e6
    print(f"      exported {args.output}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
