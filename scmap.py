#!/usr/bin/env python3
"""
stalcraft-zone-map-extractor
=============================

Render a clean top-down map image of a STALCRAFT location straight from the
game's local minimap cache (``map_cache/5.0/<location>/*.mdat``), with no base
markers, player dots or UI overlays.

Pipeline:

    .mdat  --[sc-file]-->  .mca (Minecraft 1.12.2 region)  --[this tool]-->  PNG

Container decoding (``.mdat`` -> ``.mca``) is done by the excellent
`sc-file <https://github.com/onejeuu/sc-file>`_ project. This tool adds the
missing last step: a top-down surface render (highest non-air block per column)
with a natural Minecraft map palette and relief shading, plus automatic
splitting of disconnected cached areas into separate images.

Usage:
    python scmap.py "<...>/map_cache/5.0/tournament_nizina" -o maps/nizina

Requires: sc-file, nbtlib, numpy, pillow  (pip install sc-file nbtlib numpy pillow)
"""
import argparse
import glob
import io
import os
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import nbtlib
from PIL import Image

# --- Minecraft 1.12 block id -> approximate map color -----------------------
# sc-file's mapcache remaps STALCRAFT blocks to vanilla ids chosen for their
# map colour, so a vanilla palette reproduces the in-game look closely.
AIR = {0, 166}      # air + barrier (invisible)
MARKER = {190}      # the game's invisible boundary/marker layer (sits on roofs
                    # and floats over terrain); never visible to players
WATER = {8, 9}
PALETTE = {
    1: (127, 127, 127),   # stone
    2: (106, 170, 64),    # grass
    3: (134, 96, 67),     # dirt
    4: (122, 122, 122),   # cobblestone
    5: (160, 130, 80),    # planks
    7: (50, 50, 50),      # bedrock
    12: (220, 210, 160),  # sand
    13: (132, 127, 120),  # gravel
    14: (143, 140, 125),  # gold ore (stone-ish)
    15: (140, 132, 127),  # iron ore
    16: (108, 108, 108),  # coal ore
    17: (104, 83, 50),    # log
    18: (52, 90, 38),     # leaves
    20: (200, 220, 225),  # glass
    24: (214, 205, 150),  # sandstone
    31: (80, 120, 50),    # tallgrass
    32: (120, 110, 70),   # deadbush/dry
    35: (210, 210, 205),  # wool/white
    37: (190, 175, 50),   # dandelion (yellow accent)
    38: (170, 75, 60),     # poppy (red accent)
    43: (160, 160, 160),  # double slab
    44: (165, 165, 165),  # slab
    45: (150, 84, 67),    # bricks
    48: (95, 120, 95),    # mossy cobble
    49: (28, 24, 38),     # obsidian
    65: (120, 100, 70),   # ladder
    82: (160, 166, 180),  # clay
    83: (95, 150, 70),    # reeds
    85: (150, 120, 80),   # fence
    87: (110, 60, 52),    # netherrack (reddish ground)
    97: (124, 124, 124),  # monster egg (stone)
    98: (122, 122, 122),  # stone bricks
    101: (110, 110, 110), # iron bars
    106: (70, 110, 50),   # vines
    109: (120, 120, 120), # stone brick stairs
    110: (118, 100, 118), # mycelium
    112: (45, 24, 28),    # nether brick
    113: (150, 120, 80),  # fence
    189: (190, 175, 150), # birch fence/plank tone
    190: (110, 95, 60),   # jungle tone
    191: (90, 70, 45),    # dark oak tone
    192: (150, 120, 75),  # acacia tone
}
FALLBACK = (138, 134, 126)  # neutral concrete-grey for unmapped (buildings)


def color_for(bid):
    if bid in AIR:
        return None
    if bid in WATER:
        return (58, 92, 170)
    return PALETTE.get(bid, FALLBACK)


def iter_chunks(mca_path):
    data = open(mca_path, "rb").read()
    for i in range(1024):
        off = struct.unpack(">I", b"\x00" + data[i * 4:i * 4 + 3])[0]
        cnt = data[i * 4 + 3]
        if not off or not cnt:
            continue
        start = off * 4096
        length = struct.unpack(">I", data[start:start + 4])[0]
        comp = data[start + 4]
        raw = data[start + 5:start + 5 + length - 1]
        try:
            dec = zlib.decompress(raw) if comp == 2 else raw
            yield nbtlib.File.parse(io.BytesIO(dec))
        except Exception:
            continue


def pick_surface(column, ymax, mode):
    """Choose which block represents a column, depending on view mode.

    Building structure in the cache looks like:
        marker(190) -> roof -> walls -> floor -> marker -> air(room) -> ground

    mode 'top'   : raw topmost block, including the invisible marker layer
                   (shows the marker/contour lines players never see).
    mode 'roofs' : skip the invisible marker layer only -> real building roofs
                   and terrain, no marker lines. The normal clean map.
    mode 'xray'  : also peel roofs/ceilings (anything floating over a room) to
                   reveal interior floor plans / building layouts.
    """
    if mode == "top":
        return column[ymax], ymax

    if mode == "roofs":
        for wy in range(ymax, -1, -1):
            if wy in column and column[wy] not in MARKER:
                return column[wy], wy
        return column[ymax], ymax

    # xray: descend through the top solid superstructure (marker + roof + walls)
    # until the first room (air gap), then return the floor just below it. This
    # exposes the interior floor plan. Terrain columns (solid all the way) have
    # no room, so the surface is returned unchanged.
    y = ymax
    while y >= 0 and y in column:      # skip roof / walls / markers (the cap)
        y -= 1
    while y >= 0 and y not in column:  # skip the room (air)
        y -= 1
    if y >= 0:                          # floor of the topmost room
        return column[y], y
    return column[ymax], ymax


def collect_columns(mca_dir, mode="roofs"):
    """Return {(worldx, worldz): (block_id, surface_y)} for all chunks."""
    cols = {}
    for f in glob.glob(os.path.join(mca_dir, "*.mca")):
        for nbt in iter_chunks(f):
            lvl = nbt["Level"]
            cxp, czp = int(lvl["xPos"]), int(lvl["zPos"])
            secs = {int(s["Y"]): s for s in lvl["Sections"]}
            if not secs:
                continue
            for x in range(16):
                for z in range(16):
                    column = {}
                    ymax = 0
                    for sy, s in secs.items():
                        B = s["Blocks"]
                        A = s.get("Add")
                        for y in range(16):
                            idx = y * 256 + z * 16 + x
                            bid = B[idx] & 0xFF
                            if A is not None:
                                add = A[idx // 2]
                                add = (add >> 4) if (idx & 1) else (add & 0x0F)
                                bid |= add << 8
                            if bid != 0 and bid not in AIR:
                                wy = sy * 16 + y
                                column[wy] = bid
                                ymax = max(ymax, wy)
                    if not column:
                        continue
                    cols[(cxp * 16 + x, czp * 16 + z)] = pick_surface(
                        column, ymax, mode)
    return cols


def label_clusters(occ, cell=16, gap=2):
    """Connected components on a downsampled occupancy grid. Returns list of
    (x0, z0, x1, z1) bounding boxes in full-res pixel coords, largest first."""
    H, W = occ.shape
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    grid = np.zeros((gh, gw), bool)
    for gz in range(gh):
        for gx in range(gw):
            if occ[gz * cell:(gz + 1) * cell, gx * cell:(gx + 1) * cell].any():
                grid[gz, gx] = True
    seen = np.zeros_like(grid)
    boxes = []
    for sz in range(gh):
        for sx in range(gw):
            if not grid[sz, sx] or seen[sz, sx]:
                continue
            stack = [(sz, sx)]
            seen[sz, sx] = True
            minx = maxx = sx
            minz = maxz = sz
            while stack:
                cz, cx = stack.pop()
                minx, maxx = min(minx, cx), max(maxx, cx)
                minz, maxz = min(minz, cz), max(maxz, cz)
                for dz in range(-gap, gap + 1):
                    for dx in range(-gap, gap + 1):
                        nz, nx = cz + dz, cx + dx
                        if 0 <= nz < gh and 0 <= nx < gw and grid[nz, nx] and not seen[nz, nx]:
                            seen[nz, nx] = True
                            stack.append((nz, nx))
            boxes.append((minx * cell, minz * cell, (maxx + 1) * cell, (maxz + 1) * cell))
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def render(cols):
    xs = [c[0] for c in cols]
    zs = [c[1] for c in cols]
    minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
    W, H = maxx - minx + 1, maxz - minz + 1
    img = np.zeros((H, W, 3), np.uint8)
    hgt = np.full((H, W), np.nan, np.float32)
    occ = np.zeros((H, W), bool)
    for (wx, wz), (bid, y) in cols.items():
        c = color_for(bid)
        if c is None:
            continue
        px, pz = wx - minx, wz - minz
        img[pz, px] = c
        hgt[pz, px] = y
        occ[pz, px] = True
    # relief shading (north-west light)
    hf = np.nan_to_num(hgt, nan=float(np.nanmedian(hgt)))
    dzx = np.zeros_like(hf); dzz = np.zeros_like(hf)
    dzx[:, 1:] = hf[:, 1:] - hf[:, :-1]
    dzz[1:, :] = hf[1:, :] - hf[:-1, :]
    shade = np.clip(1.0 + np.clip((dzx + dzz) * 0.12, -0.5, 0.5), 0.5, 1.5)
    img = np.clip(img.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    img[~occ] = (0, 0, 0)
    return img, occ


def main():
    ap = argparse.ArgumentParser(description="STALCRAFT location -> clean top-down map PNG")
    ap.add_argument("location", help="path to map_cache/5.0/<location> directory")
    ap.add_argument("-o", "--output", default="maps", help="output directory")
    ap.add_argument("--keep-mca", action="store_true", help="keep intermediate .mca files")
    ap.add_argument("--min-area", type=int, default=40000,
                    help="ignore clusters smaller than this many pixels")
    ap.add_argument("-s", "--scale", type=int, default=1,
                    help="pixels per block (nearest-neighbour upscale; keeps zoom crisp)")
    ap.add_argument("-m", "--mode", choices=["roofs", "xray", "top"], default="roofs",
                    help="roofs: normal map with building roofs, marker layer removed "
                         "(default). xray: peel roofs to reveal interior floor plans. "
                         "top: raw topmost block incl. the invisible marker lines.")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.location))
    os.makedirs(args.output, exist_ok=True)
    mca_dir = os.path.join(args.output, f"_mca_{name}") if args.keep_mca else tempfile.mkdtemp()

    print(f"[1/3] decoding .mdat -> .mca  ({name})")
    os.makedirs(mca_dir, exist_ok=True)
    subprocess.run([sys.executable, "-m", "scfile", "mapcache", args.location, "-O", mca_dir],
                   check=True)

    print(f"[2/3] reading blocks (mode={args.mode})")
    cols = collect_columns(mca_dir, mode=args.mode)
    if not cols:
        print("no block data found"); return
    img, occ = render(cols)

    def save(arr, path):
        im = Image.fromarray(arr)
        if args.scale > 1:
            im = im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)
        im.save(path)

    suffix = "" if args.mode == "roofs" else f"_{args.mode}"
    print("[3/3] splitting cached areas & saving")
    boxes = [b for b in label_clusters(occ)
             if (b[2] - b[0]) * (b[3] - b[1]) >= args.min_area
             and occ[b[1]:b[3], b[0]:b[2]].any()]
    if len(boxes) <= 1:
        out = os.path.join(args.output, f"{name}{suffix}.png")
        save(img, out)
        print("  saved", out, f"({img.shape[1]}x{img.shape[0]} blocks, scale x{args.scale})")
    else:
        for i, (x0, z0, x1, z1) in enumerate(boxes, 1):
            crop = img[z0:z1, x0:x1]
            filled = int(occ[z0:z1, x0:x1].sum())
            out = os.path.join(args.output, f"{name}_part{i}{suffix}.png")
            save(crop, out)
            print(f"  saved {out}  ({crop.shape[1]}x{crop.shape[0]} blocks, {filled} filled)")
        print(f"  note: {len(boxes)} disconnected cached areas (e.g. same arena cached twice).")
    print("done.")


if __name__ == "__main__":
    main()
