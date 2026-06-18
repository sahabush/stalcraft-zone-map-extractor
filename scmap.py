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
from PIL import Image, ImageEnhance

# --- Minecraft 1.12 block id -> approximate map color -----------------------
# sc-file's mapcache remaps STALCRAFT blocks to vanilla ids chosen for their
# map colour, so a vanilla palette reproduces the in-game look closely.
AIR = {0, 166}      # air + barrier (invisible)
MARKER = {190}      # the game's invisible boundary/marker layer (sits on roofs
                    # and floats over terrain); never visible to players
# Real in-game block colours, extracted automatically from the .ol world map
# (median over ~3.6M sampled pixels across kordon+bar+agroprom+svalka+yantar).
# See calibrate_from_ol.py. These ARE the game's true map colours per block id.
PALETTE = {
    2: (57, 56, 41), 1: (68, 66, 62), 13: (68, 60, 49), 112: (66, 65, 60),
    18: (115, 117, 115), 9: (54, 44, 35), 12: (82, 69, 57), 97: (66, 63, 57),
    87: (85, 78, 74), 3: (74, 67, 57), 82: (82, 70, 57), 32: (49, 48, 32),
    31: (51, 52, 33), 38: (55, 54, 35), 110: (65, 60, 49), 37: (49, 48, 32),
    189: (66, 61, 57), 58: (99, 101, 96), 49: (68, 64, 57), 83: (49, 48, 30),
    125: (65, 58, 44), 98: (66, 62, 54), 4: (85, 81, 74), 5: (66, 60, 49),
    126: (66, 59, 46), 191: (66, 60, 51), 113: (65, 60, 49), 192: (68, 62, 54),
    101: (57, 52, 41), 44: (126, 125, 123), 41: (126, 127, 126), 48: (126, 128, 126),
    176: (66, 60, 52), 35: (93, 93, 87), 144: (74, 71, 65), 217: (49, 49, 30),
    254: (57, 48, 40), 177: (62, 55, 46), 106: (63, 59, 49), 45: (71, 67, 57),
    29: (83, 72, 49), 17: (62, 55, 49), 39: (65, 65, 38), 188: (66, 65, 61),
    147: (84, 81, 74), 65: (66, 63, 57), 172: (101, 101, 95), 224: (57, 56, 41),
    47: (82, 77, 71), 159: (68, 67, 57), 43: (73, 67, 57), 68: (109, 95, 84),
    42: (57, 56, 38), 36: (32, 34, 27), 46: (99, 81, 66),
}
FALLBACK = (78, 74, 68)  # neutral for ids not present in the sampled world map


def color_for(bid):
    if bid in AIR:
        return None
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


ALIASES = {"roofs": "roof", "xray": "floor2"}


def pick_surface(column, ymax, mode):
    """Choose which block represents a column, depending on view mode.

    Building structure in the cache looks like:
        marker(190) -> roof -> walls -> floor -> [marker] -> air(room) -> ground

    mode 'top'    : raw topmost block, incl. the invisible marker layer (debug).
    mode 'roof'   : skip the invisible marker only -> real roofs + terrain.
    mode 'floor2' : peel the roof to the FIRST room from the top -> upper floor.
    mode 'floor1' : peel down to the LOWEST room -> ground floor plan.
    (aliases: roofs->roof, xray->floor2)
    """
    mode = ALIASES.get(mode, mode)

    if mode == "top":
        return column[ymax], ymax

    if mode == "roof":
        for wy in range(ymax, -1, -1):
            if wy in column and column[wy] not in MARKER:
                return column[wy], wy
        return column[ymax], ymax

    if mode == "floor2":
        # descend through the top cap (roof/walls/markers), skip the first room,
        # return the floor below it (upper interior floor).
        y = ymax
        while y >= 0 and y in column:
            y -= 1
        while y >= 0 and y not in column:
            y -= 1
        return (column[y], y) if y >= 0 else (column[ymax], ymax)

    # floor1: the ground floor = the solid surface just below the LOWEST room.
    # Scan up from the bottom: foundation solids, then the first air (room),
    # and return the last solid before that room.
    ymin = min(column)
    y = ymin
    while y in column:                 # rise through the foundation/ground floor
        y += 1
    floor = y - 1                       # last solid before the first room above
    if floor in column:
        return column[floor], floor
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


def render(cols, bbox=None):
    if bbox is None:
        xs = [c[0] for c in cols]
        zs = [c[1] for c in cols]
        minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
    else:
        minx, minz, maxx, maxz = bbox
    W, H = maxx - minx + 1, maxz - minz + 1
    img = np.zeros((H, W, 3), np.uint8)
    hgt = np.full((H, W), np.nan, np.float32)
    occ = np.zeros((H, W), bool)
    for (wx, wz), (bid, y) in cols.items():
        c = color_for(bid)
        if c is None:
            continue
        px, pz = wx - minx, wz - minz
        if not (0 <= px < W and 0 <= pz < H):
            continue
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
    ap.add_argument("-m", "--mode",
                    choices=["roof", "floor2", "floor1", "all", "roofs", "xray", "top"],
                    default="roof",
                    help="roof: normal map with building roofs (default). "
                         "floor2: upper interior floor. floor1: ground floor plan. "
                         "all: render roof+floor2+floor1 aligned (for layer overlays). "
                         "top: raw topmost incl. invisible marker. "
                         "(roofs=roof, xray=floor2)")
    ap.add_argument("--saturation", type=float, default=1.5,
                    help="colour intensity (1.0 = raw extracted/dull, >1 livelier)")
    ap.add_argument("--brightness", type=float, default=1.12,
                    help="brightness multiplier")
    ap.add_argument("--contrast", type=float, default=1.06, help="contrast multiplier")
    ap.add_argument("--palette", default="",
                    help="python file defining PALETTE = {id: (r,g,b)} to override colours "
                         "(e.g. palette_ol.py extracted from the real .ol world map)")
    args = ap.parse_args()

    if args.palette:
        ns = {}
        exec(open(args.palette, encoding="utf-8").read(), ns)
        PALETTE.update(ns["PALETTE"])
        print(f"      palette override: {len(ns['PALETTE'])} colours from {args.palette}")

    name = os.path.basename(os.path.normpath(args.location))
    os.makedirs(args.output, exist_ok=True)
    mca_dir = os.path.join(args.output, f"_mca_{name}") if args.keep_mca else tempfile.mkdtemp()

    print(f"[1/3] decoding .mdat -> .mca  ({name})")
    os.makedirs(mca_dir, exist_ok=True)
    subprocess.run([sys.executable, "-m", "scfile", "mapcache", args.location, "-O", mca_dir],
                   check=True)

    layers = ["roof", "floor2", "floor1"] if args.mode == "all" else [ALIASES.get(args.mode, args.mode)]

    def save(arr, path):
        im = Image.fromarray(arr)
        if args.saturation != 1.0:
            im = ImageEnhance.Color(im).enhance(args.saturation)
        if args.brightness != 1.0:
            im = ImageEnhance.Brightness(im).enhance(args.brightness)
        if args.contrast != 1.0:
            im = ImageEnhance.Contrast(im).enhance(args.contrast)
        if args.scale > 1:
            im = im.resize((im.width * args.scale, im.height * args.scale), Image.NEAREST)
        im.save(path)

    # shared geometry: bbox + split boxes are taken from the 'roof' layer (the
    # fullest), so every layer of an arena lines up pixel-for-pixel.
    print("[2/3] computing shared geometry (roof)")
    roof_cols = collect_columns(mca_dir, mode="roof")
    if not roof_cols:
        print("no block data found"); return
    xs = [c[0] for c in roof_cols]; zs = [c[1] for c in roof_cols]
    bbox = (min(xs), min(zs), max(xs), max(zs))
    _, roof_occ = render(roof_cols, bbox)
    boxes = [b for b in label_clusters(roof_occ)
             if (b[2] - b[0]) * (b[3] - b[1]) >= args.min_area
             and roof_occ[b[1]:b[3], b[0]:b[2]].any()]
    if len(boxes) > 1:
        print(f"      {len(boxes)} disconnected cached areas -> _partN")

    print(f"[3/3] rendering layers: {', '.join(layers)}")
    single = (len(layers) == 1)
    for layer in layers:
        cols = roof_cols if layer == "roof" else collect_columns(mca_dir, mode=layer)
        img, _ = render(cols, bbox)
        suffix = "" if (single and layer == "roof") else f"_{layer}"
        if len(boxes) <= 1:
            out = os.path.join(args.output, f"{name}{suffix}.png")
            save(img, out)
            print("  saved", out, f"({img.shape[1]}x{img.shape[0]} blocks, x{args.scale})")
        else:
            for i, (x0, z0, x1, z1) in enumerate(boxes, 1):
                out = os.path.join(args.output, f"{name}_part{i}{suffix}.png")
                save(img[z0:z1, x0:x1], out)
                print(f"  saved {out}  ({x1-x0}x{z1-z0} blocks)")
    print("done.")


if __name__ == "__main__":
    main()
