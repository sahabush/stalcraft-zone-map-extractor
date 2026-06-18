#!/usr/bin/env python3
"""
Extract the TRUE in-game block palette from the .ol world map.

The open-world map exists in two forms that share the same world grid:
  * map_cache/5.0/<loc>/reg.X.Y.mdat  -> block ids (via sc-file -> .mca)
  * <game>/modassets/assets/pda/map/r.X.Y.ol -> the real rendered colour image

Block colours are global (grass is grass everywhere), so we read both for an
open-world location, look up each block id's real colour from the .ol tiles
(1 pixel = 1 block, 512-block regions), take the median, and get the true
`block id -> colour` table. That table then colours the arenas, which only ship
as .mdat blocks.

Usage:
    python calibrate_from_ol.py <map_cache/5.0/kordon> <game>/modassets/assets/pda/map
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

import scmap

TILE = 512  # pda/map .ol tile = 512x512 blocks (Minecraft region)


def load_ol_tile(ol_path, cache):
    if ol_path in cache:
        return cache[ol_path]
    tmp = tempfile.mkdtemp()
    subprocess.run([sys.executable, "-m", "scfile", "convert", ol_path, "-O", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dds = glob.glob(os.path.join(tmp, "*.dds"))
    img = np.asarray(Image.open(dds[0]).convert("RGB"), np.int16) if dds else None
    cache[ol_path] = img
    return img


def is_fog(rgb):
    # fog/cloud = light, low-saturation grey
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    return (mx > 150) and (mx - mn < 22)


def main():
    ap = argparse.ArgumentParser(description="Extract real block palette from .ol world map")
    ap.add_argument("pda_map", help="<game>/modassets/assets/pda/map directory")
    ap.add_argument("locations", nargs="+", help="one or more map_cache/5.0/<open-world location>")
    ap.add_argument("--offx", type=int, default=0, help="block offset X (.mdat vs .ol)")
    ap.add_argument("--offz", type=int, default=0, help="block offset Z")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--out", default="palette_ol.py")
    args = ap.parse_args()

    samples = {}           # bid -> list of [r,g,b]
    tilecache = {}
    for loc in args.locations:
        lname = os.path.basename(os.path.normpath(loc))
        mca = tempfile.mkdtemp()
        print(f"[*] {lname}: decoding + sampling")
        subprocess.run([sys.executable, "-m", "scfile", "mapcache", loc, "-O", mca], check=True)
        cols = scmap.collect_columns(mca, mode="roof")
        miss = fog = ok = 0
        for (wx, wz), (bid, y) in cols.items():
            sx, sz = wx + args.offx, wz + args.offz
            tx, tz = sx // TILE, sz // TILE
            ol = os.path.join(args.pda_map, f"r.{tx}.{tz}.ol")
            if not os.path.exists(ol):
                miss += 1; continue
            img = load_ol_tile(ol, tilecache)
            if img is None:
                miss += 1; continue
            px, pz = sx % TILE, sz % TILE
            if not (0 <= pz < img.shape[0] and 0 <= px < img.shape[1]):
                miss += 1; continue
            c = img[pz, px]
            if is_fog(c):
                fog += 1; continue
            samples.setdefault(int(bid), []).append(c)
            ok += 1
        print(f"    {lname}: ok={ok} fog={fog} miss={miss}")
    name = "+".join(os.path.basename(os.path.normpath(l)) for l in args.locations)

    palette = {}
    counts = {}
    for bid, lst in samples.items():
        if len(lst) < args.min_count:
            continue
        arr = np.array(lst)
        palette[bid] = tuple(int(v) for v in np.median(arr, axis=0))
        counts[bid] = len(lst)

    print(f"      extracted {len(palette)} block colours")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Real palette extracted from .ol world map (location: {name})\n")
        f.write("PALETTE = {\n")
        for bid in sorted(palette, key=lambda b: -counts[b]):
            f.write(f"    {bid}: {palette[bid]},  # n={counts[bid]}\n")
        f.write("}\n")
    print(f"[3/3] wrote {args.out}")
    for bid in sorted(palette, key=lambda b: -counts[b])[:12]:
        print(f"      id {bid:>3}: {palette[bid]}  (n={counts[bid]})")


if __name__ == "__main__":
    main()
