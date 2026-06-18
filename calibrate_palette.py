#!/usr/bin/env python3
"""
Palette calibrator — derive the in-game colours automatically.

Idea: for every pixel we know the block id (from the .mca), and on your
reference screenshot we know the real in-game colour of that same spot. Overlay
them and, for each block id, take the MEDIAN reference colour. Median is robust
to markers (A/B/C), fog and small misalignment, so we recover the game's real
palette without touching the obfuscated client.

Usage:
    python calibrate_palette.py <map_cache/5.0/LOCATION> reference.png

Tips for a good result:
  * Crop/scale the reference so it frames the SAME area as the tool's output
    (open maps/<loc>.png to compare framing). Rough alignment is fine — median
    handles the rest.
  * Use the cleanest reference you have (least fog / fewest markers).
  * Add --flip-y or --flip-x if the result looks mirrored.

Outputs:
  * palette_calibrated.py   — ready-to-paste PALETTE dict
  * <loc>_calibrated.png    — preview rendered with the new palette
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

import scmap  # reuse collect_columns / iter_chunks / render helpers


def _fftconv(a, k):
    s = (a.shape[0] + k.shape[0] - 1, a.shape[1] + k.shape[1] - 1)
    return np.fft.irfft2(np.fft.rfft2(a, s) * np.fft.rfft2(k, s), s)


def _match_ncc(image, templ):
    """Best normalised cross-correlation of templ within image (templ<=image).
    Returns (score, y, x) of templ's top-left in image."""
    th, tw = templ.shape
    ih, iw = image.shape
    if th > ih or tw > iw:
        return -1, 0, 0
    t0 = templ - templ.mean()
    tss = float((t0 ** 2).sum()) or 1.0
    num = _fftconv(image, t0[::-1, ::-1])
    ones = np.ones_like(templ)
    s1 = _fftconv(image, ones)
    s2 = _fftconv(image ** 2, ones)
    # valid region: indices [th-1 .. ih-1] x [tw-1 .. iw-1] map to top-left 0..ih-th
    ys, xs = ih - th + 1, iw - tw + 1
    n = th * tw
    out = np.full((ys, xs), -1.0)
    for y in range(ys):
        for x in range(xs):
            yy, xx = y + th - 1, x + tw - 1
            ssum = s1[yy, xx]; sq = s2[yy, xx]
            denom = ((sq - ssum * ssum / n) * tss)
            if denom > 1e-6:
                out[y, x] = num[yy, xx] / np.sqrt(denom)
    i = np.unravel_index(np.argmax(out), out.shape)
    return out[i], i[0], i[1]


def register(ref_img, template_path, ref_w=300):
    """Find the sub-rectangle of ref_img that matches our render (template).
    Returns (x0,y0,x1,y1) in ref full-res coords, or None if weak."""
    tmpl_full = Image.open(template_path).convert("L")
    ref_g = ref_img.convert("L")
    scale = ref_w / ref_g.width
    rh = int(ref_g.height * scale)
    R = np.asarray(ref_g.resize((ref_w, rh)), float)
    best = None  # (score, x0,y0,x1,y1)
    taspect = tmpl_full.height / tmpl_full.width
    # try template widths from 45% to 98% of the reference width
    for tw in range(int(ref_w * 0.45), int(ref_w * 0.98), 8):
        th = int(tw * taspect)
        if th >= rh:
            continue
        T = np.asarray(tmpl_full.resize((tw, th)), float)
        sc, y, x = _match_ncc(R, T)
        if best is None or sc > best[0]:
            best = (sc, x, y, x + tw, y + th)
    if not best or best[0] < 0.15:
        return None
    inv = 1.0 / scale
    _, x0, y0, x1, y1 = best
    return (int(x0 * inv), int(y0 * inv), int(x1 * inv), int(y1 * inv))


def build_id_grid(cols):
    xs = [c[0] for c in cols]; zs = [c[1] for c in cols]
    minx, maxx, minz, maxz = min(xs), max(xs), min(zs), max(zs)
    W, H = maxx - minx + 1, maxz - minz + 1
    ids = np.zeros((H, W), np.int32)
    ys = np.full((H, W), -9999, np.int32)
    for (wx, wz), (bid, y) in cols.items():
        ids[wz - minz, wx - minx] = bid
        ys[wz - minz, wx - minx] = y
    return ids, ys


def main():
    ap = argparse.ArgumentParser(description="Calibrate block palette from a reference image")
    ap.add_argument("location", help="map_cache/5.0/<location> folder")
    ap.add_argument("reference", help="reference image (in-game map screenshot)")
    ap.add_argument("--min-count", type=int, default=40,
                    help="ignore ids with fewer than this many sampled pixels")
    ap.add_argument("--flip-x", action="store_true")
    ap.add_argument("--flip-y", action="store_true")
    ap.add_argument("--clusters", type=int, default=14, help="number of dominant reference colours")
    ap.add_argument("--trim-frac", type=float, default=0.35,
                    help="keep rows/cols whose edge-density exceeds this fraction of "
                         "the max (trims the smooth fog border)")
    ap.add_argument("--no-trim", action="store_true", help="don't trim fog border")
    ap.add_argument("--out", default="palette_calibrated.py")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.location))
    mca = tempfile.mkdtemp()
    print(f"[1/4] decoding .mdat -> .mca ({name})")
    subprocess.run([sys.executable, "-m", "scfile", "mapcache", args.location, "-O", mca], check=True)

    print("[2/4] building block-id grid")
    cols = scmap.collect_columns(mca, mode="roofs")
    ids, ys = build_id_grid(cols)
    H, W = ids.shape

    print("[3/4] aligning reference & sampling colours")
    ref = Image.open(args.reference).convert("RGB")
    if args.flip_x:
        ref = ref.transpose(Image.FLIP_LEFT_RIGHT)
    if args.flip_y:
        ref = ref.transpose(Image.FLIP_TOP_BOTTOM)

    # Trim the smooth fog/cloud margin so the arena fills the frame like our grid.
    # Fog (clouds) is low-detail; the arena (buildings/roads) is high-detail, so
    # we keep rows/cols whose gradient (edge) density is high.
    if not args.no_trim:
        g = np.asarray(ref.convert("L"), float)
        grad = np.zeros_like(g)
        grad[:, 1:] += np.abs(g[:, 1:] - g[:, :-1])
        grad[1:, :] += np.abs(g[1:, :] - g[:-1, :])
        rowscore = grad.mean(1); colscore = grad.mean(0)
        rthr = rowscore.max() * args.trim_frac
        cthr = colscore.max() * args.trim_frac
        rows = np.where(rowscore > rthr)[0]
        colsx = np.where(colscore > cthr)[0]
        if len(rows) and len(colsx):
            box = (int(colsx[0]), int(rows[0]), int(colsx[-1] + 1), int(rows[-1] + 1))
            ref = ref.crop(box)
            print(f"      trimmed fog -> arena crop {box}  ({ref.size[0]}x{ref.size[1]})")
    ref = ref.resize((W, H), Image.LANCZOS)
    rgb = np.asarray(ref, np.int16)

    # ---- alignment-free calibration via colour-cluster matching ----
    # Quantise the reference into dominant colours, then snap each block id to
    # the nearest reference cluster, anchored by our current hand-tuned palette.
    qimg = ref.convert("RGB").quantize(colors=args.clusters, method=Image.FASTOCTREE)
    pal = qimg.getpalette()[: args.clusters * 3]
    clusters = np.array([pal[i:i + 3] for i in range(0, len(pal), 3)], float)
    hist = np.bincount(np.asarray(qimg).ravel(), minlength=args.clusters)
    # drop clusters that are basically fog/black (very dark, low presence)
    valid = [i for i in range(len(clusters)) if hist[i] > rgb.size * 0.0005]
    clusters = clusters[valid]
    print(f"      reference dominant colours: {len(clusters)}")

    occupied = ids != 0
    palette = {}
    counts = {}
    for bid in np.unique(ids[occupied]):
        n = int((ids == bid).sum())
        if n < args.min_count:
            continue
        anchor = np.array(scmap.color_for(int(bid)) or scmap.FALLBACK, float)
        d = np.sqrt(((clusters - anchor) ** 2).sum(1))
        palette[int(bid)] = tuple(int(v) for v in clusters[int(np.argmin(d))])
        counts[int(bid)] = n

    print(f"      calibrated {len(palette)} block ids")

    # write palette file
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Auto-calibrated from reference: %s (location: %s)\n" % (
            os.path.basename(args.reference), name))
        f.write("PALETTE = {\n")
        for bid in sorted(palette, key=lambda b: -counts[b]):
            f.write(f"    {bid}: {palette[bid]},  # n={counts[bid]}\n")
        f.write("}\n")
    print(f"[4/4] wrote {args.out}")

    # preview render with calibrated palette
    fallback = palette.get(1, (110, 105, 98))
    img = np.zeros((H, W, 3), np.uint8)
    for bid, col in palette.items():
        img[ids == bid] = col
    # ids present but uncalibrated -> fallback; empty -> black
    known = np.isin(ids, list(palette.keys()))
    img[occupied & ~known] = fallback
    out_png = f"{name}_calibrated.png"
    Image.fromarray(img).resize((W * 2, H * 2), Image.NEAREST).save(out_png)
    print(f"      preview -> {out_png}")


if __name__ == "__main__":
    main()
