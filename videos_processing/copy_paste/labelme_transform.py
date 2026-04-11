#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelme_transform.py
Apply an affine transform (scale/rotate/translate or direct 2x3 matrix) to Labelme JSON annotations.
- Keeps shape labels/attributes unchanged
- Clips polygons to image bounds
- Updates imageWidth/Height if you pass --out-size
Usage examples:
  # Pure translation (pasting at offset)
  python labelme_transform.py in.json out.json --dx 120 --dy 45 --out-size 1920 1080

  # Scale then rotate 15° then translate
  python labelme_transform.py in.json out.json --sx 0.8 --sy 0.8 --angle 15 --dx 300 --dy 80 --out-size 1280 720

  # Provide a full affine matrix row-wise: [[a,b,tx],[c,d,ty]]
  python labelme_transform.py in.json out.json --matrix 1 0 120 0 1 40 --out-size 1920 1080
"""
import argparse, json, math, sys
from copy import deepcopy

def build_affine(args):
    if args.matrix:
        a,b,tx,c,d,ty = args.matrix
    else:
        sx, sy = args.sx, args.sy
        theta = math.radians(args.angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # scale -> rotate
        a = sx * cos_t
        b = -sx * sin_t
        c = sy * sin_t
        d = sy * cos_t
        tx, ty = args.dx, args.dy
    # 2x3 matrix
    return (a,b,tx,c,d,ty)

def apply_affine_to_point(x, y, M):
    a,b,tx,c,d,ty = M
    x2 = a*x + b*y + tx
    y2 = c*x + d*y + ty
    return x2, y2

def clip_point(x, y, W, H):
    # Clip to [0, W-1], [0, H-1]
    x = max(0.0, min(float(W-1), float(x)))
    y = max(0.0, min(float(H-1), float(y)))
    return x, y

def clean_shape_points(shape, W, H):
    # Remove consecutive duplicates after clipping; drop degenerate polygons
    if 'points' not in shape: 
        return shape
    pts = shape['points']
    new_pts = []
    last = None
    for (x,y) in pts:
        nx, ny = clip_point(x, y, W, H)
        if last is None or (abs(nx-last[0])>1e-6 or abs(ny-last[1])>1e-6):
            new_pts.append([nx, ny])
            last = (nx, ny)
    # For polygons need at least 3 points
    if shape.get('shape_type', 'polygon') in ('polygon','rectangle'):
        if len(new_pts) < 3:
            shape['flags'] = {"dropped_by_transform": True}
            shape['points'] = new_pts
            shape['label'] = f"{shape.get('label','obj')}_DEGENERATE"
            return shape
    shape['points'] = new_pts
    return shape

def transform_labelme(input_json, output_json, out_size, M):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data2 = deepcopy(data)
    # Update image size if provided
    if out_size:
        W, H = out_size
        data2['imageWidth']  = int(W)
        data2['imageHeight'] = int(H)
    else:
        W = data.get('imageWidth')
        H = data.get('imageHeight')
        if W is None or H is None:
            raise ValueError("imageWidth/Height missing and --out-size not provided.")

    shapes_out = []
    for sh in data.get('shapes', []):
        sh2 = deepcopy(sh)
        pts = sh2.get('points', [])
        new_pts = []
        for (x,y) in pts:
            x2, y2 = apply_affine_to_point(x, y, M)
            new_pts.append([x2,y2])
        sh2['points'] = new_pts
        sh2 = clean_shape_points(sh2, W, H)
        shapes_out.append(sh2)

    data2['shapes'] = shapes_out

    # imagePath: keep as-is; caller should update if needed after pasting
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data2, f, ensure_ascii=False, indent=2)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=str)
    ap.add_argument("output_json", type=str)
    ap.add_argument("--out-size", type=int, nargs=2, metavar=('W','H'), help="Output image size (width height) for clipping.")
    # High-level params
    ap.add_argument("--sx", type=float, default=1.0, help="Scale x")
    ap.add_argument("--sy", type=float, default=1.0, help="Scale y")
    ap.add_argument("--angle", type=float, default=0.0, help="Rotation in degrees (CCW)")
    ap.add_argument("--dx", type=float, default=0.0, help="Translate x (pixels)")
    ap.add_argument("--dy", type=float, default=0.0, help="Translate y (pixels)")
    # Direct matrix
    ap.add_argument("--matrix", type=float, nargs=6, help="a b tx c d ty (row-wise 2x3 affine)")
    return ap.parse_args()

def main():
    args = parse_args()
    M = build_affine(args)
    transform_labelme(args.input_json, args.output_json, args.out_size, M)

if __name__ == "__main__":
    main()
