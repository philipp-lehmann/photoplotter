"""Trace a single image to SVG without plotting (drawing-style test harness).

Run from the repo root, e.g.:
    python parsefile.py photos/test/1.jpg --style oneline --seed 1 --open
"""
import os
import sys
import shutil
import random
import argparse
import subprocess

import numpy as np

from photobooth.imageparser import ImageParser

# Set up argument parsing
parser = argparse.ArgumentParser(description="Trace an image to SVG without plotting (style test harness).")
parser.add_argument("file", help="Path to the source image (jpg/png).")
parser.add_argument("--style", default=None,
                    help="'+'-combinable styles: features, outline, shade, oneline (e.g. features+shade+oneline).")
parser.add_argument("--method", type=int, default=3, help="Contour method 1-4 (see extract_contours).")
parser.add_argument("--snap", default=None, choices=["none", "dynamic_grid", "poisson_disk"],
                    help="Force the point-snap style instead of the random pick.")
parser.add_argument("--max-paths", type=int, default=120)
parser.add_argument("--min-contour-area", type=int, default=16)
parser.add_argument("--radius", type=float, default=18, help="Feature-overlap radius in px (features style).")
parser.add_argument("--shades", type=int, default=2, help="Tone count incl. paper white (shade style).")
parser.add_argument("--spacing", type=float, default=10, help="Hatch line spacing in px (shade style).")
parser.add_argument("--seed", type=int, default=None, help="Seed RNGs for reproducible output.")
parser.add_argument("--no-depthmap", action="store_true")
parser.add_argument("--open", action="store_true", help="Open the resulting SVG (macOS).")
args = parser.parse_args()

valid_styles = {"default", "features", "outline", "shade", "oneline"}
if args.style:
    invalid = set(args.style.split("+")) - valid_styles
    if invalid:
        parser.error(f"Unknown style(s): {', '.join(sorted(invalid))}. Valid: {', '.join(sorted(valid_styles))}")

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

# Copy the input into a sandbox dir so _optimized/_depthmap siblings don't pollute the source dir
sandbox_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photos", "parsefile")
os.makedirs(sandbox_dir, exist_ok=True)
input_path = os.path.join(sandbox_dir, os.path.basename(args.file))
shutil.copy2(args.file, input_path)

style = None if args.style in (None, "default") else args.style
image_parser = ImageParser()
svg_path = image_parser.convert_to_svg(
    input_path,
    max_paths=args.max_paths,
    min_contour_area=args.min_contour_area,
    method=args.method,
    style=style,
    feature_radius=args.radius,
    shades=args.shades,
    hatch_spacing=args.spacing,
    snap_method=args.snap,
    apply_depthmap=not args.no_depthmap,
    suffix=f"-{args.style or 'default'}" + (f"-{args.snap}" if args.snap else ""),
)

if svg_path is None:
    print("Conversion failed.")
    sys.exit(1)

# Copy the result next to the input copy for easy inspection
final_path = os.path.join(sandbox_dir, os.path.basename(svg_path))
shutil.copy2(svg_path, final_path)

num_paths = final_path and sum(1 for line in open(final_path) if "<polyline" in line)
print(f"\nResult: {os.path.abspath(final_path)}")
print(f"Polylines: {num_paths} / Length: {image_parser.get_svgpath_length(final_path)}")

if args.open and sys.platform == "darwin":
    subprocess.run(["open", final_path])
