"""
Headless crack-extraction pipeline (no GUI windows, no gridline removal, no IOPaint).

Mirrors crack.py -> process_image():
    1. upscale the image x2 (bicubic, or RealESRGAN through IOPaint with --iopaint like crack.py)
    2. UNet (models/crack_segmentation) on 300px tiles
    3. DeepCrack (models/DeepCrack) on 400px tiles with contrast enhancement
    4. merge both predictions (pixel-wise max)
    5. threshold -> binary crack mask
    6. keep only cracks inside the dataset's reference mask

Usage:
    python run_pipeline.py --image "<path to image>" [--reference <ref.png>] [--out outputs] [--iopaint]

--iopaint needs the IOPaint server on port 8000 (see "iopaint commands").

If --reference is omitted, references/<dataset folder>_ref.png is used
(same naming as gen_ref_mask.get_reference_path).
"""
import argparse
import os
import re

import cv2
import numpy as np

from utils import (make_tiles_fixed_size, run_inference, join_tiles_after_inference,
                   overlay_binary_images, crack_segmentation_dir_string)
from deepcrack_pipeline import start_deepcrack_pipeline, deep_crack_dir_string
from remove_gridlines import intersect_masks

SCALE_FACTOR = 2        # crack.py: scale_image_factor
UNET_TILE_SIZE = 300    # crack.py: _TILE_SIZE
DEEPCRACK_TILE_SIZE = 400  # remove_gridlines(): tile_size when not making a reference
THRESHOLD_VALUE = 23    # crack.py: threshold_value
REFERENCE_FOLDER = "references"


def get_reference_path(dataset_folder):
    name = re.sub(r'[^\w\-]', '_', os.path.basename(os.path.normpath(dataset_folder)))
    return os.path.join(REFERENCE_FOLDER, f"{name}_ref.png")


def upscale(img, scale):
    # Same size rule as scale_image.scale_image(), without the RealESRGAN server.
    if (scale == 2 and img.shape[1] > 1800) or (scale == 4 and img.shape[1] > 3000):
        return img
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)


def get_binary_image_of_cracks(gen_binary_mask, threshold=23, alpha=0.68, beta=12):
    # Copied from crack.py (importing crack.py would start its tkinter GUI dependencies).
    brightened_image = cv2.convertScaleAbs(gen_binary_mask.copy(), alpha=alpha, beta=beta)
    _, binary_image = cv2.threshold(brightened_image, threshold, 255, cv2.THRESH_BINARY_INV)
    return 255 - binary_image


def run_unet(image):
    make_tiles_fixed_size(image, tile_size=UNET_TILE_SIZE)
    if not run_inference("tiles2_s", output_dir="experiment"):
        raise RuntimeError("UNet inference failed")
    return join_tiles_after_inference(crack_segmentation_dir_string, "experiment",
                                      tile_size=UNET_TILE_SIZE,
                                      original_h=image.shape[0], original_w=image.shape[1])


def run_deepcrack(image, tile_size=DEEPCRACK_TILE_SIZE):
    # Smaller tiles are upscaled more before DeepCrack (tiles are resized to 512px),
    # so thinner cracks get detected.
    return start_deepcrack_pipeline(image, f"{deep_crack_dir_string}/input_tiles",
                                    original_h=image.shape[0], original_w=image.shape[1],
                                    tile_size=tile_size, inc_contrast=True)


def main():
    parser = argparse.ArgumentParser(description="Extract a binary crack mask from one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_VALUE)
    parser.add_argument("--iopaint", action="store_true",
                        help="upscale with RealESRGAN via the IOPaint server (scale_image.py)")
    parser.add_argument("--deepcrack-tile", type=int, default=DEEPCRACK_TILE_SIZE,
                        help="DeepCrack tile size in px; lower = more aggressive")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    ref_path = args.reference or get_reference_path(os.path.dirname(os.path.abspath(args.image)))
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(ref_path)
    print(f"[INFO] image: {args.image} {image.shape}")
    print(f"[INFO] reference: {ref_path} {reference.shape}")

    if args.iopaint:
        from scale_image import scale_image
        print("[→] RealESRGAN x2 upscale via IOPaint ...")
        image = scale_image(image, SCALE_FACTOR)
    else:
        image = upscale(image, SCALE_FACTOR)
    if reference.shape != image.shape[:2]:
        print(f"[WARN] resizing reference {reference.shape} -> {image.shape[:2]}")
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    print("[→] UNet ...")
    unet = run_unet(image)
    print("[→] DeepCrack ...")
    print(f"[INFO] DeepCrack tile size: {args.deepcrack_tile}")
    deepcrack = run_deepcrack(image, args.deepcrack_tile)

    merged = overlay_binary_images(unet, deepcrack)
    binary = get_binary_image_of_cracks(merged, args.threshold)
    binary = intersect_masks(reference, binary)

    os.makedirs(args.out, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(args.image))[0])
    if args.deepcrack_tile != DEEPCRACK_TILE_SIZE:
        stem += f"_dc{args.deepcrack_tile}"
    if args.iopaint:
        stem += "_iopaint"
    overlay = image.copy()
    overlay[binary > 0] = (0, 0, 255)
    outputs = {
        f"{stem}_binary.png": binary,
        f"{stem}_overlay.png": overlay,
        f"{stem}_unet_raw.png": unet,
        f"{stem}_deepcrack_raw.png": deepcrack,
    }
    for name, img in outputs.items():
        cv2.imwrite(os.path.join(args.out, name), img)
        print(f"[✓] saved {os.path.join(args.out, name)}")
    print(f"[INFO] crack pixels: {int(np.count_nonzero(binary))}")


if __name__ == "__main__":
    main()
