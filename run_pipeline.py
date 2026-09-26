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


def run_deepcrack_only(image, args):
    print(f"[INFO] image: {args.image} {image.shape}")
    print(f"[→] DeepCrack only (tile size {args.deepcrack_tile}) ...")
    deepcrack = run_deepcrack(image, args.deepcrack_tile)
    binary = get_binary_image_of_cracks(deepcrack, args.threshold)
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    os.makedirs(args.out, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(args.image))[0]) + "_deepcrack_only"
    for name, img in {f"{stem}_binary.png": binary, f"{stem}_raw.png": deepcrack}.items():
        cv2.imwrite(os.path.join(args.out, name), img)
        print(f"[✓] saved {os.path.join(args.out, name)}")
    print(f"[INFO] crack pixels: {int(np.count_nonzero(binary))}")


def run_unet_only(image, args):
    print(f"[INFO] image: {args.image} {image.shape}")
    if args.upscale > 1:
        if args.iopaint:
            from scale_image import scale_image
            print(f"[→] RealESRGAN x{args.upscale} upscale via IOPaint ...")
            image = scale_image(image, args.upscale)
        else:
            image = upscale(image, args.upscale)
        print(f"[INFO] upscaled to {image.shape}")
    print("[→] UNet only ...")
    unet = run_unet(image)
    binary = get_binary_image_of_cracks(unet, args.threshold)
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    os.makedirs(args.out, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(args.image))[0]) + "_unet_only"
    if args.upscale > 1:
        stem += f"_x{args.upscale}" + ("_iopaint" if args.iopaint else "")
    for name, img in {f"{stem}_binary.png": binary, f"{stem}_raw.png": unet}.items():
        cv2.imwrite(os.path.join(args.out, name), img)
        print(f"[✓] saved {os.path.join(args.out, name)}")
    print(f"[INFO] crack pixels: {int(np.count_nonzero(binary))}")


def main():
    parser = argparse.ArgumentParser(description="Extract a binary crack mask from one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_VALUE)
    parser.add_argument("--iopaint", action="store_true",
                        help="upscale with RealESRGAN via the IOPaint server (scale_image.py)")
    parser.add_argument("--size", default=None, metavar="WxH",
                        help="resize image and reference to WxH (e.g. 448x224); the models still run "
                             "on the x2 upscaled image and the result is brought back to WxH")
    parser.add_argument("--deepcrack-only", action="store_true",
                        help="run only DeepCrack on the image as-is (no upscale, UNet or reference) "
                             "and threshold its output")
    parser.add_argument("--unet-only", action="store_true",
                        help="run only UNet (no DeepCrack or reference) and threshold its output")
    parser.add_argument("--upscale", type=int, default=None,
                        help="upscale factor before the models (2 or 4; RealESRGAN with --iopaint). "
                             "Default: 2 for the full pipeline, 1 for --unet-only")
    parser.add_argument("--deepcrack-tile", type=int, default=DEEPCRACK_TILE_SIZE,
                        help="DeepCrack tile size in px; lower = more aggressive")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    if args.deepcrack_only:
        run_deepcrack_only(image, args)
        return
    if args.unet_only:
        args.upscale = args.upscale or 1
        run_unet_only(image, args)
        return
    ref_path = args.reference or get_reference_path(os.path.dirname(os.path.abspath(args.image)))
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(ref_path)
    print(f"[INFO] image: {args.image} {image.shape}")
    print(f"[INFO] reference: {ref_path} {reference.shape}")

    size = None
    if args.size:
        size = tuple(int(v) for v in args.size.lower().split("x"))  # (W, H)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # INTER_AREA + ">0" keeps thin reference cracks that nearest-neighbour would drop
        reference = np.where(cv2.resize(reference, size, interpolation=cv2.INTER_AREA) > 0, 255, 0).astype(np.uint8)
        print(f"[INFO] resized image and reference to {size[0]}x{size[1]}")
    input_image = image

    factor = args.upscale or SCALE_FACTOR
    if args.iopaint:
        from scale_image import scale_image
        print(f"[→] RealESRGAN x{factor} upscale via IOPaint ...")
        image = scale_image(image, factor)
    else:
        image = upscale(image, factor)
    print(f"[INFO] upscaled to {image.shape}")
    if size is None and reference.shape != image.shape[:2]:
        print(f"[WARN] resizing reference {reference.shape} -> {image.shape[:2]}")
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    print("[→] UNet ...")
    unet = run_unet(image)
    print("[→] DeepCrack ...")
    print(f"[INFO] DeepCrack tile size: {args.deepcrack_tile}")
    deepcrack = run_deepcrack(image, args.deepcrack_tile)

    merged = overlay_binary_images(unet, deepcrack)
    if size is not None:
        merged = cv2.resize(merged, size, interpolation=cv2.INTER_AREA)
        image = input_image
    binary_no_ref = get_binary_image_of_cracks(merged, args.threshold)
    binary = intersect_masks(reference, binary_no_ref)

    os.makedirs(args.out, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(args.image))[0])
    if args.deepcrack_tile != DEEPCRACK_TILE_SIZE:
        stem += f"_dc{args.deepcrack_tile}"
    if factor != SCALE_FACTOR:
        stem += f"_x{factor}"
    if args.iopaint:
        stem += "_iopaint"
    if size is not None:
        stem += f"_{size[0]}x{size[1]}"
    overlay = image.copy()
    overlay[binary > 0] = (0, 0, 255)
    outputs = {
        f"{stem}_binary.png": binary,
        f"{stem}_binary_no_reference.png": binary_no_ref,
        f"{stem}_overlay.png": overlay,
        f"{stem}_unet_raw.png": unet,
        f"{stem}_deepcrack_raw.png": deepcrack,
    }
    if size is not None:
        outputs[f"{stem}_reference.png"] = reference
    for name, img in outputs.items():
        cv2.imwrite(os.path.join(args.out, name), img)
        print(f"[✓] saved {os.path.join(args.out, name)}")
    print(f"[INFO] crack pixels: {int(np.count_nonzero(binary))} "
          f"(without reference: {int(np.count_nonzero(binary_no_ref))})")


if __name__ == "__main__":
    main()
