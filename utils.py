

import os
from PIL import Image
import re
from remove_gridlines import remove_grid_pieces_only, inpaint_with_mask
import numpy as np
from skimage import io, exposure, img_as_ubyte
import cv2
import subprocess
import sys
crack_segmentation_dir_string = "models/crack_segmentation"
crack_segmentation_dir = os.path.join(os.getcwd(), crack_segmentation_dir_string)

deep_crack_dir_string = "models/DeepCrack/codes"
deep_crack_dir = os.path.join(os.getcwd(), deep_crack_dir_string)
endpoint = 'http://127.0.0.1:8000'
def run_inference(tiles_dir, output_dir="experiment"):
    experiment_dir = os.path.join(crack_segmentation_dir, output_dir)
    empty_folder(experiment_dir)
    command = [
        sys.executable,
        "inference_unet.py",
        "-img_dir", f"./{tiles_dir}",
        "-model_path", "models/model_unet_vgg16_best.pt",
        "-model_type", "vgg16",
        "-out_viz_dir", "viz_result",
        "-out_pred_dir", output_dir,
        "-threshold", "0.2"
    ]

    try:
        result = subprocess.run(
            command,
            cwd=crack_segmentation_dir,
            check=True
        )
    except subprocess.CalledProcessError:
        print("- Inference process failed.")
        return False

    if result.returncode == 0:
        return True
    else:
        return False


    


def extract_deepcracks(deepcrack_img_with_grids, mask):
    mask_improved = remove_grid_pieces_only(mask)
    # cv2.imwrite("deepcrack_results.png",deepcrack_img_with_grids)
    # cv2.imwrite("deepcrack_results_mask.png",mask_improved)
    output = inpaint_with_mask(
        image_np=deepcrack_img_with_grids,
        mask_np=mask_improved,
        url=f"{endpoint}/api/v1/inpaint",
        prompt="",
        cv2_radius=4
    )
    return output
def getBinaryImage(img_gray, deepcrack_img_with_grids, mask):
    print("[INFO] Extracting all possible cracks..")

    tile_size = 256
    make_tiles_fixed_size(img_gray, tile_size=tile_size)
    try:
        res = run_inference(f"tiles2_s", output_dir="experiment")
        print("Running final step.. Please wait a minute")
        reconstructedImg = join_tiles_after_inference(crack_segmentation_dir_string,"experiment", tile_size=tile_size, original_h=img_gray.shape[0], original_w=img_gray.shape[1], save=True)
        deepCrackImg = extract_deepcracks(deepcrack_img_with_grids, mask)
        # cv2.imwrite("deepcrack_results.png",deepcrack_img_with_grids)
        merged = overlay_binary_images(reconstructedImg, deepCrackImg)
        print("[INFO] Crack extraction successful.")
        return merged
    except Exception as e:
        print(f"[ERROR] Crack extraction failed: {e}")
        return None

    # reconstructedImg = cv2.imread("CS_2.png")
    # deepCrackImg = cv2.imread("DC_More.png")
    # merged = overlay_binary_images(reconstructedImg, deepCrackImg)

    # print("[INFO] Crack extraction successful.")
    # return merged


def join_tiles_after_inference(dir_string, inference_res_dir, tile_size, original_h, original_w, ext="jpg", save=True):
    folder = os.path.join(dir_string, inference_res_dir)
    original_size = (original_h, original_w)  # (H, W)

    reconstructed = join_tiles_from_folder(
        folder,
        original_size,
        tile_size,
        ext=ext
    )

    # Ensure uint8 for display & saving
    if reconstructed.dtype != 'uint8':
        reconstructed = (reconstructed * 255).astype('uint8')

    # Save (optional)
    if save:
        Image.fromarray(reconstructed).save("reconstructed_from_folder.png")

    return reconstructed


def make_tiles_fixed_size(img, tile_size=512, save_dir=f"{crack_segmentation_dir_string}/tiles2_s"):
    """
    Splits an image into fixed-size tiles.
    """
    tiles_dir = os.path.join(os.getcwd(), save_dir)
    empty_folder(tiles_dir)
    tiles = split_image(img, tile_size=tile_size, save_dir=save_dir)

    # print(f"Total tiles: {len(tiles)}")
    # print("Number of tiles:", len(tiles))
    # for i, t in enumerate(tiles):
    #     print(i+1, t.shape)  # see size of each tile


def split_image(img, tile_size=256, save_dir=None):
    """
    Split an image into tiles of size `tile_size x tile_size`.
    Leftover edges are kept as smaller tiles.

    Args:
        img (PIL.Image or np.array): Input image.
        tile_size (int): Tile size (default 256).
        save_dir (str): Optional directory to save tiles as numbered PNGs.

    Returns:
        pieces (list of np.array): List of image tiles.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    H, W = img.shape[:2]
    pieces = []
    count = 1

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size].copy()
            pieces.append(tile)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                tile_img = Image.fromarray(tile)
                tile_img.save(os.path.join(save_dir, f"{count}.png"))

            count += 1

    return pieces

def join_tiles_from_folder(folder_path, original_size, tile_size=256, ext="jpg"):
    """
    Join tiles from a folder back into the original image.

    Args:
        folder_path (str): Path to the folder containing tiles.
        original_size (tuple): Original image size (H, W)
        tile_size (int): Tile size used when splitting
        ext (str): File extension of tiles (default "png")

    Returns:
        img (np.array): Reconstructed image
    """
    # Get list of files sorted by number
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(ext)],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    # Load tiles
    tiles = [np.array(Image.open(os.path.join(folder_path, f))) for f in files]

    # Determine channels
    H, W = original_size
    if len(tiles[0].shape) == 3:
        C = tiles[0].shape[2]
        img = np.zeros((H, W, C), dtype=tiles[0].dtype)
    else:
        img = np.zeros((H, W), dtype=tiles[0].dtype)

    # Place tiles
    count = 0
    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = tiles[count]
            h, w = tile.shape[:2]
            img[y:y+h, x:x+w] = tile
            count += 1

    return img





import cv2

def overlay_binary_images(base_img, overlay_img):
    """
    Overlay two images (black & white) by taking the max pixel value.
    Both images are converted to grayscale if needed.
    """
    if base_img is None or overlay_img is None:
        raise ValueError("One of the images is None!")

    # Convert to grayscale if 3 channels
    if len(base_img.shape) == 3 and base_img.shape[2] == 3:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 3:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)

    # Resize overlay_img to match base_img if shapes differ
    if base_img.shape != overlay_img.shape:
        overlay_img = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Merge by taking maximum pixel
    merged_img = np.maximum(base_img, overlay_img)

    return merged_img

def empty_folder(folder_path):
    """
    Deletes all files and subfolders inside folder_path,
    but keeps the folder itself.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(f"⚠️ Failed to delete {path}: {e}")











def night_vision_lab(img):
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1️⃣ Reduce brightness but keep structure
    l = cv2.convertScaleAbs(l, alpha=0.6, beta=0)

    # 2️⃣ Desaturate heavily (pull towards 128 = neutral)
    a = cv2.addWeighted(a, 0.3, np.full_like(a, 128), 0.7, 0)
    b = cv2.addWeighted(b, 0.3, np.full_like(b, 128), 0.7, 0)

    # Merge back
    lab_nv = cv2.merge((l, a, b))
    bgr_nv = cv2.cvtColor(lab_nv, cv2.COLOR_LAB2BGR)

    # 3️⃣ Apply green tint (classic night vision)
    bgr_nv = bgr_nv.astype(np.float32)
    bgr_nv[:, :, 1] *= 1.4   # boost green
    bgr_nv[:, :, 0] *= 0.6   # reduce blue
    bgr_nv[:, :, 2] *= 0.6   # reduce red

    # Clip and convert
    bgr_nv = np.clip(bgr_nv, 0, 255).astype(np.uint8)

    return bgr_nv


def enhance_contrast_in_directory(
    input_dir,
    output_dir=None,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
):
    """
    Apply contrast enhancement to all images in a directory.

    Args:
        input_dir (str): folder with input images
        output_dir (str): folder to save results (if None, overwrite input)
        extensions (tuple): valid image extensions
    """

    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(extensions):
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        try:
            image = io.imread(in_path)
            image_contrast = night_vision_lab(image)
            io.imsave(out_path, image_contrast)

        except Exception as e:
            print(f"- Failed: {filename} → {e}")



def enhance_contrast(image):
    # Drop alpha if RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert single-channel 3D → 2D
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    # Apply CLAHE
    if image.ndim == 2:  # grayscale
        image_eq = exposure.equalize_adapthist(image, clip_limit=0.03)
    else:  # RGB
        image_eq = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            image_eq[:, :, c] = exposure.equalize_adapthist(
                image[:, :, c], clip_limit=0.03
            )

    return img_as_ubyte(image_eq)
def resize_image(img, w, h):

    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

# img = cv2.imread("C-sample-sharp4.jpg")

# tile_size = 512
# make_tiles_fixed_size(img, tile_size=tile_size)
# enhance_contrast_in_directory(f"{crack_segmentation_dir_string}/tiles2_s")
# res = run_inference(f"tiles2_s", output_dir="experiment")
# print("Running final step.. Please wait a minute")
# reconstructedImg = join_tiles_after_inference(crack_segmentation_dir_string,"experiment", tile_size=tile_size, original_h=img.shape[0], original_w=img.shape[1], save=True)
