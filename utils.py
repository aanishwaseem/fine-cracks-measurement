

import os
from PIL import Image
import re
from remove_gridlines import remove_grid_pieces_only, inpaint_with_mask
import numpy as np
from skimage import io, exposure, img_as_ubyte
import cv2
import subprocess
import sys
from multiprocessing import Process, Queue
import time
import shutil
import threading
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
    """
    Extract cracks by running UNet and DeepCrack in PARALLEL.
    
    ✅ PARALLEL EXECUTION:
    - UNet inference runs on separate thread
    - DeepCrack extraction runs on separate thread
    - Both complete simultaneously, then merge
    """
    print("[INFO] Extracting all possible cracks (PARALLEL mode)...")
    
    tile_size = 512
    
    # Prepare UNet tiles (lightweight, can do upfront)
    make_tiles_fixed_size(img_gray, tile_size=tile_size)
    
    # Create threads for parallel execution
    import threading
    
    # Thread 1: Run UNet inference
    unet_result = {"status": "running", "data": None}
    def run_unet_thread():
        try:
            res = run_inference(f"tiles2_s", output_dir="experiment")
            reconstructed = join_tiles_after_inference(
                crack_segmentation_dir_string,
                "experiment",
                tile_size=tile_size,
                original_h=img_gray.shape[0],
                original_w=img_gray.shape[1],
                save=False
            )
            unet_result["data"] = reconstructed
            unet_result["status"] = "complete"
            print("[✓] UNet inference completed")
        except Exception as e:
            print(f"[✗] UNet inference failed: {e}")
            unet_result["status"] = "error"
    
    # Thread 2: Run DeepCrack extraction (in parallel)
    deepcrack_result = {"status": "running", "data": None}
    def run_deepcrack_thread():
        try:
            deepCrackImg = extract_deepcracks(deepcrack_img_with_grids, mask)
            deepcrack_result["data"] = deepCrackImg
            deepcrack_result["status"] = "complete"
            print("[✓] DeepCrack extraction completed")
        except Exception as e:
            print(f"[✗] DeepCrack extraction failed: {e}")
            deepcrack_result["status"] = "error"
    
    # Start both threads
    print("[→] Launching UNet inference thread...")
    t_unet = threading.Thread(target=run_unet_thread, daemon=False)
    t_unet.start()
    
    print("[→] Launching DeepCrack extraction thread...")
    t_deepcrack = threading.Thread(target=run_deepcrack_thread, daemon=False)
    t_deepcrack.start()
    
    # Wait for both to complete
    print("[⏳] Waiting for both threads to complete...")
    t_unet.join()  # Wait for UNet
    t_deepcrack.join()  # Wait for DeepCrack
    
    # Check if both succeeded
    if unet_result["status"] != "complete" or deepcrack_result["status"] != "complete":
        print(f"[ERROR] UNet: {unet_result['status']}, DeepCrack: {deepcrack_result['status']}")
        return None
    
    # Merge results
    print("[→] Merging results from both pipelines...")
    reconstructedImg = unet_result["data"]
    deepCrackImg = deepcrack_result["data"]
    
    merged = overlay_binary_images(reconstructedImg, deepCrackImg)
    print("[INFO] Crack extraction successful (parallel execution completed).")
    return merged

    # reconstructedImg = cv2.imread("CS_2.png")
    # deepCrackImg = cv2.imread("DC_More.png")
    # merged = overlay_binary_images(reconstructedImg, deepCrackImg)

    # print("[INFO] Crack extraction successful.")
    # return merged


def join_tiles_after_inference(dir_string, inference_res_dir, tile_size, original_h, original_w, ext="jpg", save=False):
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
    ⚡ THREADED: Saves tiles to disk in parallel when save_dir is given.
    """
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(img, Image.Image):
        img = np.array(img)

    H, W = img.shape[:2]
    pieces = []
    save_tasks = []
    count = 1

    for y in range(0, H, tile_size):
        for x in range(0, W, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size].copy()
            pieces.append(tile)
            if save_dir:
                save_tasks.append((tile, count))
            count += 1

    if save_dir and save_tasks:
        os.makedirs(save_dir, exist_ok=True)

        def _save_tile(args):
            tile, idx = args
            tile_img = Image.fromarray(tile)
            tile_img.save(os.path.join(save_dir, f"{idx}.png"))

        with ThreadPoolExecutor() as executor:
            list(executor.map(_save_tile, save_tasks))

    return pieces

def join_tiles_from_folder(folder_path, original_size, tile_size=256, ext="jpg"):
    """
    Join tiles from a folder back into the original image.
    ⚡ THREADED: Loads tiles from disk in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(ext)],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    def _load_tile(f):
        return np.array(Image.open(os.path.join(folder_path, f)))

    with ThreadPoolExecutor() as executor:
        tiles = list(executor.map(_load_tile, files))

    H, W = original_size
    if len(tiles[0].shape) == 3:
        C = tiles[0].shape[2]
        img = np.zeros((H, W, C), dtype=tiles[0].dtype)
    else:
        img = np.zeros((H, W), dtype=tiles[0].dtype)

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
    ⚡ THREADED: Processes tiles in parallel using ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor

    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    if not filenames:
        return

    def _process_one(filename):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        try:
            image = io.imread(in_path)
            image_contrast = night_vision_lab(image)
            io.imsave(out_path, image_contrast)
        except Exception as e:
            print(f"- Failed: {filename} → {e}")

    with ThreadPoolExecutor() as executor:
        list(executor.map(_process_one, filenames))



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


# ============================================================================
# PARALLEL EXECUTION FUNCTIONS (New - preserve original functionality)
# ============================================================================

def _run_unet_parallel_wrapper(img_gray, tile_size, output_q):
    """
    Wrapper function to run UNet inference in a separate process.
    """
    try:
        make_tiles_fixed_size(img_gray, tile_size=tile_size)
        result = run_inference(f"tiles2_s", output_dir="experiment")
        reconstructed = join_tiles_after_inference(
            crack_segmentation_dir_string,
            "experiment",
            tile_size=tile_size,
            original_h=img_gray.shape[0],
            original_w=img_gray.shape[1],
            save=False
        )
        output_q.put(("unet", reconstructed))
        print("[✓] UNet inference completed")
    except Exception as e:
        print(f"[✗] UNet inference failed: {e}")
        output_q.put(("unet", None))


def _run_deepcrack_parallel_wrapper(img_color, tile_size, output_q):
    """
    Wrapper function to run DeepCrack pipeline in a separate process.
    """
    try:
        from deepcrack_pipeline import start_deepcrack_pipeline
        result = start_deepcrack_pipeline(
            img_color,
            f"{deep_crack_dir_string}/input_tiles",
            original_h=img_color.shape[0],
            original_w=img_color.shape[1],
            tile_size=tile_size,
            inc_contrast=True
        )
        output_q.put(("deepcrack", result))
        print("[✓] DeepCrack pipeline completed")
    except Exception as e:
        print(f"[✗] DeepCrack pipeline failed: {e}")
        output_q.put(("deepcrack", None))


def run_both_inference_parallel(img_gray, img_color, tile_size=512):
    """
    Run UNet inference and DeepCrack pipeline in parallel.
    
    This is a parallel alternative to running them sequentially.
    Original functions (getBinaryImage, run_inference, etc.) remain unchanged.
    
    Args:
        img_gray (np.ndarray): Grayscale image for UNet
        img_color (np.ndarray): Color image for DeepCrack
        tile_size (int): Tile size for splitting (default 512)
    
    Returns:
        dict: Dictionary with keys 'unet' and 'deepcrack' containing results
              Returns {'unet': result_array, 'deepcrack': result_array}
              Returns {'unet': None, 'deepcrack': None} if either fails
    
    Example:
        results = run_both_inference_parallel(img_gray, img_color, tile_size=512)
        unet_output = results['unet']
        deepcrack_output = results['deepcrack']
    """
    print("[INFO] Starting parallel UNet + DeepCrack processing...")
    start_time = time.time()
    
    output_queue = Queue()
    processes = []
    
    # Start UNet process
    print("[→] Launching UNet inference...")
    p_unet = Process(target=_run_unet_parallel_wrapper, args=(img_gray, tile_size, output_queue))
    p_unet.start()
    processes.append(("UNet", p_unet))
    
    # Start DeepCrack process
    print("[→] Launching DeepCrack pipeline...")
    p_deepcrack = Process(target=_run_deepcrack_parallel_wrapper, args=(img_color, tile_size, output_queue))
    p_deepcrack.start()
    processes.append(("DeepCrack", p_deepcrack))
    
    # Wait for all processes to complete
    for name, process in processes:
        process.join()
        print(f"[✓] {name} process finished")
    
    # Collect results from queue
    results = {}
    while not output_queue.empty():
        key, value = output_queue.get()
        results[key] = value
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Parallel processing completed in {elapsed_time:.2f}s")
    
    return results

# ============================================================================

# img = cv2.imread("C-sample-sharp4.jpg")

# tile_size = 512
# make_tiles_fixed_size(img, tile_size=tile_size)
# enhance_contrast_in_directory(f"{crack_segmentation_dir_string}/tiles2_s")
# res = run_inference(f"tiles2_s", output_dir="experiment")
# print("Running final step.. Please wait a minute")
# reconstructedImg = join_tiles_after_inference(crack_segmentation_dir_string,"experiment", tile_size=tile_size, original_h=img.shape[0], original_w=img.shape[1], save=True)
