import os
import subprocess
import sys
import numpy as np
from PIL import Image
import cv2
import shutil
import re
from skimage import io, exposure, img_as_ubyte
from multiprocessing import Process, Queue
import time
import hashlib
import threading

crack_segmentation_dir_string = "models/crack_segmentation"
crack_segmentation_dir = os.path.join(os.getcwd(), crack_segmentation_dir_string)
deep_crack_dir_string = "models/DeepCrack/codes"
deep_crack_dir = os.path.join(os.getcwd(), deep_crack_dir_string)

# ✅ OPTIMIZATION: Cache for DeepCrack results
_deepcrack_result_cache = {}

def _run_unet_inference_wrapper(tiles_dir, output_q):
    """
    Wrapper function to run UNet inference in a separate process.
    Results are put in the queue.
    """
    try:
        from utils import run_inference
        result = run_inference(tiles_dir, output_dir="experiment")
        output_q.put(("unet", result))
        print("[✓] UNet inference completed")
    except Exception as e:
        print(f"[✗] UNet inference failed: {e}")
        output_q.put(("unet", False))

def _run_deepcrack_wrapper(img, tiles_dir, original_h, original_w, tile_size, inc_contrast, output_q):
    """
    Wrapper function to run DeepCrack pipeline in a separate process.
    Results are put in the queue.
    """
    try:
        result = start_deepcrack_pipeline(img, tiles_dir, original_h, original_w, tile_size, inc_contrast)
        output_q.put(("deepcrack", result))
        print("[✓] DeepCrack pipeline completed")
    except Exception as e:
        print(f"[✗] DeepCrack pipeline failed: {e}")
        output_q.put(("deepcrack", None))

def run_both_pipelines_parallel(img, tiles_dir_unet, tiles_dir_deepcrack, original_h, original_w, 
                                tile_size=512, inc_contrast=True, run_unet=True, run_deepcrack=True):
    """
    Run UNet inference and DeepCrack pipeline in parallel.
    
    Args:
        img (np.ndarray): Input image
        tiles_dir_unet (str): Directory for UNet tiles (from crack_segmentation)
        tiles_dir_deepcrack (str): Directory for DeepCrack tiles
        original_h (int): Original image height
        original_w (int): Original image width
        tile_size (int): Tile size (default 512)
        inc_contrast (bool): Whether to enhance contrast
        run_unet (bool): Whether to run UNet inference
        run_deepcrack (bool): Whether to run DeepCrack pipeline
    
    Returns:
        dict: Dictionary with 'unet' and 'deepcrack' results
    """
    print("[INFO] Starting parallel processing...")
    start_time = time.time()
    
    output_queue = Queue()
    processes = []
    
    # Start UNet inference process
    if run_unet:
        print("[→] Launching UNet inference...")
        p_unet = Process(target=_run_unet_inference_wrapper, args=(tiles_dir_unet, output_queue))
        p_unet.start()
        processes.append(("unet", p_unet))
    
    # Start DeepCrack pipeline process
    if run_deepcrack:
        print("[→] Launching DeepCrack pipeline...")
        p_deepcrack = Process(target=_run_deepcrack_wrapper, 
                            args=(img, tiles_dir_deepcrack, original_h, original_w, tile_size, inc_contrast, output_queue))
        p_deepcrack.start()
        processes.append(("deepcrack", p_deepcrack))
    
    # Wait for all processes to complete
    results = {}
    for name, process in processes:
        process.join()
        print(f"[✓] {name.upper()} process finished")
    
    # Collect results from queue
    while not output_queue.empty():
        key, value = output_queue.get()
        results[key] = value
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Parallel processing completed in {elapsed_time:.2f}s")
    
    return results


def run_deepcrack_and_unet_parallel(img, img_gray, tiles_dir_deepcrack, deepcrack_mask_thickness=2, 
                                     original_h=None, original_w=None, tile_size=512, inc_contrast=True):
    """
    Run DeepCrack and UNet inference simultaneously on SEPARATE THREADS.
    Both run in parallel without waiting for each other.
    
    ⚡ KEY OPTIMIZATION:
    - DeepCrack preprocessing, inference, and restoration runs on Thread A
    - UNet preprocessing and inference runs on Thread B
    - Both threads execute completely in parallel
    - Results are merged after both complete
    
    Args:
        img (np.ndarray): Color image for DeepCrack
        img_gray (np.ndarray): Grayscale image for UNet
        tiles_dir_deepcrack (str): Directory for DeepCrack tiles
        deepcrack_mask_thickness (int): Thickness for grid mask
        original_h (int): Original height
        original_w (int): Original width
        tile_size (int): Tile size (default 512)
        inc_contrast (bool): Enhance contrast for DeepCrack
    
    Returns:
        tuple: (deepcrack_result, unet_result, status_dict)
    """
    import threading
    
    print("[⚡] PARALLEL EXECUTION: Running DeepCrack and UNet simultaneously...")
    start_time = time.time()
    
    # Storage for results from threads
    deepcrack_result = {"status": "running", "data": None}
    unet_result = {"status": "running", "data": None}
    
    # ====== THREAD 1: DeepCrack Pipeline ======
    def run_deepcrack_thread():
        try:
            print("[→] [Thread-DeepCrack] Starting DeepCrack pipeline...")
            deepcrack_img = start_deepcrack_pipeline(
                img,
                tiles_dir_deepcrack,
                original_h=original_h or img.shape[0],
                original_w=original_w or img.shape[1],
                tile_size=tile_size,
                inc_contrast=inc_contrast
            )
            deepcrack_result["data"] = deepcrack_img
            deepcrack_result["status"] = "complete"
            print("[✓] [Thread-DeepCrack] DeepCrack pipeline completed")
        except Exception as e:
            print(f"[✗] [Thread-DeepCrack] DeepCrack failed: {e}")
            deepcrack_result["status"] = "error"
            deepcrack_result["error"] = str(e)
    
    # ====== THREAD 2: UNet Inference ======
    def run_unet_thread():
        try:
            from utils import make_tiles_fixed_size, run_inference, join_tiles_after_inference, crack_segmentation_dir_string
            
            print("[→] [Thread-UNet] Starting UNet inference...")
            make_tiles_fixed_size(img_gray, tile_size=tile_size, save_dir=f"{crack_segmentation_dir_string}/tiles2_s")
            res = run_inference(f"tiles2_s", output_dir="experiment")
            
            if res:
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
                print("[✓] [Thread-UNet] UNet inference completed")
            else:
                unet_result["status"] = "error"
                print("[✗] [Thread-UNet] UNet inference failed")
        except Exception as e:
            print(f"[✗] [Thread-UNet] UNet failed: {e}")
            unet_result["status"] = "error"
            unet_result["error"] = str(e)
    
    # Launch both threads simultaneously
    print("[→] Creating and starting threads...")
    t_deepcrack = threading.Thread(target=run_deepcrack_thread, name="DeepCrack-Thread", daemon=False)
    t_unet = threading.Thread(target=run_unet_thread, name="UNet-Thread", daemon=False)
    
    print("[→] Starting DeepCrack thread...")
    t_deepcrack.start()
    
    print("[→] Starting UNet thread...")
    t_unet.start()
    
    # Wait for both threads to complete
    print("[⏳] Waiting for both DeepCrack and UNet threads to complete...")
    t_deepcrack.join()
    print("[✓] DeepCrack thread finished")
    
    t_unet.join()
    print("[✓] UNet thread finished")
    
    elapsed_time = time.time() - start_time
    print(f"[⚡] PARALLEL EXECUTION COMPLETED in {elapsed_time:.2f}s")
    
    # Return results and status
    status = {
        "deepcrack_status": deepcrack_result["status"],
        "unet_status": unet_result["status"],
        "total_time": elapsed_time,
        "deepcrack_error": deepcrack_result.get("error"),
        "unet_error": unet_result.get("error")
    }
    
    return deepcrack_result["data"], unet_result["data"], status


def _compute_image_hash(img):
    """Compute MD5 hash of an image for caching."""
    return hashlib.md5(cv2.imencode('.png', img)[1]).hexdigest()


def _get_deepcrack_cache_key(img, tile_size, inc_contrast):
    """Generate cache key for DeepCrack results."""
    img_hash = _compute_image_hash(img)
    return f"{img_hash}_{tile_size}_{inc_contrast}"


def start_deepcrack_pipeline(img, tiles_dir, original_h, original_w, tile_size=512, inc_contrast=True):
    """
    Run DeepCrack pipeline with caching.
    Results are cached based on image hash to avoid reprocessing identical images.
    """
    # ✅ OPTIMIZATION: Check cache first
    cache_key = _get_deepcrack_cache_key(img, tile_size, inc_contrast)
    
    if cache_key in _deepcrack_result_cache:
        print(f"[✓ CACHED] DeepCrack result retrieved from cache (key: {cache_key[:16]}...)")
        return _deepcrack_result_cache[cache_key]
    
    print(f"[→] DeepCrack processing (cache key: {cache_key[:16]}...)")
    
    make_tiles_fixed_size(img, tile_size=tile_size, save_dir=tiles_dir)
    if inc_contrast:
        enhance_contrast_in_directory(tiles_dir)
    rename_images_with_original_size(tiles_dir, target_size=512)
    resize_images_in_dir(tiles_dir)
    generate_test_example(tiles_dir)
    run_deepcrack()
    restore_images_to_original_size(f"{deep_crack_dir_string}/deepcrack_results")
    reconstructedImg = join_tiles_after_inference(deep_crack_dir_string,"deepcrack_results", tile_size=tile_size, original_h=original_h, original_w=original_w, ext="png", save=True)
    
    # ✅ OPTIMIZATION: Store in cache
    _deepcrack_result_cache[cache_key] = reconstructedImg
    
    return reconstructedImg

def run_deepcrack(output_dir="deepcrack_results"):
    experiment_dir = os.path.join(deep_crack_dir, output_dir)
    empty_folder(experiment_dir)
    command = [
        sys.executable,
        "test.py"
    ]
    try:
        result = subprocess.run(
            command,
            cwd=deep_crack_dir, 
            # stdout=subprocess.DEVNULL,   # suppress normal output
            # stderr=subprocess.DEVNULL,   # suppress errors
            check=True
        )
    except subprocess.CalledProcessError:
        print("- Deep Crack process failed.")
        return False

    if result.returncode == 0:
        return True
    else:
        return False
    
    

def make_tiles_fixed_size(img, tile_size=512, save_dir=f"{crack_segmentation_dir_string}/tiles2_s"):
    """
    Splits an image into fixed-size tiles.
    """
    tiles_dir = os.path.join(os.getcwd(), save_dir)
    empty_folder(tiles_dir)
    tiles = split_image(img, tile_size=tile_size, save_dir=save_dir)

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
    save_tasks = []  # (tile, count) pairs for parallel saving
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


def enhance_contrast_in_directory(
    input_dir,
    output_dir=None,
    extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
):
    """
    Apply contrast enhancement to all images in a directory.
    ⚡ THREADED: Processes tiles in parallel using ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

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
            image_contrast = clahe_plus_hist_stretch_safe(image)
            io.imsave(out_path, image_contrast)
        except Exception as e:
            print(f"- Failed: {filename} → {e}")

    with ThreadPoolExecutor() as executor:
        list(executor.map(_process_one, filenames))


def rename_images_with_original_size(img_dir, target_size):
    """
    Rename all images in a directory to include their original width and height.

    Args:
        img_dir (str): Directory containing images.
        target_size (int): Target size (not used for resizing here, just passed as param).
    """
    img_dir = os.path.abspath(img_dir)
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory does not exist: {img_dir}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_files.sort()

    if not img_files:
        print(f"No images found in {img_dir}.")
        return

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        h, w = img.shape[:2]
        name, ext = os.path.splitext(img_name)
        new_name = f"{name}_{w}x{h}{ext}"
        new_path = os.path.join(img_dir, new_name)

        # Rename the file
        os.rename(img_path, new_path)

    # print(f"Processed {len(img_files)} images.")





def resize_images_in_dir(img_dir, overwrite=True, save_dir=None):
    """
    Resize all PNG/JPG images in a directory to 512x512.
    ⚡ THREADED: Processes tiles in parallel using ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor

    img_dir = os.path.abspath(img_dir)

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory does not exist: {img_dir}")

    if save_dir:
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_files.sort()

    if not img_files:
        print(f"No images found in {img_dir}.")
        return

    def _resize_one(img_name):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}. Skipping.")
            return
        resized_img = resize_image_512(img)
        save_path = os.path.join(save_dir, img_name) if save_dir else img_path
        cv2.imwrite(save_path, resized_img)

    with ThreadPoolExecutor() as executor:
        list(executor.map(_resize_one, img_files))



def generate_test_example(crack_segmentation_dir_string: str):
    """
    Generates a test_example.txt file listing all PNGs in the directory.
    Each line contains: <full_image_path>\t<full_image_path>
    
    Args:
        crack_segmentation_dir_string (str): Path to directory containing PNG images
    """
    # Ensure absolute path
    img_dir = os.path.abspath(crack_segmentation_dir_string)
    txt_file_string = os.path.join(f"{deep_crack_dir_string}/data", "test_example.txt")
    txt_file = os.path.abspath(txt_file_string)
    os.makedirs(os.path.dirname(txt_file), exist_ok=True)
    # List all PNGs
    png_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    png_files.sort()  # optional: sort alphabetically or numerically

    if not png_files:
        print(f"No PNG files found in {img_dir}. test_example.txt will be empty.")

    # Empty the file first and write full paths
    with open(txt_file, "w") as f:
        for img_name in png_files:
            full_path = os.path.join(img_dir, img_name).replace("\\", "/")
            f.write(f'"{full_path}" "{full_path}"\n')

    # print(f"test_example.txt generated at {txt_file} with {len(png_files)} entries.")

def restore_images_to_original_size_padded(img_dir):
    """
    Restore images to original size by REMOVING padding (no resizing).
    Assumes images were padded to 512x512 with content at top-left,
    and original size is encoded in filename as _WxH.

    Example filename:
        image_380x420.png  → original width=380, height=420

    Args:
        img_dir (str): Directory containing padded images
    """

    img_dir = os.path.abspath(img_dir)
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory does not exist: {img_dir}")

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not img_files:
        print(f"No images found in {img_dir}")
        return

    # Matches _WIDTHxHEIGHT before file extension
    pattern = re.compile(r"_(\d+)x(\d+)\.")

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_name}. Skipping.")
            continue

        match = pattern.search(img_name)
        if not match:
            print(f"Warning: No size info in {img_name}. Skipping.")
            continue

        orig_w = int(match.group(1))
        orig_h = int(match.group(2))

        # --- CROP instead of resize ---
        restored = img[:orig_h, :orig_w]

        # Remove _WxH from filename
        new_name = pattern.sub('.', img_name)
        new_path = os.path.join(img_dir, new_name)

        cv2.imwrite(new_path, restored)

        if new_path != img_path:
            os.remove(img_path)

    # print(f"Restored {len(img_files)} images by cropping padding.")

def restore_images_to_original_size(img_dir):
    """
    Resize images back to their original width and height based on filename suffix
    and rename them to remove the _WxH part.
    ⚡ THREADED: Processes tiles in parallel using ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor

    img_dir = os.path.abspath(img_dir)
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory does not exist: {img_dir}")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    img_files.sort()

    if not img_files:
        print(f"No images found in {img_dir}.")
        return

    pattern = re.compile(r"_(\d+)x(\d+)\.")  # matches _WIDTHxHEIGHT.

    def _restore_one(img_name):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            return
        match = pattern.search(img_name)
        if not match:
            print(f"Warning: Could not find original size in {img_name}. Skipping.")
            return
        orig_w, orig_h = int(match.group(1)), int(match.group(2))
        restored = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        new_name = pattern.sub('.', img_name)
        new_path = os.path.join(img_dir, new_name)
        cv2.imwrite(new_path, restored)
        if new_path != img_path:
            os.remove(img_path)

    with ThreadPoolExecutor() as executor:
        list(executor.map(_restore_one, img_files))


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
        Image.fromarray(reconstructed).save("reconstructed_from_folder_second.png")

    return reconstructed


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


def clahe_plus_hist_stretch_safe(img_bgr,
                                 clip_limit=2.0,
                                 tile_grid_size=(8, 8)):

    # 1️⃣ Convert to grayscale (required)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    # 2️⃣ CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    I_cla = clahe.apply(gray)

    # 3️⃣ Histogram stretching (Eq. 5)
    I_min = I_cla.min()
    I_max = I_cla.max()
    L = 256

    if I_max > I_min:
        I_hs = ((I_cla - I_min) / (I_max - I_min)) * (L - 1)
    else:
        I_hs = I_cla.copy()

    return I_hs.astype(np.uint8)

def enhance_contrast(image):
    # Drop alpha if RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert single-channel 3D → 2D
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    # Apply CLAHE
    if image.ndim == 2:  # grayscale
        image_eq = exposure.equalize_adapthist(image, clip_limit=0.1)
    else:  # RGB
        image_eq = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):
            image_eq[:, :, c] = exposure.equalize_adapthist(
                image[:, :, c], clip_limit=0.1
            )

    return img_as_ubyte(image_eq)

def resize_image_512(img):
    """
    Resize an input image to 512x512 pixels.

    Args:
        img: numpy.ndarray, shape [H, W] or [H, W, C]

    Returns:
        numpy.ndarray, resized image [512, 512] or [512, 512, C]
    """
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return resized


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

def pad_image_512(img):
    """
    Pads an image to 512x512 without resizing or stretching.
    Padding is added to:
      - bottom if height < 512
      - right if width < 512

    Args:
        img (np.ndarray): [H, W] or [H, W, C]

    Returns:
        np.ndarray: [512, 512] or [512, 512, C]
    """

    h, w = img.shape[:2]

    if h > 512 or w > 512:
        raise ValueError("Image is larger than 512x512. Cropping is not allowed.")

    pad_bottom = 512 - h
    pad_right = 512 - w

    # Padding format: (top, bottom), (left, right)
    if img.ndim == 2:  # grayscale
        padded = np.pad(
            img,
            ((0, pad_bottom), (0, pad_right)),
            mode="constant",
            constant_values=0
        )
    else:  # color
        padded = np.pad(
            img,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant",
            constant_values=0
        )

    return padded