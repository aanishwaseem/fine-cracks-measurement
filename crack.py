##############################################################################
                        # LIBRARY IMPORTS
##############################################################################

import cv2
import numpy as np
import math
import os
import re
import threading
import subprocess
import atexit
import tkinter as tk
from tkinter import messagebox, simpledialog
from tkinter import filedialog
from crack_analysis import CrackAnalyse
from crack_visualization import visualize_crack_overlay
from utils import (getBinaryImage, resize_image, extract_deepcracks,
                   overlay_binary_images, make_tiles_fixed_size,
                   run_inference, join_tiles_after_inference,
                   crack_segmentation_dir_string as _unet_dir_string)
from remove_gridlines import remove_gridlines, intersect_masks
from scale_image import scale_image
from mask_tuning import MaskTuningUI
from gen_ref_mask import get_reference_path, create_reference as gen_ref_create_reference
##############################################################################
                        # GLOBAL VARIABLES & CONSTANTS
##############################################################################
make_reference_image = False
NUM_SCALING_IMAGES = 13  
image_files = []
current_image_index = 0
original_image = None
reference_image = None
image = None
crack_image_on_bright = None
zoom_factor = 5  
is_zoomed = False
mouse_x, mouse_y = 0, 0
distance_mode = False
compute_cell_area_mode = False  
polygon_mode = False            
clicked_points = []             
polygon_points = []             
roi_x1, roi_y1 = 0, 0         
small_box_size_mm = 100  
scaling_factor_width = False
scaling_factor_height = False
distance_text = ""  
grid_spacing_mm = 100  
drag_start = None 
drag_end = None
highlighted_cell = None
disable_cache = False  # Set to True to disable caching
displacement_x = 0.0 
binary_mask = None

contrast_value = 0.08
threshold_value = 23
scale_image_factor = 2

grid_mask_thickness = 3
activate_grid_mask_recreation = False

#############################################################################
                        # HELPER FUNCTIONS
##############################################################################

# ✅ NEW: Cache management functions
def clear_all_caches():
    """Clear all caching systems."""
    global _grid_removal_cache, _crack_detection_cache
    _grid_removal_cache.clear()
    _crack_detection_cache.clear()
    
    # Clear DeepCrack cache
    from deepcrack_pipeline import _deepcrack_result_cache
    _deepcrack_result_cache.clear()
    
    print("[✓] All caches cleared")

def print_cache_status():
    """Print current cache status."""
    global _grid_removal_cache, _crack_detection_cache
    from deepcrack_pipeline import _deepcrack_result_cache
    
    print("\n" + "="*50)
    print("CACHE STATUS")
    print("="*50)
    print(f"Grid Removal Cache: {len(_grid_removal_cache)} items")
    print(f"Crack Detection Cache: {len(_crack_detection_cache)} items")
    print(f"DeepCrack Cache: {len(_deepcrack_result_cache)} items")
    print("="*50 + "\n")

# ✅ NEW: Parallel execution helper
def run_parallel_pipeline(image, image_gray):
    """
    Run DeepCrack and UNet in PARALLEL for maximum speed.
    
    ⚡ PARALLEL EXECUTION:
    - DeepCrack runs on Thread 1
    - UNet runs on Thread 2
    - Both run simultaneously
    
    Usage:
        deepcrack_img, unet_result, status = run_parallel_pipeline(image, image_gray)
        if status['deepcrack_status'] == 'complete' and status['unet_status'] == 'complete':
            # Both succeeded, proceed with merging
    """
    from deepcrack_pipeline import run_deepcrack_and_unet_parallel, deep_crack_dir_string as _dc_dir
    
    print("\n" + "="*60)
    print("🚀 STARTING PARALLEL PIPELINE EXECUTION")
    print("="*60)
    
    deepcrack_img, unet_result, status = run_deepcrack_and_unet_parallel(
        image,
        image_gray,
        tiles_dir_deepcrack=f"{_dc_dir}/input_tiles",
        original_h=image.shape[0],
        original_w=image.shape[1],
        tile_size=512,
        inc_contrast=True
    )
    
    print("\n" + "="*60)
    print("✓ PARALLEL EXECUTION RESULTS:")
    print(f"  DeepCrack: {status['deepcrack_status']}")
    print(f"  UNet: {status['unet_status']}")
    print(f"  Total Time: {status['total_time']:.2f}s")
    print("="*60 + "\n")
    
    return deepcrack_img, unet_result, status

def draw_transparent_rectangle(img, x, y, w, h, color=(0,55,0), alpha=0.2):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def perspective_correction(img, contour):
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        pts = sorted(np.squeeze(approx), key=lambda x: (x[1], x[0]))
        pts1 = np.float32([pts[0], pts[1], pts[3], pts[2]])
        width, height = 100, 100
        pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, matrix, (width, height))
    return None

def detect_scaling_factors_in_image_for_debug(img_gray, debug=False):
    debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        adaptive_thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    found_factors = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
            # Draw contour
            cv2.drawContours(debug_img, [contour], -1, (0,255,0), 2)

            # Draw bounding box
            cv2.rectangle(debug_img, (x,y), (x+w, y+h), (0,0,255), 2)

            # Label size
            label = f"{w}x{h}px"
            cv2.putText(debug_img, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            sfx = small_box_size_mm / w
            sfy = small_box_size_mm / h
            found_factors.append((sfx, sfy))

    if debug:
        cv2.imshow("Scaling Box Debug", debug_img)
        cv2.waitKey(0)
        cv2.destroyWindow("Scaling Box Debug")

    return found_factors


def detect_scaling_factors_in_image(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5,5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_factors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
            corrected_box = perspective_correction(img_gray, contour)
            if corrected_box is not None:
                sfx = small_box_size_mm / w
                sfy = small_box_size_mm / h
                found_factors.append((sfx, sfy))
    return found_factors


##############################################################################
                        # FILENAME SORTING FUNCTION
##############################################################################

def numeric_key(path):
    filename = os.path.basename(path)
    match = re.search(r'\((\d+)\)', filename)
    return int(match.group(1)) if match else 999999

##############################################################################
                        # COMPUTE AVERAGE SCALING FACTOR
##############################################################################

def compute_average_scaling_factor_for_folder(folder_path):
    """
    Compute average scaling factors from images in a folder.
    ⚡ THREADED: Loads and analyses images in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    all_sfx = []
    all_sfy = []
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    all_files = sorted(all_files, key=numeric_key)
    num_images = min(NUM_SCALING_IMAGES, len(all_files))
    files_to_process = all_files[:num_images]

    def _process_one(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return []
        factors = detect_scaling_factors_in_image(img)
        return factors if factors else []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(_process_one, files_to_process))

    for factors in results:
        for (sfx, sfy) in factors:
            all_sfx.append(sfx)
            all_sfy.append(sfy)

    if len(all_sfx) == 0:
        return None, None
    avg_sfx = np.mean(all_sfx)
    avg_sfy = np.mean(all_sfy)
    return avg_sfx, avg_sfy

# Function to prompt the user to set the grid displacement
def set_displacement():
    displacement_value = simpledialog.askfloat("Grid Displacement", "Enter displacement in mm:", initialvalue=0.0)
    if displacement_value is not None:
        print(f"Displacement set to: {displacement_value} mm")
        return displacement_value
    return 0.0
# Function to convert displacement in mm to pixels using scaling factor
def convert_displacement_to_pixels(displacement_mm, scaling_factor):
    return int(displacement_mm / scaling_factor)


##############################################################################
                        # LOAD IMAGE FUNCTIONS
##############################################################################
def select_reference_image(initial_dir):
    """
    [DEPRECATED] Manual reference image selection.
    Kept for backward compatibility. The pipeline now auto-detects/generates references.

    Args:
        initial_dir (str): Folder path to start dialog in

    Returns:
        str or None: Selected reference image path
    """
    ref_path = filedialog.askopenfilename(
        title="Select Reference Image",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")
        ]
    )

    if not ref_path:
        messagebox.showerror("Error", "Reference image selection is required.")
        return None

    return ref_path

def load_folder(folder_path):
    global image_files, current_image_index, scaling_factor_width, scaling_factor_height, reference_image
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    image_files = sorted(image_files, key=numeric_key)
    if not image_files:
        messagebox.showerror("Error", "No image files found in the specified folder.")
        return

    # --- Automatic reference image detection / generation ---
    ref_path = get_reference_path(folder_path)

    if os.path.isfile(ref_path):
        print(f"[INFO] Reference image found: {ref_path}")
    else:
        print(f"[INFO] No reference image found for this dataset.")
        print(f"[INFO] Launching reference image generator...")
        result_path = gen_ref_create_reference(dataset_folder=folder_path)
        if result_path is None:
            messagebox.showerror("Error", "Reference image generation was cancelled or failed.")
            return
        ref_path = result_path
        print(f"[INFO] Reference image created: {ref_path}")

    reference_image = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        messagebox.showerror("Error", f"Failed to load reference image: {ref_path}")
        return
    print(f"[INFO] Using reference image: {ref_path}")

    avg_sfx, avg_sfy = compute_average_scaling_factor_for_folder(folder_path)
    if avg_sfx is not None:
        scaling_factor_width = avg_sfx
        scaling_factor_height = avg_sfy
        if scale_image_factor > 1: 
            scaling_factor_width = scaling_factor_width / scale_image_factor
            scaling_factor_height = scaling_factor_height / scale_image_factor
        print(f"Global scaling factor (X): {scaling_factor_width:.2f} mm/px")
        print(f"Global scaling factor (Y): {scaling_factor_height:.2f} mm/px")
    else:
        scaling_factor_width = scaling_factor_height = None
        messagebox.showerror("Error", "No suitable scaling boxes found in the folder.")
    current_image_index = 0
    load_image()
_image_loading = False
def load_image():
    global image, original_image, crack_image_on_bright, highlighted_cell, polygon_points, polygon_mode, _image_loading

    if _image_loading:
        return  # block re-entrant calls while a load is already in progress
    _image_loading = True

    if current_image_index >= len(image_files):
        messagebox.showinfo("End", "All images in the folder have been processed.")
        return
    image_path = image_files[current_image_index]
    cv2.setWindowTitle("Crack Detection", "Crack Detection - " + os.path.basename(image_path))
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Image at path '{image_path}' could not be loaded.")
        return
    original_image = img
    highlighted_cell = None
    polygon_points = []
    polygon_mode = False
    reset_main_image(img)
    _image_loading = False

def reset_main_image(new_img):
    global image
    image = new_img
    process_image()

##############################################################a################
                        # IMAGE PROCESSING FUNCTIONS
##############################################################################

def remove_grid_lines_fallback(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 10)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
    vertical_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, horizontal_kernel)
    grid_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
    dilated_grid_mask = cv2.dilate(grid_mask, np.ones((3,3), np.uint8), iterations=2)
    img_no_grid = cv2.inpaint(img, dilated_grid_mask, 5, cv2.INPAINT_TELEA)
    return img_no_grid
def apply_lab_brightness(img, brightness_factor=0.5, a_contrast=1.0, b_contrast=1.0):
    """
    Apply LAB-based brightness/contrast adjustments to the image.
    """
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Adjust brightness in L channel
    l_channel = cv2.convertScaleAbs(l_channel, alpha=brightness_factor)

    # Adjust A and B channels for color contrast
    a_channel = cv2.convertScaleAbs(a_channel, alpha=a_contrast)
    b_channel = cv2.convertScaleAbs(b_channel, alpha=b_contrast)

    # Merge channels back
    lab_adjusted = cv2.merge((l_channel, a_channel, b_channel))

    # Convert back to BGR
    output_img = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    return output_img

def detect_cracks_no_area_good(image_no_grid, upscale_factor=2):
    global binary_mask

    # --- 1. Binary mask (your existing logic) ---
    testimg = getBinaryImage(image_no_grid)   # cracks expected white
    binary_mask = testimg.copy()

    gray = testimg.copy()

    # --- 2. Optional upscale (SUB-PIXEL) ---
    if upscale_factor > 1:
        gray_up = cv2.resize(
            gray, None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_NEAREST
        )
        img_up = cv2.resize(
            image_no_grid, None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC
        )
    else:
        gray_up = gray
        img_up = image_no_grid.copy()

    # --- 3. dx / dy gradients (KEY CHANGE) ---
    dx = cv2.Sobel(gray_up, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_up, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(dx, dy)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)

    # --- 4. Tight threshold on gradient ---
    _, binary_image = cv2.threshold(
        magnitude,
        35,      # 🔒 tighten here (30–45 range)
        255,
        cv2.THRESH_BINARY
    )

    # --- 5. Morphological cleanup ---
    kernel = np.ones((2, 2), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # --- 6. Find contours ---
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # --- 7. Draw contours ---
    crack_image_up = img_up.copy()
    cv2.drawContours(crack_image_up, contours, -1, (0, 0, 255), 1)
    cv2.drawContours(crack_image_up, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)

    # --- 8. Downscale back ---
    if upscale_factor > 1:
        crack_image = cv2.resize(
            crack_image_up,
            (image_no_grid.shape[1], image_no_grid.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        crack_image = crack_image_up

    # --- Debug view ---
    cv2.imshow("Gradient Binary (dx/dy)", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Generated crack image with upscaling + dx/dy.")
    return crack_image

def get_binary_image_of_cracks(gen_binary_mask, threshold = 23, alpha = 0.68, beta=12):
    gray_image = gen_binary_mask.copy()
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    _, binary_image = cv2.threshold(brightened_image, threshold, 255, cv2.THRESH_BINARY_INV)
    binary_image = 255 - binary_image  # invert black<->white
    return binary_image

def draw_contours_on_img(orginal_image, cracks_binary_image):
    contours, _ = cv2.findContours(cracks_binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Make output a 3-channel BGR image for drawing
    crack_image = orginal_image.copy()
    # Draw contours in red
    cv2.drawContours(crack_image, contours, -1, (0, 0, 255),1, lineType=cv2.LINE_AA)   # anti-aliased
    return crack_image

def detect_cracks_no_area(image_no_grid, deepcrack_img, mask):
    global binary_mask

    testimg = getBinaryImage(image_no_grid,deepcrack_img, mask)
    if testimg is None:
        print("[ERROR] Falling back to manual crack extraction.")
        return detect_cracks_no_area_fallback(image_no_grid)
    binary_mask = testimg
    cracks_binary = get_binary_image_of_cracks(binary_mask, threshold_value)
    cracks_binary = intersect_masks(reference_image, cracks_binary)
    binary_mask = cracks_binary
    crack_image = draw_contours_on_img(image_no_grid, cracks_binary)
    return crack_image

# def detect_cracks_no_area(image_no_grid, original_img, deepcrack_img, mask):
#     global binary_mask

#     # testimg = getBinaryImage(image_no_grid,deepcrack_img, mask)
#     # if testimg is None:
#     #     print("[ERROR] Falling back to manual crack extraction.")
#     #     return detect_cracks_no_area_fallback(image_no_grid)
#     # binary_mask = testimg
#     # gray_image = testimg.copy()
#     # brightened_image = cv2.convertScaleAbs(gray_image, alpha=0.68, beta=12)
#     # _, binary_image = cv2.threshold(brightened_image, 20, 255, cv2.THRESH_BINARY_INV)
#     # binary_image = 255 - binary_image  # invert black<->white
#     # cv2.imwrite("reference_D_91.png", binary_image)

#     binary_image = cv2.imread("reference_D_91.png", cv2.IMREAD_GRAYSCALE)
#     # binary_image = intersect_masks(reference_image, binary_image)
#     # --- Debug view ---
#     cv2.imshow("Gradient Binary (dx/dy)", binary_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     # Make output a 3-channel BGR image for drawing
#     crack_image = image_no_grid.copy()
#     # Draw contours in red
#     cv2.drawContours(crack_image, contours, -1, (0, 0, 255),1, lineType=cv2.LINE_AA)   # anti-aliased
#     return crack_image


def detect_cracks_no_area_fallback(image_no_grid, show_binary=True):
    # Optional LAB adjustment
    lab_adjusted = apply_lab_brightness(image_no_grid, brightness_factor=0.5, a_contrast=1.0, b_contrast=1.0)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_no_grid, cv2.COLOR_BGR2GRAY)

    # Brighten the image
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=0.68, beta=12)

    # Threshold to get binary image (inverted)
    _, binary_image = cv2.threshold(brightened_image, 55, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of original image
    crack_image = image_no_grid.copy()
    cv2.drawContours(crack_image, contours, -1, (0,0,255), 2)

    return crack_image, binary_image

def loading_window():
    window_name = "Crack Detection"

    # Create a black image
    loading_img = np.zeros((400, 600, 3), dtype=np.uint8)

    # Put text "LOADING" in the center
    cv2.putText(
        loading_img,
        "IMG LOADING...",
        (50, 200),                  # position
        cv2.FONT_HERSHEY_SIMPLEX,    # font
        2,                           # font scale
        (255, 255, 255),             # color (white)
        3,                           # thickness
        cv2.LINE_AA                  # line type
    )

    # Show in a named window (create if doesn't exist)
    cv2.imshow(window_name, loading_img)
    cv2.waitKey(1)  # Small delay to refresh window

##############################################################################
                        # OPTIMIZATION CACHING
##############################################################################
# Cache processed results to avoid recomputation
_grid_removal_cache = {}
_crack_detection_cache = {}

def _get_cache_key(img_path, params):
    """Generate cache key based on image path and parameters."""
    return (img_path, tuple(sorted(params.items())))

def clear_cache():
    """Clear all cached results."""
    global _grid_removal_cache, _crack_detection_cache
    _grid_removal_cache.clear()
    _crack_detection_cache.clear()
    print("[INFO] Cache cleared")

##############################################################################
                        # OPTIMIZED IMAGE PROCESSING
##############################################################################

def process_image(debug=False):
    global image, crack_image_on_bright, scaling_factor_width, scaling_factor_height
    global _grid_removal_cache, _crack_detection_cache
    
    # ✅ OPTIMIZATION 1: Skip scaling factor detection after first pass
    if current_image_index < NUM_SCALING_IMAGES and current_image_index == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            if 0.9 < aspect_ratio < 1.1 and 1000 < area < 20000:
                corrected_box = perspective_correction(gray, contour)
    
    # ✅ OPTIMIZATION 2: Cache text rendering (only render once per session)
    if scaling_factor_width and scaling_factor_height:
        cv2.putText(image, f'Scaling Factor X: {scaling_factor_width:.2f} mm/px', (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(image, f'Scaling Factor Y: {scaling_factor_height:.2f} mm/px', (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    if scale_image_factor > 1:
        image = scale_image(image, 2)
    
    # ✅ OPTIMIZATION 3: Cache grid removal results
    current_img_path = image_files[current_image_index]
    cache_params = {
        'grid_mask_thickness': grid_mask_thickness,
        'make_reference_image': make_reference_image
    }
    cache_key = _get_cache_key(current_img_path, cache_params)
    
    # ✅ OPTIMIZATION 4: Cache crack detection results (key computed upfront)
    crack_cache_key = _get_cache_key(current_img_path, {'crack_detection': True})

    if cache_key in _grid_removal_cache and crack_cache_key in _crack_detection_cache:
        # ── Both results cached ── fast path ──────────────────────────────────
        print("[INFO] Using cached grid removal results")
        output_img, deepcrack_img_with_grids, mask = _grid_removal_cache[cache_key]
        print("[INFO] Using cached crack detection results")
        crack_image_on_bright = _crack_detection_cache[crack_cache_key]

    else:
        # ── Cache miss: run DeepCrack + UNet truly in parallel ────────────────
        # UNet and DeepCrack use completely separate tile directories:
        #   UNet   →  models/crack_segmentation/tiles2_s
        #   DeepCrack → models/DeepCrack/codes/input_tiles
        # There is zero shared state, so both can run simultaneously.

        _TILE_SIZE = 512

        # Check whether grid removal result is already cached
        if cache_key in _grid_removal_cache:
            print("[INFO] Using cached grid removal results")
            output_img, deepcrack_img_with_grids, mask = _grid_removal_cache[cache_key]
            _grid_cache_hit = True
        else:
            _grid_cache_hit = False

        # ── STEP 1: Prepare UNet tiles and start UNet inference NOW ───────────
        # This runs concurrently while the DeepCrack pipeline (test.py) runs below.
        print("[⚡] Preparing UNet tiles and launching UNet inference thread...")
        make_tiles_fixed_size(image, tile_size=_TILE_SIZE)

        _unet_data = {"result": None, "status": "pending"}

        def _run_unet_bg(_img_ref=image):
            try:
                res = run_inference("tiles2_s", output_dir="experiment")
                if res:
                    reconstructed = join_tiles_after_inference(
                        _unet_dir_string,
                        "experiment",
                        tile_size=_TILE_SIZE,
                        original_h=_img_ref.shape[0],
                        original_w=_img_ref.shape[1],
                        save=False
                    )
                    _unet_data["result"] = reconstructed
                    _unet_data["status"] = "complete"
                    print("[✓] UNet inference complete")
                else:
                    _unet_data["status"] = "error"
                    print("[✗] UNet inference failed")
            except Exception as _e:
                _unet_data["status"] = "error"
                print(f"[✗] UNet inference error: {_e}")

        _t_unet = threading.Thread(target=_run_unet_bg, name="UNet-Thread", daemon=False)
        _t_unet.start()
        print("[→] UNet thread started — DeepCrack pipeline starting now (both run in parallel)...")

        # ── STEP 2: Run DeepCrack pipeline (blocks ~30s while UNet runs above) ─
        if not _grid_cache_hit:
            result = remove_gridlines(
                image,
                os.path.dirname(current_img_path),
                grid_mask_thickness,
                activate_grid_mask_recreation=False,
                make_reference_image=make_reference_image
            )
            if result is None:
                print("[ERROR] remove_gridlines returned None")
                _t_unet.join()
                return
            output_img, deepcrack_img_with_grids, mask = result
            _grid_removal_cache[cache_key] = (output_img, deepcrack_img_with_grids, mask)
            print("[✓] DeepCrack pipeline complete")

        # ── STEP 3: Start inpainting thread (needs deepcrack_img_with_grids) ──
        # By the time DeepCrack finishes, UNet may already be done or nearly done.
        _inpaint_data = {"result": None}

        def _run_inpaint_bg(_dc_img=deepcrack_img_with_grids, _msk=mask):
            try:
                _inpaint_data["result"] = extract_deepcracks(_dc_img, _msk)
                print("[✓] DeepCrack inpainting complete")
            except Exception as _e:
                print(f"[✗] DeepCrack inpainting error: {_e}")

        _t_inpaint = threading.Thread(target=_run_inpaint_bg, name="Inpaint-Thread", daemon=False)
        _t_inpaint.start()

        # ── STEP 4: Join both threads ─────────────────────────────────────────
        print("[⏳] Waiting for UNet and inpainting to complete...")
        _t_unet.join()
        _t_inpaint.join()

        # ── STEP 5: Merge results ─────────────────────────────────────────────
        if _unet_data["status"] == "complete" and _inpaint_data["result"] is not None:
            merged = overlay_binary_images(_unet_data["result"], _inpaint_data["result"])
            global binary_mask
            binary_mask = merged
            _cracks_binary = get_binary_image_of_cracks(merged, threshold_value)
            _cracks_binary = intersect_masks(reference_image, _cracks_binary)
            binary_mask = _cracks_binary
            crack_image_on_bright = draw_contours_on_img(image, _cracks_binary)
        else:
            print("[WARN] Parallel path incomplete — falling back to sequential detect_cracks_no_area()")
            crack_image_on_bright = detect_cracks_no_area(image, deepcrack_img_with_grids, mask)

        _crack_detection_cache[crack_cache_key] = crack_image_on_bright
    
    if crack_image_on_bright is not None and debug:
        cv2.imshow("Cracks Only", crack_image_on_bright)
        cv2.waitKey(0)
        cv2.destroyWindow("Cracks Only")

##############################################################################
                    # CALCULATION & GRID FUNCTIONS
##############################################################################

def calculate_distance(point1, point2, scaling_factor_x, scaling_factor_y):
    dx = (point2[0] - point1[0]) * scaling_factor_x if scaling_factor_x else 0
    dy = (point2[1] - point1[1]) * scaling_factor_y if scaling_factor_y else 0
    return math.sqrt(dx**2 + dy**2)

def compute_polygon_area(points):
    if len(points) < 3:
        return 0
    pts = np.array(points, dtype=np.int32)
    area_px = cv2.contourArea(pts)
    if scaling_factor_width and scaling_factor_height:
        area_mm2 = area_px * (scaling_factor_width * scaling_factor_height)
        return area_mm2
    return area_px

# Modified function to draw grid with horizontal displacement
def draw_grid_on_image(img, grid_spacing_x=50, grid_spacing_y=50, displacement_x=0):
    img_with_grid = img.copy()
    h, w = img_with_grid.shape[:2]
    displacement_x = int(displacement_x)

    # Horizontal lines are full width; displacement is for vertical grid only.
    for y in range(0, h, grid_spacing_y):
        cv2.line(img_with_grid, (0, y), (w - 1, y), (0, 0, 0), 2)

    for x in range(0, w, grid_spacing_x):
        x_shifted = x + displacement_x
        if 0 <= x_shifted < w:
            cv2.line(img_with_grid, (x_shifted, 0), (x_shifted, h - 1), (0, 0, 0), 2)

    box_number = 1
    rows = h // grid_spacing_y
    cols = w // grid_spacing_x
    for row in range(rows):
        for col in range(cols):
            cell_x = col * grid_spacing_x + displacement_x
            cell_y = row * grid_spacing_y
            center_x = cell_x + grid_spacing_x // 2
            center_y = cell_y + grid_spacing_y // 2
            if 0 <= center_x < w and 0 <= center_y < h:
                cv2.putText(
                    img_with_grid,
                    str(box_number),
                    (center_x - 10, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
            box_number += 1

    return img_with_grid

# Global variables to store the crack areas for the current and previous image
previous_image_areas = {}

# Function to compute the cracked area in each cell and track changes
# Define the threshold for significant change in pixels
significant_change_threshold = 1  # Change in pixels that is considered significant

def compute_cell_area(x, y):
    global crack_image_on_bright, grid_spacing_mm, scaling_factor_width, scaling_factor_height, highlighted_cell
    src = crack_image_on_bright if crack_image_on_bright is not None else image
    
    # Cell size in pixels based on scaling factors
    if scaling_factor_width and scaling_factor_height:
        cell_width_px = int(grid_spacing_mm / scaling_factor_width)
        cell_height_px = int(grid_spacing_mm / scaling_factor_height)
    else:
        cell_width_px = cell_height_px = 50

    col = x // cell_width_px
    row = y // cell_width_px
    x1 = col * cell_width_px
    y1 = row * cell_height_px
    x2 = x1 + cell_width_px
    y2 = y1 + cell_height_px
    cell = src[y1:y2, x1:x2].copy()
    
    # Convert the selected cell to grayscale for processing
    gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, binary_cell = cv2.threshold(gray_cell, 85, 255, cv2.THRESH_BINARY_INV)
    # Get the total and white (crack) pixels
    total_pixels = binary_cell.size
    white_pixels = cv2.countNonZero(binary_cell)
    black_pixels = total_pixels - white_pixels
    
    # Calculate the cracked area in pixels and millimeters (if scaling factors are available)
    if scaling_factor_width and scaling_factor_height:
        px_to_mm2 = scaling_factor_width * scaling_factor_height
        crack_area_mm2 = white_pixels * px_to_mm2
        non_crack_area_mm2 = black_pixels * px_to_mm2
        crack_percent = (crack_area_mm2 / (total_pixels * px_to_mm2)) * 100 if total_pixels > 0 else 0
        non_crack_percent = 100 - crack_percent
        print(f"Cell ({row}, {col}):")
        print(f"  Crack Area (Black): {white_pixels} px, {crack_area_mm2:.2f} mm² ({crack_percent:.1f}%)")
        print(f"  Non-Crack Area (White): {black_pixels} px, {non_crack_area_mm2:.2f} mm² ({non_crack_percent:.1f}%)")
    else:
        crack_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        non_crack_percent = 100 - crack_percent
        print(f"Cell ({row}, {col}):")
        print(f"  Crack Area (Black): {white_pixels} px ({crack_percent:.1f}%), Non-Crack Area (White): {black_pixels} px ({non_crack_percent:.1f}%)")
    
    # Highlight the selected cell on the image
    highlighted_cell = (x1, y1, x2, y2)
    
    # Track the crack area for this cell in the current image
    area_data = {
        "crack_area_px": white_pixels,
        "crack_area_mm2": crack_area_mm2 if scaling_factor_width and scaling_factor_height else None
    }
    
    # Get the key for this cell
    cell_key = (row, col)

    # If it's not the first image, compare with previous areas
    if previous_image_areas.get(cell_key) is not None:
        prev_area = previous_image_areas[cell_key]["crack_area_px"]
        area_change = white_pixels - prev_area

        # If the change is below the threshold, ignore it and print "No significant change"
        if abs(area_change) < significant_change_threshold:
            print(f"No significant change in cracked area for Cell ({row}, {col}).")
        else:
            print(f"Change in cracked area for Cell ({row}, {col}): {area_change} pixels.")
    else:
        print(f"Change in crack area in this Cell ({row}, {col}), same as cracked area.")
    
    # Store current area for next image comparison
    previous_image_areas[cell_key] = area_data
    highlighted_cell = (x1, y1, x2, y2)  # Highlight the clicked cell



# Add the `c` button functionality to activate the mode
def on_key_press(event):
    global compute_cell_area_mode
    if event.char == 'c':
        compute_cell_area_mode = not compute_cell_area_mode
        if compute_cell_area_mode:
            print("Cell Area Mode ON")
        else:
            print("Cell Area Mode OFF")

##############################################################################
                        # MOUSE & MAIN LOOP FUNCTIONS
##############################################################################

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, clicked_points, is_zoomed, drag_start, drag_end
    global distance_mode, distance_text, compute_cell_area_mode, polygon_mode, polygon_points, highlighted_cell

    if polygon_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if is_zoomed:
                h_img, w_img = image.shape[:2]
                crop_w = int(w_img / zoom_factor)
                crop_h = int(h_img / zoom_factor)
                adjusted_x = roi_x1 + int(x * crop_w / w_img)
                adjusted_y = roi_y1 + int(y * crop_h / h_img)
            else:
                adjusted_x, adjusted_y = x, y
            polygon_points.append((adjusted_x, adjusted_y))
            cv2.circle(image, (adjusted_x, adjusted_y), 4, (255,0,0), -1)
            if len(polygon_points) > 1:
                cv2.line(image, polygon_points[-2], polygon_points[-1], (255,0,0), 2)
            cv2.imshow("Crack Detection", image)
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(polygon_points) >= 3:
                cv2.line(image, polygon_points[-1], polygon_points[0], (255,0,0), 2)
                area = compute_polygon_area(polygon_points)
                text = f"Poly Area: {area:.2f} mm^2" if (scaling_factor_width and scaling_factor_height) else f"Poly Area: {area:.2f} px"
                print(text)
                cv2.putText(image, text, (polygon_points[0][0], polygon_points[0][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

                    # --- Create mask from polygon ---
                mask = np.zeros(binary_mask.shape[:2], dtype=np.uint8)
                pts = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

                # --- Get bounding rectangle around polygon ---
                x, y, w, h = cv2.boundingRect(pts)
                mask_cropped = mask[y:y+h, x:x+w]

                # --- Crop the **binary crack image** using the mask ---
                # Extract only the crack pixels within the polygon region
                selected_poly = cv2.bitwise_and(binary_mask[y:y+h, x:x+w],
                                                binary_mask[y:y+h, x:x+w],
                                                mask=mask_cropped)

                # --- Analyze cracks in polygon ---
                # Pass grayscale binary mask directly (values: 0 for background, 255 for cracks)
                analyser = CrackAnalyse(predict_image_array=selected_poly, scaling_factor_x=scaling_factor_width,
                                       scaling_factor_y=scaling_factor_height)
                max_width = analyser.get_crack_max_width()
                mean_width = analyser.get_crack_mean_width()
                max_width_mm = analyser.get_crack_max_width_mm()
                mean_width_mm = analyser.get_crack_mean_width_mm()
                print(f"Max Crack Width: {max_width:.2f} px ({max_width_mm:.2f} mm)")
                print(f"Mean Crack Width: {mean_width:.2f} px ({mean_width_mm:.2f} mm)")
                # --- Display features on image ---
                feat_text2 = f"Max Width: {max_width_mm:.2f} mm"
                feat_text3 = f"Mean Width: {mean_width_mm:.2f} mm"

                cv2.putText(image, feat_text3, (polygon_points[0][0], polygon_points[0][1]-50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
                cv2.putText(image, feat_text2, (polygon_points[0][0], polygon_points[0][1]-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                # cv2.imshow("Selected Polygon", selected_poly)
                # cv2.waitKey(0)
                # cv2.destroyWindow("Selected Polygon")

                cv2.imshow("Crack Detection", image)
                
                # --- Visualize crack analysis ---
                visualize_crack_overlay(analyser, alpha_mask=0.4, alpha_heatmap=0.6)

            polygon_mode = False
            polygon_points = []
            return

    if compute_cell_area_mode and event == cv2.EVENT_LBUTTONDOWN:
        compute_cell_area(x, y)
        return

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
        if drag_start:
            drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if is_zoomed:
            h_img, w_img = image.shape[:2]
            crop_w = int(w_img / zoom_factor)
            crop_h = int(h_img / zoom_factor)
            adjusted_x = roi_x1 + int(x * crop_w / w_img)
            adjusted_y = roi_y1 + int(y * crop_h / h_img)
        else:
            adjusted_x, adjusted_y = x, y

        if distance_mode:
            if len(clicked_points) >= 2:
                clicked_points.clear()
                idx_img = cv2.imread(image_files[current_image_index])
                # if idx_img is not None:
                #     reset_main_image(idx_img)
            clicked_points.append((adjusted_x, adjusted_y))
            if len(clicked_points) == 1:
                cv2.circle(image, (adjusted_x, adjusted_y), 3, (0,0,255), -1)
            elif len(clicked_points) == 2:
                cv2.circle(image, (adjusted_x, adjusted_y), 3, (0,0,255), -1)
                cv2.line(image, clicked_points[0], (adjusted_x, adjusted_y), (0,255,0),2)
                dist = calculate_distance(clicked_points[0], (adjusted_x, adjusted_y),
                                          scaling_factor_width, scaling_factor_height)
                distance_text = f"Distance: {dist:.2f} mm"
                    # 🔴 DRAW TEXT ON IMAGE
                text_x = min(clicked_points[0][0], adjusted_x)
                text_y = min(clicked_points[0][1], adjusted_y) - 10

                cv2.putText(
                    image,
                    distance_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            cv2.imshow("Crack Detection", image)
            return

        if drag_start is None:
            drag_start = (x, y)
            drag_end = None

    elif event == cv2.EVENT_LBUTTONUP:
        if drag_start:
            x1, y1 = drag_start
            x2, y2 = x, y
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            if scaling_factor_width and scaling_factor_height:
                area_pixels = width * height
                area_mm2 = area_pixels * scaling_factor_width * scaling_factor_height
                print(f"Selected Area: {area_mm2:.2f} mm²")
            else:
                print("Selected Area: (No scaling factor)")
            drag_start = None

# Main loop function to handle keypresses directly within the loop:
def main_loop():
    global image, crack_image_on_bright, current_image_index, is_zoomed, distance_mode
    global roi_x1, roi_y1, distance_text, compute_cell_area_mode, highlighted_cell, grid_spacing_mm
    global polygon_mode, polygon_points, displacement_x  # Added displacement_x to globals
    global grid_mask_thickness, activate_grid_mask_recreation
    global contrast_value, threshold_value
    if image is None:
        messagebox.showerror("Error", "No image loaded. Please check the folder path.")
        return

    cv2.namedWindow("Crack Detection",cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("Crack Detection", "Crack Detection - " + os.path.basename(image_files[current_image_index]))
    window_h, window_w = image.shape[:2]
    window_h = int(window_h/scale_image_factor)
    window_w = int(window_w/scale_image_factor)
    cv2.resizeWindow("Crack Detection", window_w, window_h)

    cv2.setMouseCallback("Crack Detection", on_mouse)

    while True:
        h, w = image.shape[:2]
        if crack_image_on_bright is not None:
            base_display = cv2.addWeighted(image, 0.7, crack_image_on_bright, 0.3, 0)
        else:
            base_display = image.copy()

        if is_zoomed:
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            zoom_x1 = max(0, min(mouse_x - crop_w // 2, w - crop_w))
            zoom_y1 = max(0, min(mouse_y - crop_h // 2, h - crop_h))
            roi_x1, roi_y1 = zoom_x1, zoom_y1
            if scaling_factor_width and scaling_factor_height:
                grid_spacing_x = int(grid_spacing_mm / scaling_factor_width)
                grid_spacing_y = int(grid_spacing_mm / scaling_factor_height)
            else:
                grid_spacing_x = grid_spacing_y = 80

            full_grid = draw_grid_on_image(
                image,
                grid_spacing_x=grid_spacing_x,
                grid_spacing_y=grid_spacing_y,
                displacement_x=displacement_x,
            )
            grid_crop = full_grid[zoom_y1:zoom_y1+crop_h, zoom_x1:zoom_x1+crop_w]
            grid_resized = cv2.resize(grid_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            crop = base_display[zoom_y1:zoom_y1+crop_h, zoom_x1:zoom_x1+crop_w]
            display_image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
            display_image = cv2.addWeighted(display_image, 0.9, grid_resized, 0.1, 0)
            if highlighted_cell is not None:
                hx1, hy1, hx2, hy2 = highlighted_cell
                new_hx1 = int((hx1 - roi_x1) * w / crop_w)
                new_hy1 = int((hy1 - roi_y1) * h / crop_h)
                new_hx2 = int((hx2 - roi_x1) * w / crop_w)
                new_hy2 = int((hy2 - roi_y1) * h / crop_h)
                cv2.rectangle(display_image, (new_hx1, new_hy1), (new_hx2, new_hy2), (0,0,255), 3)
        else:
            display_image = base_display.copy()
            if highlighted_cell is not None:
                cv2.rectangle(display_image,
                              (highlighted_cell[0], highlighted_cell[1]),
                              (highlighted_cell[2], highlighted_cell[3]),
                              (0,0,255), 3)
            if scaling_factor_width and scaling_factor_height:
                grid_spacing_x = int(grid_spacing_mm / scaling_factor_width)
                grid_spacing_y = int(grid_spacing_mm / scaling_factor_height)
            else:
                grid_spacing_x = grid_spacing_y = 80
            display_image = draw_grid_on_image(
                display_image,
                grid_spacing_x=grid_spacing_x,
                grid_spacing_y=grid_spacing_y,
                displacement_x=displacement_x,
            )

        cv2.imshow("Crack Detection", display_image)
        # win_w = cv2.getWindowImageRect("Crack Detection")[2]
        # win_h = cv2.getWindowImageRect("Crack Detection")[3]

        # display_resized = cv2.resize(
        #     display_image,
        #     (win_w, win_h),
        #     interpolation=cv2.INTER_LINEAR
        # )

        # cv2.imshow("Crack Detection", display_resized)
        key = cv2.waitKey(1) & 0xFF
       
        if key == ord('n') and not _image_loading:
            current_image_index += 1
            if current_image_index < len(image_files):
                load_image()
            else:
                print("End of images in folder.")
        elif key == ord('b') and not _image_loading:
            current_image_index = max(0, current_image_index - 1)
            load_image()
        elif key == ord('m'):
            is_zoomed = not is_zoomed
        elif key == ord('d'):
            distance_mode = not distance_mode
        elif key == ord('c'):
            compute_cell_area_mode = not compute_cell_area_mode
            if not compute_cell_area_mode:
                highlighted_cell = None
            print("Cell Area Mode ON" if compute_cell_area_mode else "Cell Area Mode OFF")
        elif key == ord('p'):
            polygon_mode = not polygon_mode
            if polygon_mode:
                polygon_points = []
                print("Polygon mode ON: Left-click to add vertices, right-click to finish.")
            else:
                print("Polygon mode OFF.")
        elif key == ord('g'):
            new_spacing = simpledialog.askfloat("Grid Spacing", "Enter grid spacing in mm:", initialvalue=grid_spacing_mm)
            if new_spacing is not None and new_spacing > 0:
                grid_spacing_mm = new_spacing
        elif key == ord('h'):
            # This is where the displacement is asked
            displacement_x = simpledialog.askfloat("Grid Displacement", "Enter horizontal displacement (in pixels):", initialvalue=0)
            print(f"Grid displacement set to: {displacement_x} pixels")
        elif key == ord('s'):
            ui = MaskTuningUI(reference_image, binary_mask)
            apply_changes = ui.run()
            final_mask = ui.get_final_mask()
            threshold_value = ui.get_confirmed_threshold()
            crack_image_on_bright = draw_contours_on_img(image, final_mask)
            cv2.imshow("Crack Detection", crack_image_on_bright)

            print(f"Apply changes: {apply_changes}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

##############################################################################
                                # MAIN BLOCK
##############################################################################

def select_folder():
    folder_selected = filedialog.askdirectory(title="Select Folder")
    if folder_selected:
        print(f"Selected folder: {folder_selected}")
        return folder_selected
    else:
        print("No folder selected")
        return None

folder_path = select_folder() 
if folder_path:
    load_folder(folder_path)
    main_loop()
else:
    messagebox.showerror("Error", "No folder selected.")
    
# settings = open_settings_dialog(contrast_value, threshold_value)
