import cv2
import numpy as np
from deepcrack_pipeline import make_tiles_fixed_size, start_deepcrack_pipeline, empty_folder, join_tiles_after_inference
import os
import re
import requests
import base64
import io
import json
import numpy as np
import sys
from PIL import Image
import hashlib

crack_segmentation_dir_string = "models/crack_segmentation"
deep_crack_dir_string = "models/DeepCrack/codes"

line_segment_dir_string = "models/Unified-Line-Segment-Detection"
line_segment_dir = os.path.join(os.getcwd(), line_segment_dir_string)
import threading
import subprocess

endpoint = 'http://127.0.0.1:8000/'

# ✅ OPTIMIZATION: Cache for remove_gridlines results
_remove_gridlines_cache = {}
_intersect_masks_cache = {}

def extract_and_hough_filter_tol(image_path, tol_horizontal=5, tol_vertical=1, thickness=3):
    """
    Detects orange lines, tolerating minor deviations in horizontal/vertical angles.

    tol_horizontal: degrees tolerance from 0° (horizontal)
    tol_vertical: degrees tolerance from 90° (vertical)
    """
    # Load image
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- PART 1: MASK EXTRACTION ---
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # --- PART 2: HOUGH TRANSFORM ---
    height, width = img.shape[:2]
    min_line_length = int(0.15 * min(width, height))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=20)

    # --- PART 3: FILTER BY ANGLE ---
    hough_result = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))  # -180° to 180°

            # Normalize angle to 0–180
            angle = abs(angle)
            if angle > 180:
                angle = 360 - angle

            # Keep lines within tolerance
            if (angle <= tol_horizontal) or (abs(angle - 180) <= tol_horizontal):  # horizontal
                cv2.line(hough_result, (x1, y1), (x2, y2), (255,255,255), thickness)
            elif (abs(angle - 90) <= tol_vertical):  # vertical
                cv2.line(hough_result, (x1, y1), (x2, y2), (255,255,255), thickness)

    return mask, hough_result

def pad_to_size(mask, h, w):
    padded = np.zeros((h, w), dtype=mask.dtype)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded

def merge_masks(mask1, mask2, mode="or", tighten=False, auto_fix=True):
    """
    Merge two binary masks safely, even if size differs slightly.
    """

    if mask1 is None or mask2 is None:
        raise ValueError("One or both masks are None")

    # --- Convert to single-channel ---
    if mask1.ndim == 3:
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)

    if mask2.ndim == 3:
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

    h1, w1 = mask1.shape
    h2, w2 = mask2.shape

    # --- Auto-fix minor size mismatch ---
    if (h1, w1) != (h2, w2):
        if not auto_fix:
            raise ValueError(f"Mask size mismatch: {(h1,w1)} vs {(h2,w2)}")

        h = max(h1, h2)
        w = max(w1, w2)

        print(f"[WARN] Auto-cropping masks: {(h1,w1)} & {(h2,w2)} → {(h,w)}")


        mask1 = pad_to_size(mask1, h, w)
        mask2 = pad_to_size(mask2, h, w)

    # --- Force binary ---
    _, m1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    # --- Merge ---
    if mode == "or":
        merged = cv2.bitwise_or(m1, m2)
    elif mode == "and":
        merged = cv2.bitwise_and(m1, m2)
    elif mode == "xor":
        merged = cv2.bitwise_xor(m1, m2)
    else:
        raise ValueError("mode must be: 'or', 'and', or 'xor'")

    # --- Optional tightening ---
    if tighten:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel)
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)

    return merged

def run_line_segmentor(output_dir="output/pinhole"):
    experiment_dir = os.path.join(line_segment_dir, output_dir)
    empty_folder(experiment_dir)
    command = [
        sys.executable,
        "test.py",
        "--config_file", f"pinhole.yaml",
        "--dataset_name", "pinhole",
        "--save_image"
    ]
    try:
        result = subprocess.run(
            command,
            cwd=line_segment_dir,
            check=True
        )
    except subprocess.CalledProcessError:
        print("- Line segmentor failed.")
        return False

    if result.returncode == 0:
        return True
    else:
        return False

def get_sanitized_folder_name(full_path):
    # Get the last folder name
    folder_name = os.path.basename(full_path)
    # Sanitize: keep only alphanumeric, dash, underscore
    sanitized = re.sub(r'[^A-Za-z0-9_\-]', '_', folder_name)
    return sanitized
def remove_gridlines(img, parent_folder, mask_thickness=2, activate_grid_mask_recreation=True, make_reference_image=False, tile_size = None):
    if tile_size == None:
        tile_size = img.shape[0]

    parent_folder = get_sanitized_folder_name(parent_folder)
    print(f"[INFO] Running grid lines removal...{parent_folder}")

    DEFAULT_MASK_PATH = f"default_grids_mask/{parent_folder}.jpg"
    THICKNESS = mask_thickness

    mask = None

    try:
            print(f"[INFO] Proceeding with grid lines removal with mask thickness: {THICKNESS}...")
            #we need to drive out more details so tile size should be reduced
            if (not make_reference_image):
                tile_size = 300
            deepCrackImg = start_deepcrack_pipeline(img,
                                                    f"{deep_crack_dir_string}/input_tiles",
                                                    original_h=img.shape[0],
                                                    original_w=img.shape[1], tile_size=tile_size, inc_contrast=True)
            mask = create_mask_pipeline(deepCrackImg, THICKNESS, activate_grid_mask_recreation)


    except Exception as e:
        print(f"[ERROR] Grid lines removal failed: {e}")
        return None

    if mask is None:
        print("[ERROR] Mask creation failed. Exiting.")
        return None

    # --- Step 3: Inpainting ---
    try:
        output = None
        # output = inpaint_with_mask(
        #     image_np=deepCrackImg,
        #     mask_np=mask,
        #     url=f"{endpoint}/api/v1/inpaint",
        #     prompt="",
        #     cv2_radius=4
        #)
    except Exception as e:
        print(f"[ERROR] NGROK server error: {e}")
        return None

    print("[INFO] Grid lines removal successful.")
    return output, deepCrackImg,mask

def create_mask_pipeline(deepcrack_crack_image, thickness, run_line_segmentator=False):
    if (run_line_segmentator):
        save_dir=f"{line_segment_dir_string}/dataset/pinhole"
        move_img_to_input_folder(deepcrack_crack_image, save_dir)
        try:
            run_line_segmentor()
        except Exception as e:
            print(f"[ERROR] Line segementor error: {e}")
            return None
    
        _,img_from_line_segmentor = extract_and_hough_filter_tol(f"{line_segment_dir_string}/output/pinhole/1.png",3,3,thickness)
    hough_mask = make_mask_from_deepcrack_img(deepcrack_crack_image, thickness)
    if (run_line_segmentator):
        output = merge_masks(img_from_line_segmentor, hough_mask)
    else:
        output = hough_mask
    return output
def move_img_to_input_folder(img, save_dir):
    empty_folder(save_dir)

    if not isinstance(img, np.ndarray):
        raise TypeError("Expected numpy image")

    cv2.imwrite(
        os.path.join(save_dir, "1.png"),
        img
    )

def make_mask_from_deepcrack_img(img, parent_folder="random", THICKNESS=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200, apertureSize=3)
    h, w = gray.shape
    MIN_LINE_RATIO = 0.2
    min_len = int(min(h, w) * MIN_LINE_RATIO)
    # Probabilistic Hough Transform (actual line segments)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=120,       # loosen threshold to detect more lines
        minLineLength=15,   # ignore tiny noise
        maxLineGap=20       # merge nearby points
    )

    # --- Create empty binary mask ---
    mask = np.zeros(gray.shape, dtype=np.uint8)

    # Draw horizontal/vertical only, thicker lines

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Keep only horizontal or vertical lines
            if dx > dy and dy < dx * 0.05:       # horizontal
                cv2.line(mask, (x1, y1), (x2, y2), 255, THICKNESS)
            elif dy > dx and dx < dy * 0.05:     # vertical
                cv2.line(mask, (x1, y1), (x2, y2), 255, THICKNESS)

    # Save binary mask
    
    # cv2.imwrite(f'default_grids_mask/{parent_folder}.jpg', mask)
    return mask

def remove_grid_pieces_only(
    binary_img,
    min_piece_ratio=0.7,
    angle_tol_deg=3,
    gap_tol_ratio=0.15,
    bottom_ignore_px=25,   # 👈 NEW
):
    """
    Removes small isolated line fragments ("pieces").
    Also removes everything near the bottom of the image.
    """

    h, w = binary_img.shape
    out = np.zeros_like(binary_img)

    lines = cv2.HoughLinesP(
        binary_img,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.2 * min(h, w)),
        maxLineGap=15,
    )

    if lines is None:
        return out

    horizontals, verticals = [], []

    for x1, y1, x2, y2 in lines[:, 0]:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # -------------------------------------------------
        # Rule E: Kill lower-height region (absolute)
        # -------------------------------------------------
        if cy > h - bottom_ignore_px:
            continue

        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        length = np.hypot(dx, dy)

        if angle < angle_tol_deg:
            horizontals.append((x1, y1, x2, y2, length))
        elif abs(angle - 90) < angle_tol_deg:
            verticals.append((x1, y1, x2, y2, length))

    def filter_pieces(lines, axis_len, axis="y"):
        if len(lines) < 2:
            return []

        coords = []
        for x1, y1, x2, y2, _ in lines:
            coords.append((y1 + y2) // 2 if axis == "y" else (x1 + x2) // 2)

        coords = np.array(coords)
        gaps = np.abs(coords[:, None] - coords[None, :])
        gaps = gaps[gaps > 0]

        if len(gaps) == 0:
            return []

        G = np.median(gaps)

        kept = []

        for (x1, y1, x2, y2, length), c in zip(lines, coords):
            if length >= min_piece_ratio * axis_len:
                kept.append((x1, y1, x2, y2))
                continue

            deltas = np.abs(coords - c)
            if np.any(np.abs(deltas - G) < gap_tol_ratio * G):
                kept.append((x1, y1, x2, y2))

        return kept

    h_keep = filter_pieces(horizontals, w, "y")
    v_keep = filter_pieces(verticals, h, "x")

    for x1, y1, x2, y2 in h_keep + v_keep:
        cv2.line(out, (x1, y1), (x2, y2), 255, 3)

    return out


def numpy_image_to_base64(img_np):
    """
    Convert uint8 numpy image → base64 PNG string
    """
    if img_np.dtype != np.uint8:
        raise ValueError("Image must be uint8")

    if img_np.ndim == 2:
        img_pil = Image.fromarray(img_np, mode="L")
    else:
        img_pil = Image.fromarray(img_np)

    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def inpaint_with_mask(
    image_np,
    mask_np,
    url,
    prompt="",
    negative_prompt="",
    ldm_steps=20,
    hd_strategy="Crop",
    cv2_flag="INPAINT_NS",
    cv2_radius=4,
):
    """
    image_np : uint8 numpy image (H,W,3)
    mask_np  : uint8 numpy mask (H,W) or (H,W,3)
    returns  : uint8 numpy image
    """

    # --- Safety checks ---
    if image_np.dtype != np.uint8 or mask_np.dtype != np.uint8:
        raise ValueError("Both image and mask must be uint8")

    # Ensure mask is single-channel
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # --- Encode images ---
    image_b64 = numpy_image_to_base64(image_np)
    mask_b64 = numpy_image_to_base64(mask_np)

    # --- Payload ---
    data = {
        "image": image_b64,
        "mask": mask_b64,
        "ldm_steps": ldm_steps,
        "hd_strategy": hd_strategy,
        "cv2_flag": cv2_flag,
        "cv2_radius": cv2_radius,
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }

    headers = {"Content-Type": "application/json"}

    # --- Request ---
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()

    # --- Decode response ---
    content_type = response.headers.get("Content-Type", "")

    if "application/json" in content_type:
        out_img_str = response.json()
        out_img_bytes = base64.b64decode(out_img_str)
        out_img = Image.open(io.BytesIO(out_img_bytes))

    elif "image" in content_type or "octet-stream" in content_type:
        out_img = Image.open(io.BytesIO(response.content))

    else:
        raise ValueError(f"Unknown response type: {content_type}")

    out_img.save("output_from_ngrok.png")
    return np.array(out_img, dtype=np.uint8)

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
import numpy as np

def intersect_masks(A, B):
    """
    Compute intersection of two binary masks.

    Args:
        A (np.ndarray): Binary mask (0/255 or 0/1)
        B (np.ndarray): Binary mask (0/255 or 0/1)

    Returns:
        np.ndarray: Binary intersection mask (0/255)
    """

    if A.shape != B.shape:
        raise ValueError("Masks must have the same shape")

    intersection = np.logical_and(A > 0, B > 0)

    return (intersection.astype(np.uint8) * 255)


# a = cv2.imread("gray_image_combined_testing_dots_cleanup_111.png", cv2.IMREAD_GRAYSCALE)
# b = cv2.imread("gray_image_B_11.png", cv2.IMREAD_GRAYSCALE)

# intersection = intersect_masks(a, b)
# cv2.imwrite("11_result_by_intersection.png", intersection)

# img = cv2.imread("C-sample-sharp4.jpg")
# # gaussian_blur = cv2.GaussianBlur(img,(7,7),sigmaX=2)
# # img = cv2.addWeighted(img,7.5,gaussian_blur,-6.5,0)

# # _,_,m = remove_gridlines(img, 'grid_withoutpieces', activate_grid_mask_recreation=True)
# deepCrackImg = start_deepcrack_pipeline(img,
#                                         f"{crack_segmentation_dir_string}/tiles2_s",
#                                         original_h=img.shape[0],
#                                         original_w=img.shape[1], tile_size=742,inc_contrast=True)
# # # mask = make_mask_from_deepcrack_img(deepCrackImg, 'parent_folder', THICKNESS=2)
# # # out = remove_grid_pieces_only(m)
# cv2.imwrite("testing-C-sharp4.jpg", deepCrackImg)



# img_b64 = img_to_b64("C-sample.jpg")
# img = realesrgan_to_np(img_b64, 2)
# cv2.imwrite("C-sample-sharp2-success.jpg", img)


# img_b64 = img_to_b64("C-sample.jpg")

# r = requests.post(
#     "https://unfunereal-unconvertibly-tresa.ngrok-free.dev/api/v1/run_plugin_gen_image",
#     json={
#         "name": "RealESRGAN",
#         "image": img_b64,
#         "scale": 4
#     }
# )
# print("Status code:", r.status_code)
# with open("C-sample-sharp4.jpg", "wb") as f:
#     f.write(r.content)
# b64_to_img(r.json(), "C-sample-testing-from-api.jpg")
