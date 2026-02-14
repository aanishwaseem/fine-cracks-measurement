import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# Import YOUR actual functions
from utils import getBinaryImage
from remove_gridlines import remove_gridlines
from crack import get_binary_image_of_cracks   # <-- change if needed
from scale_image import scale_image
# ------------------------------
# SETTINGS
# ------------------------------
THRESHOLD_VALUE = 23
GRID_MASK_THICKNESS = 3
MAKE_REFERENCE_IMAGE = True
ACTIVATE_GRID_MASK_RECREATION = True
REFERENCE_FOLDER = "references"


# ------------------------------
# Select Image
# ------------------------------
def select_image():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Image for Reference",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not file_path:
        messagebox.showerror("Error", "No image selected.")
        return None

    return file_path
def threshold_tuning_gui(binary_mask, initial_threshold=23, initial_alpha=0.68, initial_beta=12):
    window_name = "Crack Threshold Tuning"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    # Trackbars
    cv2.createTrackbar("Threshold", window_name, initial_threshold, 255, lambda x: None)
    cv2.createTrackbar("Alpha x100", window_name, int(initial_alpha * 100), 200, lambda x: None)
    cv2.createTrackbar("Beta", window_name, int(initial_beta), 100, lambda x: None)

    confirmed_values = None

    while True:
        threshold = cv2.getTrackbarPos("Threshold", window_name)
        alpha = cv2.getTrackbarPos("Alpha x100", window_name) / 100.0
        beta = cv2.getTrackbarPos("Beta", window_name)

        # Generate preview
        preview = get_binary_image_of_cracks(
            binary_mask,
            threshold=threshold,
            alpha=alpha,
            beta=beta
        )

        display = preview.copy()
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        info_text = f"Threshold={threshold}  Alpha={alpha:.2f}  Beta={beta}"
        cv2.putText(display, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.putText(display, "ENTER = Confirm | ESC = Cancel",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # ENTER
            confirmed_values = (threshold, alpha, beta)
            break
        elif key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)

    return confirmed_values
# ------------------------------
# Create Reference
# ------------------------------
def create_reference():
    image_path = select_image()
    if image_path is None:
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return
    image = scale_image(image, 2)
    print("[INFO] Removing gridlines...")
    output_img, deepcrack_img_with_grids, mask = remove_gridlines(
        image,
        os.path.dirname(image_path),
        GRID_MASK_THICKNESS,
        activate_grid_mask_recreation=ACTIVATE_GRID_MASK_RECREATION,
        make_reference_image=MAKE_REFERENCE_IMAGE
    )

    print("[INFO] Running Deep Crack Extraction...")
    binary_mask = getBinaryImage(
        image,
        deepcrack_img_with_grids,
        mask
    )

    if binary_mask is None:
        print("[ERROR] Binary mask generation failed.")
        return

    print("[INFO] Opening threshold tuning window...")

    confirmed = threshold_tuning_gui(
        binary_mask,
        initial_threshold=THRESHOLD_VALUE,
        initial_alpha=0.68,
        initial_beta=12
    )

    if confirmed is None:
        print("[INFO] User cancelled threshold tuning.")
        return

    confirmed_threshold, confirmed_alpha, confirmed_beta = confirmed

    cracks_binary = get_binary_image_of_cracks(
        binary_mask,
        threshold=confirmed_threshold,
        alpha=confirmed_alpha,
        beta=confirmed_beta
    )

    # Create references folder
    os.makedirs(REFERENCE_FOLDER, exist_ok=True)

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    save_path = os.path.join(
        REFERENCE_FOLDER,
        f"{name[:16]}_reference.png"
    )

    cv2.imwrite(save_path, cracks_binary)

    print(f"[SUCCESS] Reference saved at: {save_path}")


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    create_reference()
