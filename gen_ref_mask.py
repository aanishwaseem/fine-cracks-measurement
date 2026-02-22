import os
import re
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# Import YOUR actual functions
from utils import getBinaryImage
from remove_gridlines import remove_gridlines
from scale_image import scale_image
from reference_image_editor import ReferenceImageEditor
# ------------------------------
# SETTINGS
# ------------------------------
THRESHOLD_VALUE = 23
GRID_MASK_THICKNESS = 3
MAKE_REFERENCE_IMAGE = True
ACTIVATE_GRID_MASK_RECREATION = True
REFERENCE_FOLDER = "references"


# ------------------------------
# Helpers
# ------------------------------

def sanitize_folder_name(folder_path):
    """Convert a folder path to a safe filename string."""
    name = os.path.basename(os.path.normpath(folder_path))
    # Replace non-alphanumeric chars (except - and _) with underscore
    return re.sub(r'[^\w\-]', '_', name)


def get_reference_path(dataset_folder):
    """Return the expected reference image path for a dataset folder."""
    sanitized = sanitize_folder_name(dataset_folder)
    return os.path.join(REFERENCE_FOLDER, f"{sanitized}_ref.png")


def _pick_last_image(dataset_folder):
    """Pick the last image file (alphabetically) from the dataset folder."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(extensions)
    ])
    if not image_files:
        return None
    last_file = image_files[-1]
    return os.path.join(dataset_folder, last_file)


# ------------------------------
# Select Image (kept for standalone/debug use)
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
        root.destroy()
        return None

    root.destroy()
    return file_path


# ------------------------------
# Create Reference
# ------------------------------

def create_reference(dataset_folder=None, image_path=None):
    """
    Generate a reference image for a dataset folder.

    Args:
        dataset_folder: Path to dataset folder. Last image is auto-selected.
        image_path:     (Optional) Explicit image path for debugging.

    Returns:
        str or None: Path to the saved reference image, or None on failure/cancel.
    """
    # Determine the image to use
    if image_path is None:
        if dataset_folder is None:
            # Fallback: manual selection (standalone mode)
            image_path = select_image()
            if image_path is None:
                return None
            dataset_folder = os.path.dirname(image_path)
        else:
            image_path = _pick_last_image(dataset_folder)
            if image_path is None:
                print(f"[ERROR] No image files found in {dataset_folder}")
                return None
            print(f"[INFO] Auto-selected last image: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return None
    
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
        return None

    print("[INFO] Opening Reference Image Editor...")

    # Launch the modern UI editor
    editor = ReferenceImageEditor(
        original_image=image,
        binary_mask=binary_mask,
        deepcrack_img=deepcrack_img_with_grids,
        grid_mask=mask
    )
    
    confirmed = editor.run()
    
    if not confirmed:
        print("[INFO] User cancelled reference editor.")
        return None
    
    final_reference = editor.get_final_reference()
    
    # Save using consistent naming based on dataset folder
    os.makedirs(REFERENCE_FOLDER, exist_ok=True)
    save_path = get_reference_path(dataset_folder)

    cv2.imwrite(save_path, final_reference)
    print(f"[SUCCESS] Reference saved at: {save_path}")
    return save_path


# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Usage: python gen_ref_mask.py <dataset_folder>
        create_reference(dataset_folder=sys.argv[1])
    else:
        # Interactive: prompt for folder
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        root.destroy()
        if folder:
            create_reference(dataset_folder=folder)
        else:
            print("[ERROR] No folder selected.")
