
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


class MaskTuningUI:
    def __init__(self, reference_mask, generated_mask):
        self.window = "Mask Tuning"
        self.ref = (reference_mask > 0).astype(np.uint8) * 255
        self.gen = generated_mask.astype("uint8")
        self.full_screen = False
        self.ref_copy = (reference_mask > 0).astype(np.uint8) * 255
        self.gen_copy = generated_mask.astype("uint8")
        self.confirmed_threshold = None

        # Ensure same size
        if self.ref.shape != self.gen.shape:
            self.gen = cv2.resize(
                self.gen,
                (self.ref.shape[1], self.ref.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        self.gray_thresh = 55
        self.ref_thickness = 1

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        cv2.createTrackbar(
            "Threshold",
            self.window,
            self.gray_thresh,
            255,
            self._on_trackbar
        )

        cv2.createTrackbar(
            "Ref. Thick",
            self.window,
            self.ref_thickness,
            20,
            self._on_trackbar
        )
        cv2.setMouseCallback(self.window, self._on_mouse)
        self.fullscreen_window = "Fullscreen"
        self.last_images = None  # will store current images


    def _on_trackbar(self, _):
        try:
            self.gray_thresh = cv2.getTrackbarPos("Threshold", self.window)
            self.ref_thickness = max(
                1, cv2.getTrackbarPos("Ref. Thick", self.window)
            )
        except cv2.error:
            pass
    def _process(self):
        # Binary generated mask
        _, gen_bin = cv2.threshold(
            self.gen,
            self.gray_thresh,
            255,
            cv2.THRESH_BINARY
        )

        # Binary reference (safety)
        ref_bin = (self.ref > 0).astype(np.uint8) * 255

        # Thickness kernel
        kernel = np.ones(
            (2 * self.ref_thickness + 1, 2 * self.ref_thickness + 1),
            np.uint8
        )

        # Hard thickening
        ref_thick = cv2.dilate(ref_bin, kernel)

        final = cv2.bitwise_and(ref_thick, gen_bin)

        return gen_bin, ref_thick, final
    def _on_mouse(self, event, x, y, flags, param):
        if self.last_images is None:
            return

        ref_thick, gen_bin, final = self.last_images
        h, w = ref_thick.shape
        section_width = w

        # ---------------- LEFT CLICK (Fullscreen Preview) ----------------
        if event == cv2.EVENT_LBUTTONDOWN:

            if x < section_width:
                img = ref_thick
            elif x < section_width * 2:
                img = gen_bin
            else:
                img = final
            self.full_screen = True
            cv2.namedWindow(self.fullscreen_window, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.fullscreen_window,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            cv2.imshow(self.fullscreen_window, img)

        # ---------------- RIGHT CLICK (Change Reference) ----------------
        if event == cv2.EVENT_RBUTTONDOWN:

            # Only allow if clicking inside reference section (left panel)
            if x < section_width:
                print("Opening file browser to change reference...")

                # Hide tkinter root window
                root = tk.Tk()
                root.withdraw()

                file_path = filedialog.askopenfilename(
                    title="Select New Reference Mask",
                    filetypes=[
                        ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif"),
                        ("All Files", "*.*")
                    ]
                )

                root.destroy()

                if file_path:
                    new_ref = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                    if new_ref is not None:
                        # Resize to match generated mask
                        new_ref = cv2.resize(
                            new_ref,
                            (self.gen.shape[1], self.gen.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                        self.ref = (new_ref > 0).astype(np.uint8) * 255
                        print("Reference updated successfully.")
                    else:
                        print("Failed to load selected image.")

    def run(self):
        apply_changes = True
        while True:
            gen_bin, ref_thick, final = self._process()

            # Store images for mouse click access
            self.last_images = (ref_thick, gen_bin, final)

            preview = np.hstack([
                cv2.cvtColor(ref_thick, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gen_bin, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            ])

            cv2.imshow(self.window, preview)

            key = cv2.waitKey(30) & 0xFF

            # ESC or q closes everything
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                apply_changes = True
                break
            # Close fullscreen with 'c'
            if key == ord('c'):
                if self.full_screen:
                    cv2.destroyWindow(self.fullscreen_window)
                    self.full_screen = False
        
        if apply_changes:
            self.confirmed_threshold = self.gray_thresh
            self.confirmed_ref_thickness = self.ref_thickness
        cv2.destroyWindow(self.window)
        if self.full_screen:
            cv2.destroyWindow(self.fullscreen_window)        
        return apply_changes
    def get_confirmed_threshold(self):
        return self.confirmed_threshold

    def get_final_mask(self):
        _, _, final = self._process()
        return final


# reference_mask = cv2.imread("default_grids_mask/MASK_GEMINI.png", cv2.IMREAD_GRAYSCALE)
# generated_mask = cv2.imread("default_grids_mask/MASK_IOPAINT.png", cv2.IMREAD_GRAYSCALE)

# ui = MaskTuningUI(reference_mask, generated_mask)
# ui.run()

# final_mask = ui.get_final_mask()
# cv2.imwrite("extra_imgs/final_mask.png", final_mask)

#TIP - if ref.mask is thin, increase threshold to get more pixels. If ref.mask is thick, decrease threshold. The reference mask is considered a "guideline or a bound" where the cracks can.