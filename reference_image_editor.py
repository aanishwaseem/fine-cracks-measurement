"""
═══════════════════════════════════════════════════════════════════════════════
REFERENCE IMAGE EDITOR - Modern UI for Binary Mask Configuration
═══════════════════════════════════════════════════════════════════════════════

A modern, user-friendly Tkinter-based tool for creating and editing reference
binary masks for crack detection. Features:
  ✓ Real-time side-by-side preview (original vs. binary)
  ✓ Interactive trackbars (Threshold, Alpha, Beta, Grid Thickness)
  ✓ Paint-like brush tool (White/Black modes)
  ✓ Adjustable brush size
  ✓ Undo stack (5 steps)
  ✓ Load new images on-the-fly
  ✓ Clean, professional Tkinter GUI

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from collections import deque
from functools import wraps

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMAGE PROCESSING HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_binary_image_of_cracks(gen_binary_mask, threshold=23, alpha=0.68, beta=12, thickness=0):
    """Convert binary mask to crack visualization with adjustable parameters.
    
    Args:
        gen_binary_mask: Input grayscale mask
        threshold: Fixed threshold for binary conversion
        alpha: Brightness/contrast multiplier
        beta: Brightness offset
        thickness: Morphological expansion of crack pixels (0 = none)
    """
    gray_image = gen_binary_mask.copy()
    brightened_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
    _, binary_image = cv2.threshold(brightened_image, threshold, 255, cv2.THRESH_BINARY_INV)
    binary_image = 255 - binary_image  # invert black <-> white
    
    # Apply thickness: erode white region to expand black crack pixels
    if thickness > 0:
        kernel_size = 2 * thickness + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary_image = cv2.erode(binary_image, kernel, iterations=1)
    
    return binary_image


def opencv_to_photoimage(cv_frame):
    """Convert OpenCV BGR image to PIL PhotoImage for Tkinter display."""
    cv_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)
    return ImageTk.PhotoImage(pil_img)


def resize_image_for_display(img, target_width=400, target_height=300):
    """Resize image while preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UNDO MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UndoStack:
    """Simple undo/redo stack (LIFO, max 5 states)."""
    
    def __init__(self, max_size=5):
        self.stack = deque(maxlen=max_size)
        self.max_size = max_size
    
    def push(self, state):
        """Push a state onto the stack. Accepts arrays or tuples of arrays."""
        if isinstance(state, tuple):
            self.stack.append(tuple(s.copy() for s in state))
        else:
            self.stack.append(state.copy())
    
    def pop(self):
        """Pop and return the most recent state, or None if empty."""
        if len(self.stack) > 0:
            return self.stack.pop()
        return None
    
    def is_empty(self):
        return len(self.stack) == 0
    
    def size(self):
        return len(self.stack)
    
    def clear(self):
        self.stack.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN UI CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReferenceImageEditor:
    """
    Modern reference image editor with Tkinter UI.
    
    Attributes:
        original_image : np.ndarray (BGR)
            Preprocessed original image (after scale_image + preprocessing)
        
        binary_mask : np.ndarray (grayscale)
            Raw binary mask from getBinaryImage() (before crack visualization)
        
        current_ref : np.ndarray (grayscale)
            Current edited reference binary image (what user sees)
        
        deepcrack_img : np.ndarray (BGR)
            Optional: DeepCrack result image
        
        grid_mask : np.ndarray (grayscale)
            Optional: Grid mask for remove_gridlines
    """
    
    def __init__(self, original_image, binary_mask, deepcrack_img=None, grid_mask=None):
        """
        Initialize the editor.
        
        Args:
            original_image: The scaled original image (BGR)
            binary_mask: The binary mask (grayscale) from getBinaryImage()
            deepcrack_img: Optional DeepCrack result
            grid_mask: Optional grid mask
        """
        self.original_image = original_image.copy()
        self.binary_mask = binary_mask.copy()
        self.deepcrack_img = deepcrack_img
        self.grid_mask = grid_mask
        
        # Current edited reference (what user sees in the preview)
        self.current_ref = binary_mask.copy()
        self.undo_stack = UndoStack(max_size=5)
        
        # Current parameters
        self.thickness = 0     # morphological thickness (replaces threshold)
        self.alpha = 0.68
        self.beta = 12
        self.grid_mask_thickness = 3
        
        # Brush overlay — tracks painted pixels independently of slider params
        h, w = binary_mask.shape[:2]
        self.brush_painted = np.zeros((h, w), dtype=bool)
        self.brush_values = np.zeros((h, w), dtype=np.uint8)
        
        # Brush settings
        self.brush_mode = "black"  # "white" or "black"
        self.brush_size = 5
        self.brush_cursor_id = None
        self.is_drawing = False
        self.last_pos = None
        
        # Result variables
        self.confirmed = False
        self.final_reference = None
        self.root = None
        
        # Display references
        self.canvas_left = None
        self.canvas_right = None
        self.photo_left = None
        self.photo_right = None
        self.pil_left = None
        self.pil_right = None
    
    def _update_preview(self):
        """Update the preview display and regenerate the reference binary image.
        
        Regenerates from params, then re-applies any brush overlay on top
        so brush edits are never lost when sliders change.
        """
        # Generate base reference from parameters
        base = get_binary_image_of_cracks(
            self.binary_mask,
            alpha=self.alpha,
            beta=self.beta,
            thickness=self.thickness
        )
        # Re-apply brush overlay on top of the freshly generated base
        base[self.brush_painted] = self.brush_values[self.brush_painted]
        self.current_ref = base
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the side-by-side display in the Tkinter canvas."""
        if self.canvas_left is None or self.canvas_right is None:
            return
        
        # Resize images for display (full width, stacked vertically)
        left_display = resize_image_for_display(self.original_image, 850, 280)
        right_display = resize_image_for_display(self.current_ref, 850, 280)
        
        # Convert grayscale reference to BGR for display
        if len(right_display.shape) == 2:
            right_display = cv2.cvtColor(right_display, cv2.COLOR_GRAY2BGR)
        
        # Convert to PIL Image first (store to prevent garbage collection)
        self.pil_left = Image.fromarray(cv2.cvtColor(left_display, cv2.COLOR_BGR2RGB))
        self.pil_right = Image.fromarray(cv2.cvtColor(right_display, cv2.COLOR_BGR2RGB))
        
        # Convert PIL to PhotoImage — bind to self.root so it registers
        # with the correct Tk interpreter (avoids "pyimage doesn't exist")
        self.photo_left = ImageTk.PhotoImage(self.pil_left, master=self.root)
        self.photo_right = ImageTk.PhotoImage(self.pil_right, master=self.root)
        
        # Clear canvases before drawing new images
        if not hasattr(self, "left_img_id"):
            self.left_img_id = self.canvas_left.create_image(0, 0, anchor=tk.NW)
        if not hasattr(self, "right_img_id"):
            self.right_img_id = self.canvas_right.create_image(0, 0, anchor=tk.NW)

        # Update existing canvas images instead of recreating
        self.canvas_left.itemconfig(self.left_img_id, image=self.photo_left)
        self.canvas_right.itemconfig(self.right_img_id, image=self.photo_right)

        # Keep strong references
        self.canvas_left.image = self.photo_left
        self.canvas_right.image = self.photo_right
            
    def _on_thickness_change(self, val):
        """Trackbar callback for crack thickness."""
        self.thickness = max(0, int(float(val)))
        self._update_preview()
    
    def _on_alpha_change(self, val):
        """Trackbar callback for alpha."""
        self.alpha = float(val) / 100.0
        self._update_preview()
    
    def _on_beta_change(self, val):
        self.beta = int(float(val))
        self._update_preview()
    
    def _on_brush_size_change(self, val):
        self.brush_size = max(1, int(float(val)))
    def _on_brush_mode_change(self, mode):
        """Toggle brush mode between white and black."""
        self.brush_mode = mode
        # Update button visuals
        if hasattr(self, 'brush_white_btn') and hasattr(self, 'brush_black_btn'):
            if mode == "white":
                self.brush_white_btn.config(text="\u25b6 Paint White")
                self.brush_black_btn.config(text="Paint Black")
            else:
                self.brush_white_btn.config(text="Paint White")
                self.brush_black_btn.config(text="\u25b6 Paint Black")
    
    def _get_display_geometry(self):
        """Compute displayed image size and scale factor for coordinate mapping."""
        ref_h, ref_w = self.current_ref.shape[:2]
        target_w, target_h = 850, 280
        scale = min(target_w / ref_w, target_h / ref_h)
        disp_w = int(ref_w * scale)
        disp_h = int(ref_h * scale)
        return disp_w, disp_h, scale
    
    def _canvas_to_image_coords(self, event):
        """Convert canvas event coordinates to original image coordinates."""
        ref_h, ref_w = self.current_ref.shape[:2]
        disp_w, disp_h, scale = self._get_display_geometry()
        if disp_w <= 0 or disp_h <= 0:
            return None
        
        # Image is anchored at (0, 0) NW — check bounds
        if event.x < 0 or event.x >= disp_w or event.y < 0 or event.y >= disp_h:
            return None
        
        img_x = int(event.x / scale)
        img_y = int(event.y / scale)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, ref_w - 1))
        img_y = max(0, min(img_y, ref_h - 1))
        return (img_x, img_y)
    
    def _on_canvas_hover(self, event):
        """Show brush cursor circle following the mouse on the reference canvas."""
        if self.canvas_right is None:
            return
        # Remove old cursor circle
        if self.brush_cursor_id is not None:
            self.canvas_right.delete(self.brush_cursor_id)
            self.brush_cursor_id = None
        
        _, _, scale = self._get_display_geometry()
        canvas_radius = max(1, self.brush_size * scale)
        
        x, y = event.x, event.y
        self.brush_cursor_id = self.canvas_right.create_oval(
            x - canvas_radius, y - canvas_radius,
            x + canvas_radius, y + canvas_radius,
            outline="red", width=1
        )
    
    def _on_canvas_leave(self, event):
        """Remove brush cursor circle when mouse leaves the canvas."""
        if self.brush_cursor_id is not None:
            self.canvas_right.delete(self.brush_cursor_id)
            self.brush_cursor_id = None
    
    def _on_canvas_right_click(self, event):
        """Handle mouse click on the reference preview canvas (drawing)."""
        if self.canvas_right is None:
            return
        
        coords = self._canvas_to_image_coords(event)
        if coords is None:
            return
        img_x, img_y = coords
        
        # Save brush overlay state for undo
        if not self.is_drawing:
            self.undo_stack.push((self.brush_painted.copy(), self.brush_values.copy()))
        
        self.is_drawing = True
        self.last_pos = (img_x, img_y)
        self._paint_on_image(img_x, img_y)
        self._refresh_display()
        self._on_canvas_hover(event)
    
    def _on_canvas_right_drag(self, event):
        """Handle mouse drag on the reference preview canvas."""
        if not self.is_drawing or self.last_pos is None:
            return
        
        coords = self._canvas_to_image_coords(event)
        if coords is None:
            return
        img_x, img_y = coords
        
        # Draw line from last position to current
        self._draw_line(self.last_pos, (img_x, img_y))
        self.last_pos = (img_x, img_y)
        self._refresh_display()
        self._on_canvas_hover(event)
    
    def _on_canvas_right_release(self, event):
        """Handle mouse release on the reference preview canvas."""
        self.is_drawing = False
        self.last_pos = None
        self._refresh_display()
    
    def _paint_on_image(self, x, y):
        """Paint a single point on the reference mask and brush overlay."""
        radius = self.brush_size
        color = 255 if self.brush_mode == "white" else 0
        
        # Paint on current display image
        cv2.circle(self.current_ref, (x, y), radius, color, -1)
        
        # Also record in brush overlay so edits survive slider changes
        cv2.circle(self.brush_painted.astype(np.uint8), (x, y), radius, 1, -1)
        # Re-derive brush_painted as bool from the uint8 stamp
        stamp = np.zeros_like(self.brush_painted, dtype=np.uint8)
        cv2.circle(stamp, (x, y), radius, 1, -1)
        self.brush_painted[stamp == 1] = True
        self.brush_values[stamp == 1] = color
    
    def _draw_line(self, pt1, pt2):
        """Draw a line between two points on the reference mask and brush overlay."""
        color = 255 if self.brush_mode == "white" else 0
        cv2.line(self.current_ref, pt1, pt2, color, self.brush_size)
        
        # Record in brush overlay
        stamp = np.zeros_like(self.brush_painted, dtype=np.uint8)
        cv2.line(stamp, pt1, pt2, 1, self.brush_size)
        self.brush_painted[stamp == 1] = True
        self.brush_values[stamp == 1] = color
    
    def _on_undo(self):
        """Undo the last brush stroke."""
        prev_state = self.undo_stack.pop()
        if prev_state is not None:
            self.brush_painted, self.brush_values = prev_state
            self._update_preview()  # regenerate from params + restored overlay
        else:
            messagebox.showinfo("Undo", "No more undo steps available.")
    
    def _on_reset(self):
        """Reset to the original reference (before any editing)."""
        if messagebox.askyesno("Reset", "Reset all changes to original?"):
            h, w = self.binary_mask.shape[:2]
            self.brush_painted = np.zeros((h, w), dtype=bool)
            self.brush_values = np.zeros((h, w), dtype=np.uint8)
            self.undo_stack.clear()
            self._update_preview()
    
    def _on_save(self):
        """Save the current reference as a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_ref)
            messagebox.showinfo("Success", f"Reference saved to:\n{file_path}")
            self.final_reference = self.current_ref
            self.confirmed = True
            self.root.quit()
    
    def _on_confirm(self):
        """Confirm and close the editor."""
        self.final_reference = self.current_ref
        self.confirmed = True
        self.root.quit()
    
    def _on_cancel(self):
        """Cancel and close the editor."""
        self.confirmed = False
        self.root.quit()
    
    def run(self):
        """
        Launch the editor UI.
        
        Returns:
            bool: True if user confirmed, False if cancelled
        """
        self.root = tk.Tk()
        self.root.title("Reference Image Editor")
        self.root.geometry("900x900")
        
        # ═══════════════════════════════════════════════════════════════════════
        #  TOP ACTION BAR (Save & Confirm buttons, right-aligned)
        # ═══════════════════════════════════════════════════════════════════════
        action_bar = ttk.Frame(self.root)
        action_bar.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        ttk.Button(action_bar, text="\u2715 Cancel", width=12, command=self._on_cancel).pack(side=tk.RIGHT, padx=3)
        ttk.Button(action_bar, text="\u2713 Confirm (No Save)", width=18, command=self._on_confirm).pack(side=tk.RIGHT, padx=3)
        ttk.Button(action_bar, text="\U0001f4be Save & Confirm", width=18, command=self._on_save).pack(side=tk.RIGHT, padx=3)
        
        # ═══════════════════════════════════════════════════════════════════════
        #  DISPLAY AREA — vertically stacked
        # ═══════════════════════════════════════════════════════════════════════
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Top: Original Image
        top_frame = ttk.LabelFrame(display_frame, text="Original Image", padding=5)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 3))
        
        self.canvas_left = tk.Canvas(top_frame, width=850, height=280, bg="gray20")
        self.canvas_left.pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Reference Binary Mask (editable)
        bottom_frame = ttk.LabelFrame(display_frame, text="Reference Binary Mask (click to paint)", padding=5)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(3, 0))
        
        self.canvas_right = tk.Canvas(bottom_frame, width=850, height=280, bg="gray20")
        self.canvas_right.pack(fill=tk.BOTH, expand=True)
        
        # Bind paint events to right canvas
        self.canvas_right.bind("<Button-1>", self._on_canvas_right_click)
        self.canvas_right.bind("<B1-Motion>", self._on_canvas_right_drag)
        self.canvas_right.bind("<ButtonRelease-1>", self._on_canvas_right_release)
        self.canvas_right.bind("<Motion>", self._on_canvas_hover)
        self.canvas_right.bind("<Leave>", self._on_canvas_leave)
        
        # ═══════════════════════════════════════════════════════════════════════
        #  CONTROL PANEL (Bottom)
        # ═══════════════════════════════════════════════════════════════════════
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Row 1: Thickness, Alpha
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="Thickness:", width=12).pack(side=tk.LEFT, padx=5)
        self.thickness_var = tk.IntVar(value=self.thickness)
        thickness_scale = ttk.Scale(
            row1, from_=0, to=20, variable=self.thickness_var,
            command=self._on_thickness_change, orient=tk.HORIZONTAL
        )
        thickness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.thickness_label = ttk.Label(row1, text=str(self.thickness), width=4)
        self.thickness_label.pack(side=tk.LEFT, padx=5)
        
        self.thickness_var.trace("w", lambda *args: self.thickness_label.config(text=str(self.thickness_var.get())))
        
        ttk.Label(row1, text="Alpha (x0.01):", width=15).pack(side=tk.LEFT, padx=5)
        self.alpha_var = tk.IntVar(value=int(self.alpha * 100))
        alpha_scale = ttk.Scale(
            row1, from_=0, to=200, variable=self.alpha_var,
            command=self._on_alpha_change, orient=tk.HORIZONTAL
        )
        alpha_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.alpha_label = ttk.Label(row1, text=f"{self.alpha:.2f}", width=6)
        self.alpha_label.pack(side=tk.LEFT, padx=5)
        
        self.alpha_var.trace("w", lambda *args: self.alpha_label.config(text=f"{self.alpha_var.get()/100:.2f}"))
        
        # Row 2: Beta, Brush Size
        row2 = ttk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2, text="Beta:", width=12).pack(side=tk.LEFT, padx=5)
        self.beta_var = tk.IntVar(value=self.beta)
        beta_scale = ttk.Scale(
            row2, from_=0, to=100, variable=self.beta_var,
            command=self._on_beta_change, orient=tk.HORIZONTAL
        )
        beta_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.beta_label = ttk.Label(row2, text=str(self.beta), width=4)
        self.beta_label.pack(side=tk.LEFT, padx=5)
        
        self.beta_var.trace("w", lambda *args: self.beta_label.config(text=str(self.beta_var.get())))
        
        ttk.Label(row2, text="Brush Size:", width=15).pack(side=tk.LEFT, padx=5)
        self.brush_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(
            row2, from_=1, to=30, variable=self.brush_var,
            command=self._on_brush_size_change, orient=tk.HORIZONTAL
        )
        brush_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.brush_label = ttk.Label(row2, text=str(self.brush_size), width=4)
        self.brush_label.pack(side=tk.LEFT, padx=5)
        
        self.brush_var.trace("w", lambda *args: self.brush_label.config(text=str(self.brush_var.get())))
        
        # Row 3: Brush Mode & Buttons
        row3 = ttk.Frame(control_frame)
        row3.pack(fill=tk.X, pady=5)
        
        ttk.Label(row3, text="Brush Mode:", width=12).pack(side=tk.LEFT, padx=5)
        
        self.brush_white_btn = ttk.Button(
            row3, text="Paint White", width=14,
            command=lambda: self._on_brush_mode_change("white")
        )
        self.brush_white_btn.pack(side=tk.LEFT, padx=3)
        
        self.brush_black_btn = ttk.Button(
            row3, text="\u25b6 Paint Black", width=14,
            command=lambda: self._on_brush_mode_change("black")
        )
        self.brush_black_btn.pack(side=tk.LEFT, padx=3)
        
        ttk.Button(row3, text="↶ Undo", width=10, command=self._on_undo).pack(side=tk.LEFT, padx=3)
        ttk.Button(row3, text="⟲ Reset", width=10, command=self._on_reset).pack(side=tk.LEFT, padx=3)
        

        # Initial display
        self._refresh_display()
        
        # Run the UI
        self.root.mainloop()
        
        return self.confirmed
    
    def get_final_reference(self):
        """Get the final edited reference image."""
        return self.final_reference if self.final_reference is not None else self.current_ref


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STANDALONE TEST (for debugging)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Create dummy test data
    test_original = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    test_binary = np.random.randint(0, 255, (400, 400), dtype=np.uint8)
    
    editor = ReferenceImageEditor(test_original, test_binary)
    confirmed = editor.run()
    
    if confirmed:
        print("[✓] Editor confirmed")
        result = editor.get_final_reference()
        print(f"Result shape: {result.shape}")
    else:
        print("[✕] Editor cancelled")
