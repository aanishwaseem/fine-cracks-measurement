"""
Crack Visualization Module
──────────────────────────
Generates professional overlay visualizations from a CrackAnalyse instance.

Usage:
    from crack_analysis import CrackAnalyse
    from crack_visualization import visualize_crack_overlay

    analyser = CrackAnalyse(predict_image_array=my_mask)
    visualize_crack_overlay(analyser)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def _to_rgb(img):
    """Convert a grayscale or single-channel image to 3-channel RGB float [0, 1]."""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / img.max()
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


def visualize_crack_overlay(crack_analyser, alpha_mask=0.4, alpha_heatmap=0.6):
    """
    Produce a single figure with three professional subplots:

      1. Crack Segmentation Overlay  – red mask over original
      2. Crack Skeleton Overlay      – green skeleton over original
      3. Crack Width Heatmap         – jet colormap over original

    Parameters
    ----------
    crack_analyser : CrackAnalyse
        A fully-initialised CrackAnalyse instance.
    alpha_mask : float
        Opacity of the binary mask overlay (default 0.4).
    alpha_heatmap : float
        Opacity of the width heatmap overlay (default 0.6).
    """
    # ------------------------------------------------------------------
    # 1. Retrieve data from the analyser
    # ------------------------------------------------------------------
    original = crack_analyser.get_prediction()      # grayscale prediction image
    binary_mask = crack_analyser.img_bnr.astype(bool)
    skeleton = crack_analyser.get_skeleton()         # already slightly dilated
    width_map = crack_analyser.img_skl               # float width map

    # ------------------------------------------------------------------
    # 2. Prepare base RGB image
    # ------------------------------------------------------------------
    base_rgb = _to_rgb(original)

    # ------------------------------------------------------------------
    # 3. Subplot 1 – Crack Segmentation Overlay (red)
    # ------------------------------------------------------------------
    mask_overlay = base_rgb.copy()
    red = np.array([1.0, 0.0, 0.0])
    mask_overlay[binary_mask] = (
        (1 - alpha_mask) * mask_overlay[binary_mask] + alpha_mask * red
    )

    # ------------------------------------------------------------------
    # 4. Subplot 2 – Skeleton Overlay (green, dilated for visibility)
    # ------------------------------------------------------------------
    skel_bool = skeleton > 0
    # Extra dilation so the thin skeleton is clearly visible
    skel_dilated = ndi.binary_dilation(skel_bool, iterations=1)

    skel_overlay = base_rgb.copy()
    green = np.array([0.0, 1.0, 0.0])
    skel_overlay[skel_dilated] = (
        (1 - alpha_mask) * skel_overlay[skel_dilated] + alpha_mask * green
    )

    # ------------------------------------------------------------------
    # 5. Subplot 3 – Width Heatmap Overlay (jet)
    # ------------------------------------------------------------------
    # Convert width map from pixels to mm using scaling factors
    avg_scaling_factor = (crack_analyser.scaling_factor_x + crack_analyser.scaling_factor_y) / 2.0
    width_map_mm = width_map * avg_scaling_factor
    
    # Normalise width map to [0, 1]
    wmax_mm = width_map_mm.max()
    if wmax_mm > 0:
        width_norm = width_map_mm / wmax_mm
    else:
        width_norm = width_map_mm.copy()

    cmap = plt.cm.jet
    heatmap_rgba = cmap(width_norm)           # (H, W, 4) RGBA
    heatmap_rgb = heatmap_rgba[:, :, :3]      # drop alpha channel

    # Blend only where there is a non-zero width value
    heat_mask = width_map > 0
    heat_overlay = base_rgb.copy()
    heat_overlay[heat_mask] = (
        (1 - alpha_heatmap) * heat_overlay[heat_mask]
        + alpha_heatmap * heatmap_rgb[heat_mask]
    )

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.imshow(heat_overlay, aspect="equal")
    ax.set_title("Crack Width Heatmap", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet",
                               norm=plt.Normalize(vmin=0, vmax=wmax_mm if wmax_mm > 0 else 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Crack Width (mm)", fontsize=12)

    plt.tight_layout()
    plt.show()
