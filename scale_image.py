

import base64
import requests
import cv2
import numpy as np
url = 'http://127.0.0.1:8000/api/v1/run_plugin_gen_image'
def img_to_b64(img):
    """
    Accepts:
      - np.ndarray (BGR or grayscale)
      - OR file path (str)
    Returns:
      - base64 string (no data URI)
    """

    if isinstance(img, str):
        with open(img, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    if isinstance(img, np.ndarray):
        # Encode to PNG (lossless – better for cracks)
        success, buffer = cv2.imencode(".png", img)
        if not success:
            raise RuntimeError("cv2.imencode failed")

        return base64.b64encode(buffer).decode("utf-8")

    raise TypeError("img_to_b64 expects file path or numpy array")

import base64


def realesrgan_to_np(img_b64, scale=4):
    r = requests.post(
        url,
        json={
            "name": "RealESRGAN",
            "image": img_b64,
            "scale": scale
        },
        timeout=180
    )

    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"RealESRGAN failed:")

    # bytes → numpy
    img_np = np.frombuffer(r.content, dtype=np.uint8)

    # decode (OpenCV returns BGR)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img_np is None:
        raise RuntimeError("OpenCV failed to decode RealESRGAN output")
    return img_np

def scale_image(img, scale):
    if (scale == 2 and img.shape[1] > 1800) or (scale == 4 and img.shape[1] > 3000):
        return img

    # Super-resolve
    img_b64 = img_to_b64(img)
    output = realesrgan_to_np(img_b64, scale)

    orig_h, orig_w = img.shape[:2]
    target_h, target_w = orig_h * scale, orig_w * scale

    # Force exact size
    output_h, output_w = output.shape[:2]
    if output_h != target_h or output_w != target_w:
        output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return output
# img_b64 = img_to_b64("references/cropped_B.jpg")
# img = realesrgan_to_np(img_b64, 4)
# cv2.imwrite("references/cropped_B_scaled4.jpg", img)