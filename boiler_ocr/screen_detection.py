#!/usr/bin/env python3
"""
Automatic boiler-display screen detection.

The whole idea in one sentence: the LCD glass is bright and is framed by a
very dark bezel, so if we threshold the photo into "bright" vs "dark", the dark
bezel becomes a black ring that isolates the glass as its own bright blob in the
middle of the picture. We grab that blob, take its four corners, and warp it to
a fixed-size canonical image. ROIs can then be defined once on the canonical
image instead of in absolute pixel coordinates, so they survive camera moves.

Run standalone to process the sample images and dump debug overlays:
    python screen_detection.py                # processes SAMPLES below
    python screen_detection.py img1.jpg ...   # processes given images
Outputs go to ./screen_debug/
"""

import cv2
import numpy as np
import os
import sys

# Canonical screen size (portrait). ROIs are defined relative to this.
CANONICAL_W = 400
CANONICAL_H = 640

# Sanity limits so we reject a bad detection instead of returning garbage.
# The physical glass is clearly taller than wide (~1.7) and fills a sensible
# chunk of the frame.
MIN_ASPECT = 1.3
MAX_ASPECT = 2.1
MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.95


def detect_screen(image):
    """
    Find the LCD glass in a BGR image.

    Returns the 4 corners as a (4, 2) float32 array ordered
    [top-left, top-right, bottom-right, bottom-left], or None if no
    screen-like blob was found.
    """
    h, w = image.shape[:2]

    # 1. Grayscale + a light blur to calm down sensor noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Split into bright vs dark. Otsu picks a good split point automatically,
    #    which keeps this working whether the backlight is orange or green.
    #    We threshold a bit below Otsu's level so the dim top of the glass (it
    #    has a top-to-bottom brightness gradient) still counts as "bright".
    otsu_level, _ = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (gray > 0.5 * otsu_level).astype(np.uint8) * 255

    # 3. Clean up: "open" erases thin bridges and small speckles, so a stray
    #    bezel highlight touching the glass can't merge them into one blob.
    ksize = max(5, (min(h, w) // 50) | 1)  # odd, scales with image size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

    # 4. The dark bezel ring separates the glass from the (also-bright) foam
    #    surround, so keep the bright blob sitting on the image centre.
    num, labels = cv2.connectedComponents(bright)
    center_label = labels[h // 2, w // 2]
    if center_label == 0:
        # Centre landed on a dark symbol; fall back to the biggest bright blob.
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore the black background
        if counts.max() == 0:
            return None
        center_label = counts.argmax()
    glass = (labels == center_label).astype(np.uint8) * 255

    # 5. Fit the tightest rotated rectangle around that blob and read its 4
    #    corners. The convex hull ignores small notches at the rounded corners.
    contours, _ = cv2.findContours(glass, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cv2.convexHull(blob))
    (rw, rh) = rect[1]
    if rw == 0 or rh == 0:
        return None

    # 6. Reject anything that doesn't look like the screen.
    aspect = max(rw, rh) / min(rw, rh)
    area_frac = cv2.contourArea(blob) / float(w * h)
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return None
    if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
        return None

    return order_corners(cv2.boxPoints(rect))


def order_corners(pts):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)          # x + y  -> smallest is TL, largest is BR
    d = np.diff(pts, axis=1).ravel()  # y - x -> smallest is TR, largest is BL
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def screen_matrix(corners, size=(CANONICAL_W, CANONICAL_H)):
    """Perspective matrix mapping this image's screen onto the canonical rect."""
    cw, ch = size
    dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst)


def warp_to_canonical(image, corners, size=(CANONICAL_W, CANONICAL_H)):
    """Perspective-warp the detected screen to a fixed canonical image."""
    M = screen_matrix(corners, size)
    return cv2.warpPerspective(image, M, size), M


def _map_points(points, M):
    """Push a list of (x, y) points through a perspective matrix."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def canonical_roi_to_image(canon_roi, corners, size=(CANONICAL_W, CANONICAL_H)):
    """
    Project a canonical ROI back onto a real image whose screen corners are
    known. Returns the 4 polygon corners (they form a quadrilateral once the
    camera is at an angle) as an int array ready for cv2.polylines.
    """
    cw, ch = size
    dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    # canonical -> image is the inverse of image -> canonical
    M_inv = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))
    y1, y2, x1, x2 = canon_roi
    box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return _map_points(box, M_inv).astype(int)


# ---------------------------------------------------------------------------
# Standalone debug harness
# ---------------------------------------------------------------------------

SAMPLES = [
    "../samples/calibration_image_20260505_233357.jpg",
    "../samples/20251108_210821.jpg",
    "../samples/20260818_211354.jpg",
    "../samples/feed/photo_feed.jpg",
    "../samples/feed/snapshot_camera_zero2w_12_1_2025, 10_48_00 PM.jpg",
    "../samples/feed/snapshot_camera_zero2w_12_1_2025, 11_29_05 PM.jpg",
    "../samples/feed/snapshot_camera_zero2w_12_2_2025, 12_12_25 AM.jpg",
    "../samples/feed/snapshot_camera_zero2w_12_2_2025, 1_24_29 PM.jpg",
]
CORNER_LABELS = ["TL", "TR", "BR", "BL"]


def _draw_screen_box(image, corners):
    pts = corners.astype(int)
    cv2.polylines(image, [pts], True, (0, 0, 255), 2)
    for i, pt in enumerate(pts):
        cv2.circle(image, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(image, CORNER_LABELS[i], (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def _grid(warped, step=40):
    g = warped.copy()
    h, w = g.shape[:2]
    for x in range(0, w, step):
        cv2.line(g, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(g, (0, y), (w, y), (0, 255, 0), 1)
    return g


def main(argv):
    """Geometry check: draw the detected screen box and the flattened warp."""
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "screen_debug")
    os.makedirs(out_dir, exist_ok=True)

    images = argv[1:] if len(argv) > 1 else SAMPLES
    print(f"Processing {len(images)} image(s) -> {out_dir}\n")

    for rel in images:
        path = rel if os.path.isabs(rel) else os.path.join(here, rel)
        img = cv2.imread(path)
        name = os.path.splitext(os.path.basename(rel))[0].replace(" ", "_").replace(",", "")
        if img is None:
            print(f"  [SKIP] cannot read {rel}")
            continue
        corners = detect_screen(img)
        if corners is None:
            print(f"  [FAIL] {name}: no screen found")
            continue
        w_top = np.linalg.norm(corners[1] - corners[0])
        h_left = np.linalg.norm(corners[3] - corners[0])
        aspect = max(w_top, h_left) / max(1e-6, min(w_top, h_left))
        print(f"  [ OK ] {name:44s} {img.shape[1]}x{img.shape[0]}  aspect={aspect:.3f}")

        overlay = img.copy()
        _draw_screen_box(overlay, corners)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{name}.jpg"), overlay)
        warped, _ = warp_to_canonical(img, corners)
        cv2.imwrite(os.path.join(out_dir, f"warped_{name}.jpg"), _grid(warped))

    print(f"\nDone. Open overlay_*.jpg and warped_*.jpg in {out_dir}")


if __name__ == "__main__":
    main(sys.argv)
