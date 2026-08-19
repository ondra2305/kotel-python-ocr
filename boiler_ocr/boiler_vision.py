#!/usr/bin/env python3
"""
Boiler-display vision: locate the LCD, flatten it, and read every feature from
screen-relative ROIs (so readings survive the camera moving).

Detection is backlight- and shadow-robust; the reasoning lives on the functions
that do the work: _local_dark_mask, _edge_shadow_mask, detect_flame_level.
"""

import cv2
import numpy as np
import os
import json
import shutil
import subprocess
import tempfile

from .screen_detection import detect_screen, warp_to_canonical, CANONICAL_W, CANONICAL_H

# ---------------------------------------------------------------------------
# Configuration (ROIs relative to the screen; fractions of width/height)
# ---------------------------------------------------------------------------

# [y1, y2, x1, x2] as fractions 0..1 of the canonical screen.
DEFAULT_ROIS = {
    "temperature":    [0.014, 0.292, 0.098, 0.631],
    "flame_icon":     [0.531, 0.952, 0.080, 0.191],
    "bar_graph":      [0.530, 0.952, 0.186, 0.335],
    "heating_icon":   [0.564, 0.720, 0.634, 0.932],
    "hot_water_icon": [0.563, 0.721, 0.408, 0.636],
    "winter_icon":    [0.340, 0.550, 0.710, 0.950],
    "summer_icon":    [0.350, 0.540, 0.420, 0.710],
}

DEFAULT_PARAMS = {
    "icon_on_ratio": 0.08,       # min dark-stroke coverage to call an icon "on"
    "temp_min": 30,
    "temp_max": 90,
    "temp_ink_min": 0.10,        # min ink coverage in the temp ROI to call digits present
    "bar_segments": 6,
    "bar_fill_threshold": 0.30,
    "min_frame_brightness": 15,  # whole-frame mean below this = lens covered
    "adapt_block": 15,           # local-threshold neighborhood (odd)
    "adapt_c": 10,               # how much darker than local mean counts as ink
    "min_component": 25,         # drop dark blobs smaller than this (grain)
}

# Optional external override, written next to this file.
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "canonical_roi_config.json")


def load_config():
    """Return (rois, params), applying canonical_roi_config.json if present."""
    rois = {k: list(v) for k, v in DEFAULT_ROIS.items()}
    params = dict(DEFAULT_PARAMS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            rois.update(cfg.get("rois", {}))
            params.update(cfg.get("params", {}))
        except Exception as e:  # keep running on a broken config
            print(f"WARN: could not read {CONFIG_FILE}: {e}")
    return rois, params


HAVE_SSOCR = shutil.which("ssocr") is not None


# ---------------------------------------------------------------------------
# Low-level detectors (operate on the flattened canonical screen)
# ---------------------------------------------------------------------------

def _roi_slice(image, roi_frac):
    y1, y2, x1, x2 = roi_frac
    return image[int(y1 * CANONICAL_H):int(y2 * CANONICAL_H),
                 int(x1 * CANONICAL_W):int(x2 * CANONICAL_W)]


def _local_dark_mask(gray, params):
    """Icon ink: pixels darker than a SMALL local window. Small = immune to
    backlight level and gradients (a wide dark patch is its own background)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = g.shape
    bs = min(params["adapt_block"], (min(h, w) - 1) | 1)
    if bs < 3:
        return np.zeros_like(g)
    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, bs, params["adapt_c"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= params["min_component"]:
            clean[labels == i] = 255
    return clean


def _edge_shadow_mask(gray):
    """Dark region connected to the screen border (bezel shadow) - to ignore."""
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    darkish = (gray < 0.62 * otsu).astype(np.uint8)
    n, labels = cv2.connectedComponents(darkish, 8)
    border = set(labels[0, :]) | set(labels[-1, :]) | \
             set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    ring = np.isin(labels, list(border)).astype(np.uint8) * 255
    k = max(5, (min(gray.shape) // 60) | 1)
    return cv2.dilate(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def build_icon_mask(gray, params):
    """Whole-screen mask of real icon ink, with the edge shadow removed."""
    strokes = _local_dark_mask(gray, params)
    strokes[_edge_shadow_mask(gray) > 0] = 0
    return strokes


def _icon_ratio(icon_mask, roi_frac):
    sub = _roi_slice(icon_mask, roi_frac)
    return float(sub.mean() / 255) if sub.size else 0.0


def detect_flame_level(gray, roi_frac, params):
    """Count filled bar-graph segments bottom-up. Bars are solid blocks, so
    threshold relative to the bright glass rather than with the icon mask."""
    seg = _roi_slice(gray, roi_frac)
    if seg.size == 0:
        return None
    bg = np.percentile(seg, 80)                     # bright glass reference
    filled = (seg < bg - 40).astype(np.uint8)
    h = seg.shape[0]
    seg_h = h // params["bar_segments"]
    if seg_h == 0:
        return None
    active = 0
    for i in range(params["bar_segments"]):
        band = filled[h - (i + 1) * seg_h: h - i * seg_h, :]
        if band.size == 0:
            break
        if np.sum(band > 0) / band.size > params["bar_fill_threshold"]:
            active = i + 1
        else:
            break
    return active


def _warp_roi_fullres(image, corners, roi_frac):
    """Deskew a screen-relative ROI straight from the full-res image, keeping
    the digit resolution the small canonical warp would throw away."""
    y1, y2, x1, x2 = roi_frac
    canon = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) * \
        np.float32([CANONICAL_W, CANONICAL_H])
    dst_full = np.float32([[0, 0], [CANONICAL_W, 0],
                           [CANONICAL_W, CANONICAL_H], [0, CANONICAL_H]])
    m_inv = cv2.getPerspectiveTransform(dst_full, corners.astype(np.float32))
    src = cv2.perspectiveTransform(canon.reshape(-1, 1, 2), m_inv).reshape(-1, 2)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 2 or h < 2:
        return None
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    return cv2.warpPerspective(image, m, (w, h))


def _flatfield(gray):
    """Divide out a bright-background estimate to remove the display's
    brightness gradient (bg -> ~200); dark digits stay dark. This is what lets
    both the digit-present check and OCR cope with dim, uneven backlights."""
    k = max(15, (min(gray.shape) // 3) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return cv2.divide(gray, bg, scale=200)


def temp_has_digits(crop, params):
    """True if the temperature ROI actually shows digit ink (vs blank), so a
    failed read can be told apart from 'no number displayed'. Flat-fielding
    handles the gradient; the component filter drops glass grain so thick, faint
    digits still register while a blank screen does not."""
    if crop is None or crop.size == 0:
        return False
    mask = (_flatfield(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < 140).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    ink = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n)
              if stats[i, cv2.CC_STAT_AREA] >= 60) / mask.size
    return ink > params["temp_ink_min"]


def read_temperature(crop, params):
    """OCR the 2-digit temperature with ssocr; None if unreadable/unavailable."""
    if not HAVE_SSOCR or crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    votes = []
    # Grey only, across a spread of thresholds (covers high-contrast and dim /
    # orange backlights). Colour channels and flat-fielding recover a few dim
    # frames but AMPLIFY faint LCD ghost segments into confident misreads
    # (a "6" reads as "8"), which is worse than a miss for a temperature.
    for thr in (60, 80, 100, 120, 140, 160):
        _, proc = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, proc)
            try:
                out = subprocess.run(
                    ["ssocr", "--number-digits=2", "-d", "2", tmp.name],
                    capture_output=True, text=True, timeout=5)
                s = out.stdout.strip()
                if out.returncode == 0 and s.isdigit() and len(s) == 2 \
                        and params["temp_min"] <= int(s) <= params["temp_max"]:
                    votes.append(int(s))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
    if not votes:
        return None
    winner = max(set(votes), key=votes.count)
    # Require >=2 agreeing reads. A lone read on a degraded frame is usually a
    # misread, and a wrong temperature is worse than none.
    return winner if votes.count(winner) >= 2 else None


# ---------------------------------------------------------------------------
# Top-level analysis with health checks
# ---------------------------------------------------------------------------

def _blank_result(status, error=None):
    return {
        "status": status,
        "temperature": None,
        "temp_status": None,
        "flame_active": None,
        "flame_level": None,
        "heating_active": None,
        "hot_water_active": None,
        "mode": None,
        "error": error,
    }


def read_canonical(warped, rois=None, params=None, debug=False,
                   full_image=None, corners=None):
    """Read all features from a flattened screen with the given ROIs (fractions).
    No 'status' field - the caller sets that. Shared by analyze() and calibrator.
    Pass full_image+corners so the temperature is OCR'd from the full-res original
    (deskewed) instead of the downscaled canonical crop."""
    if rois is None or params is None:
        cfg_rois, cfg_params = load_config()
        rois = rois or cfg_rois
        params = params or cfg_params

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    icon_mask = build_icon_mask(gray, params)
    on = params["icon_on_ratio"]
    ratios = {name: _icon_ratio(icon_mask, rois[name])
              for name in ("flame_icon", "heating_icon", "hot_water_icon",
                           "winter_icon", "summer_icon")}
    winter = ratios["winter_icon"] > on
    summer = ratios["summer_icon"] > on

    if full_image is not None and corners is not None:
        temp_crop = _warp_roi_fullres(full_image, corners, rois["temperature"])
    else:
        temp_crop = _roi_slice(warped, rois["temperature"])
    # Three-way temperature outcome: read a number, digits present but OCR
    # failed, or no digits displayed at all.
    if not temp_has_digits(temp_crop, params):
        temp, temp_status = None, "absent"
    elif (temp := read_temperature(temp_crop, params)) is not None:
        temp_status = "ok"
    else:
        temp_status = "ocr_fail"
    res = {
        "temperature": temp,
        "temp_status": temp_status,
        "flame_active": ratios["flame_icon"] > on,
        "flame_level": detect_flame_level(gray, rois["bar_graph"], params),
        "heating_active": ratios["heating_icon"] > on,
        "hot_water_active": ratios["hot_water_icon"] > on,
        "mode": ("winter" if winter and not summer else
                 "summer" if summer and not winter else "unknown"),
        "error": None,
    }
    _validate(res)
    if debug:
        res["_ratios"] = {k: round(v, 3) for k, v in ratios.items()}
    return res


def analyze(image, debug=False):
    """Read the whole display; returns the readings dict with a 'status' field
    ('ok', or a failure reason meaning the values are None)."""
    rois, params = load_config()

    if image is None:
        return _blank_result("no_image")

    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray_full.mean()) < params["min_frame_brightness"]:
        return _blank_result("too_dark")          # lens covered / no light

    corners = detect_screen(image)
    if corners is None:
        return _blank_result("no_screen")          # obscured / moved / no glass

    warped, _ = warp_to_canonical(image, corners)
    res = read_canonical(warped, rois, params, debug=debug,
                         full_image=image, corners=corners)
    # A consistency violation means something was misread -> don't trust any of it.
    res["status"] = "inconsistent" if res["error"] else "ok"
    return res


def _validate(res):
    """Flag logically impossible combinations. Their presence means the vision
    misread, so the whole reading is untrusted (analyze turns this into an
    'inconsistent' status) - we do NOT guess a correction."""
    notes = []
    if res["heating_active"] and res["hot_water_active"]:
        notes.append("heating+hot_water both on")     # boiler drives one, not both
    lvl, flame = res["flame_level"], res["flame_active"]
    if lvl is not None and flame is not None:
        if flame and lvl == 0:
            notes.append("flame on but level 0")
        elif not flame and lvl > 0:
            notes.append("flame off but level>0")
    res["error"] = "; ".join(notes) if notes else None
