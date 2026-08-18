#!/usr/bin/env python3
"""
Debug harness: run boiler_vision.analyze on the sample images and draw each ROI
+ the value read, to eyeball results. Writes debug/screen_debug/result_*.jpg.

    python debug/detection_preview.py [img ...]

Temperature needs ssocr; without it it shows None here.
"""

import os
import sys

import cv2

# Run as a plain script from the repo root: put the repo root on the path so the
# boiler_ocr package imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from boiler_ocr import boiler_vision as bv
from boiler_ocr.screen_detection import (detect_screen, warp_to_canonical,
                                         canonical_roi_to_image,
                                         CANONICAL_W, CANONICAL_H)

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

# label + BGR color per ROI name (names match boiler_vision.DEFAULT_ROIS).
STYLE = {
    "temperature":    ("Temp",   (255, 0, 255)),
    "flame_icon":     ("Flame",  (0, 255, 0)),
    "bar_graph":      ("Level",  (255, 0, 0)),
    "heating_icon":   ("Heat",   (0, 165, 255)),
    "hot_water_icon": ("Water",  (0, 255, 255)),
    "winter_icon":    ("Winter", (255, 128, 0)),
    "summer_icon":    ("Summer", (0, 128, 255)),
}


def _label_for(name, res):
    """Short result string to draw next to a box."""
    r = res.get("_ratios", {})
    if name == "temperature":
        return "Temp:%s" % (res["temperature"] if res["temperature"] is not None else "n/a")
    if name == "bar_graph":
        return "Level:%s/6" % (res["flame_level"] if res["flame_level"] is not None else "?")
    field = {"flame_icon": "flame_active", "heating_icon": "heating_active",
             "hot_water_icon": "hot_water_active",
             "winter_icon": None, "summer_icon": None}[name]
    lbl = STYLE[name][0]
    if name in ("winter_icon", "summer_icon"):
        on = res["mode"] == ("winter" if name == "winter_icon" else "summer")
    else:
        on = bool(res.get(field))
    return f"{lbl}:{'ON' if on else 'off'} {r.get(name, 0):.2f}"


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "screen_debug")
    os.makedirs(out_dir, exist_ok=True)
    rois = bv.load_config()[0]

    images = argv[1:] if len(argv) > 1 else SAMPLES
    for rel in images:
        path = rel if os.path.isabs(rel) else os.path.join(here, rel)
        img = cv2.imread(path)
        name = os.path.splitext(os.path.basename(rel))[0].replace(" ", "_").replace(",", "")
        if img is None:
            print(f"[SKIP] {rel}")
            continue

        res = bv.analyze(img, debug=True)
        if res["status"] != "ok":
            print(f"[{res['status'].upper()}] {name}")
            continue

        corners = detect_screen(img)
        warped, _ = warp_to_canonical(img, corners)
        for key, roi_frac in rois.items():
            label, color = STYLE.get(key, (key, (255, 255, 255)))
            y1 = int(roi_frac[0] * CANONICAL_H); y2 = int(roi_frac[1] * CANONICAL_H)
            x1 = int(roi_frac[2] * CANONICAL_W); x2 = int(roi_frac[3] * CANONICAL_W)
            cv2.rectangle(warped, (x1, y1), (x2, y2), color, 2)
            cv2.putText(warped, _label_for(key, res), (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"result_{name}.jpg"), warped)

        print(f"[ OK ] {name:44s} Temp={res['temperature']} Mode={res['mode']} "
              f"Flame={res['flame_active']}({res['flame_level']}) "
              f"Heat={res['heating_active']} Water={res['hot_water_active']}"
              + (f"  ! {res['error']}" if res['error'] else ""))

    note = "" if bv.HAVE_SSOCR else "  (ssocr not installed -> Temp=None here)"
    print(f"\nDone. See screen_debug/result_*.jpg{note}")


if __name__ == "__main__":
    main(sys.argv)
