#!/usr/bin/env python3
"""
ROI calibrator web UI. Edit the ROI boxes on the full camera image; they're
stored as fractions of the detected screen (canonical_roi_config.json), so one
calibration holds at any camera angle. The backend converts pixels <-> fractions
via the detected screen corners.
"""

import json
import os
import sys

import cv2
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, Response

# Work both as a module (python -m boiler_ocr.calibrator.app) and as a plain
# script (python boiler_ocr/calibrator/app.py) by ensuring the repo root is
# importable, then using absolute imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from boiler_ocr import boiler_vision as bv               # noqa: E402
from boiler_ocr.screen_detection import (                # noqa: E402
    detect_screen, warp_to_canonical, screen_matrix, canonical_roi_to_image,
    _map_points, CANONICAL_W, CANONICAL_H)

app = Flask(__name__)

CAMERA_URL = "http://127.0.0.1:5000/photo_feed"
CONFIG_FILE = bv.CONFIG_FILE                     # canonical_roi_config.json

# In-memory state for the current calibration image.
STATE = {"raw": None, "corners": None, "status": None}


# ---------------------------------------------------------------------------
# image-pixel  <->  screen-fraction conversion (uses the detected corners)
# ---------------------------------------------------------------------------

def _frac_to_image_box(frac):
    """A screen-fraction ROI -> its axis-aligned bounding box in image pixels."""
    canon = [frac[0] * CANONICAL_H, frac[1] * CANONICAL_H,
             frac[2] * CANONICAL_W, frac[3] * CANONICAL_W]
    quad = canonical_roi_to_image(canon, STATE["corners"])
    xs, ys = quad[:, 0], quad[:, 1]
    return [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]


def _image_box_to_frac(box):
    """An image-pixel ROI box [y1,y2,x1,x2] -> screen fractions."""
    y1, y2, x1, x2 = box
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    m = _map_points(pts, screen_matrix(STATE["corners"]))  # image -> canonical px
    xs, ys = m[:, 0], m[:, 1]
    return [float(ys.min()) / CANONICAL_H, float(ys.max()) / CANONICAL_H,
            float(xs.min()) / CANONICAL_W, float(xs.max()) / CANONICAL_W]


def _boxes_to_fracs(px_rois):
    return {name: _image_box_to_frac(box) for name, box in px_rois.items()}


# ---------------------------------------------------------------------------
# Image intake
# ---------------------------------------------------------------------------

def _ingest(image):
    STATE["raw"] = image
    if image is None:
        STATE["corners"], STATE["status"] = None, "no_image"
    elif float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()) < \
            bv.DEFAULT_PARAMS["min_frame_brightness"]:
        STATE["corners"], STATE["status"] = None, "too_dark"
    else:
        corners = detect_screen(image)
        STATE["corners"] = corners
        STATE["status"] = "ok" if corners is not None else "no_screen"
    return STATE["status"]


def _dims():
    if STATE["raw"] is None:
        return 0, 0
    h, w = STATE["raw"].shape[:2]
    return w, h


def _corners_list():
    return STATE["corners"].astype(int).tolist() if STATE["corners"] is not None else None


@app.route("/")
def index():
    return render_template("calibrator.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file"}), 400
    arr = np.frombuffer(request.files["image"].read(), np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Failed to decode image"}), 400
    status = _ingest(image)
    w, h = _dims()
    return jsonify({"success": True, "status": status, "width": w, "height": h,
                    "corners": _corners_list()})


@app.route("/load_camera", methods=["POST"])
def load_camera():
    try:
        resp = requests.get(CAMERA_URL, timeout=10)
        resp.raise_for_status()
        arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if image is None:
        return jsonify({"error": "Failed to decode camera image"}), 400
    status = _ingest(image)
    w, h = _dims()
    return jsonify({"success": True, "status": status, "width": w, "height": h,
                    "corners": _corners_list()})


@app.route("/full_image")
def full_image():
    if STATE["raw"] is None:
        return jsonify({"error": "No image loaded"}), 404
    ok, buf = cv2.imencode(".png", STATE["raw"])
    return Response(buf.tobytes(), mimetype="image/png")


# ---------------------------------------------------------------------------
# Config + projection + detection
# ---------------------------------------------------------------------------

@app.route("/config", methods=["GET"])
def get_config():
    rois, params = bv.load_config()
    return jsonify({"rois": rois, "params": params})


@app.route("/config", methods=["POST"])
def save_config():
    """Save fractions directly (used by Import)."""
    data = request.json or {}
    payload = {"rois": data.get("rois", {}), "params": data.get("params", {})}
    with open(CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    return jsonify({"success": True, "path": CONFIG_FILE})


@app.route("/save", methods=["POST"])
def save_from_canvas():
    """Save ROI boxes drawn in image pixels (converted to screen fractions)."""
    if STATE["corners"] is None:
        return jsonify({"error": "No screen detected; cannot save"}), 400
    data = request.json or {}
    payload = {"rois": _boxes_to_fracs(data.get("rois", {})),
               "params": data.get("params", {})}
    with open(CONFIG_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    return jsonify({"success": True, "path": CONFIG_FILE, "rois": payload["rois"]})


@app.route("/project", methods=["POST"])
def project():
    """Project screen-fraction ROIs onto the current image (-> pixel boxes)."""
    if STATE["corners"] is None:
        return jsonify({"error": "No screen detected", "status": STATE["status"]}), 400
    fr = (request.json or {}).get("rois", {})
    return jsonify({"rois": {name: _frac_to_image_box(f) for name, f in fr.items()}})


@app.route("/detect", methods=["POST"])
def detect():
    if STATE["corners"] is None:
        return jsonify({"error": "No screen detected", "status": STATE["status"]}), 400
    data = request.json or {}
    rois, params = bv.load_config()
    rois.update(_boxes_to_fracs(data.get("rois", {})))  # preview edited boxes
    params.update(data.get("params", {}))
    warped, _ = warp_to_canonical(STATE["raw"], STATE["corners"])
    return jsonify(bv.read_canonical(warped, rois, params, debug=True,
                                     full_image=STATE["raw"], corners=STATE["corners"]))


@app.route("/export")
def export_config():
    rois, params = bv.load_config()
    cfg = {"rois": rois, "params": params}
    py = "# canonical_roi_config.json (screen-relative fractions)\n"
    py += "CANONICAL_ROIS = {\n"
    for name, f in rois.items():
        py += f"    {name!r}: {[round(v, 4) for v in f]},\n"
    py += "}\n"
    return jsonify({"json": json.dumps(cfg, indent=2), "python": py})


if __name__ == "__main__":
    print("=" * 60)
    print("BOILER ROI CALIBRATION TOOL (canonical)")
    print(f"Config: {CONFIG_FILE}")
    print("Open: http://localhost:5001")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=False)