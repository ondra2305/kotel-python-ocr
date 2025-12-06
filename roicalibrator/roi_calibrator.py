#!/usr/bin/env python3
"""
Boiler Display ROI Calibration Tool
With live detection preview and threshold saving
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import json
import os
import requests
from datetime import datetime
import subprocess
import tempfile

app = Flask(__name__)

# Default ROI values
DEFAULT_CONFIG = {
    'ROI_FLAME_ICON': [350, 530, 500, 540],
    'ROI_BAR_GRAPH': [350, 530, 540, 580],
    'ROI_HEATING_ICON': [370, 440, 650, 730],
    'ROI_HOT_WATER_ICON': [360, 440, 590, 650],
    'ROI_TEMPERATURE': [140, 260, 520, 650],
    'THRESHOLDS': {
        'ICON_THRESHOLD': 100,
        'FLAME_ICON_RATIO': 0.08,
        'HEATING_ICON_RATIO': 0.05,
        'HOT_WATER_ICON_RATIO': 0.05,
        'TEMP_THRESHOLDS': [80, 100, 60, 90, 110]
    }
}

CONFIG_FILE = 'roi_config.json'
CAMERA_URL = 'http://127.0.0.1:5000/photo_feed'
CURRENT_IMAGE = None
CURRENT_IMAGE_PATH = None

def load_config():
    """Load ROI configuration from file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Ensure THRESHOLDS key exists
            if 'THRESHOLDS' not in config:
                config['THRESHOLDS'] = DEFAULT_CONFIG['THRESHOLDS'].copy()
            return config
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save ROI configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {CONFIG_FILE}")

def draw_rois(image, config):
    """Draw ROI boxes on image with labels"""
    img_copy = image.copy()

    colors = {
        'ROI_FLAME_ICON': (0, 255, 0),
        'ROI_BAR_GRAPH': (255, 0, 0),
        'ROI_HEATING_ICON': (0, 165, 255),
        'ROI_HOT_WATER_ICON': (0, 255, 255),
        'ROI_TEMPERATURE': (255, 0, 255),
    }

    labels = {
        'ROI_FLAME_ICON': 'Flame',
        'ROI_BAR_GRAPH': 'Level',
        'ROI_HEATING_ICON': 'Heat',
        'ROI_HOT_WATER_ICON': 'Water',
        'ROI_TEMPERATURE': 'Temp',
    }

    for name, roi in config.items():
        if name.startswith('ROI_'):
            y1, y2, x1, x2 = roi
            color = colors.get(name, (255, 255, 255))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = labels.get(name, name)
            cv2.putText(img_copy, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_copy

def detect_icon_active(image, roi, threshold=100, min_ratio=0.05):
    """Detect if an icon is active"""
    try:
        y1, y2, x1, x2 = roi
        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None, 0.0

        roi_image = image[y1:y2, x1:x2]
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

        dark_pixels = np.sum(roi_gray < threshold)
        total_pixels = roi_gray.size

        if total_pixels == 0:
            return None, 0.0

        dark_ratio = dark_pixels / total_pixels
        return bool(dark_ratio > min_ratio), dark_ratio
    except:
        return None, 0.0

def detect_flame_level(image, roi, num_segments=6):
    """Detect flame level from bar graph"""
    try:
        y1, y2, x1, x2 = roi
        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

        _, binary = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)

        height = roi_gray.shape[0]
        segment_height = height // num_segments

        if segment_height == 0:
            return None

        active_segments = 0
        for i in range(num_segments):
            seg_start = height - (i + 1) * segment_height
            seg_end = height - i * segment_height
            segment = binary[seg_start:seg_end, :]

            if segment.size == 0:
                break

            filled_ratio = np.sum(segment > 0) / segment.size
            if filled_ratio > 0.3:
                active_segments = i + 1
            else:
                break

        return int(active_segments)
    except:
        return None

def read_temperature_ssocr(image, roi, thresholds=[80, 100, 60, 90, 110]):
    """Read temperature using ssocr with multiple thresholds"""
    try:
        y1, y2, x1, x2 = roi
        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None, []

        temp_roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(temp_roi, cv2.COLOR_BGR2GRAY)

        valid_temps = []
        threshold_results = []

        for thresh in thresholds:
            try:
                _, processed = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, processed)

                    try:
                        result = subprocess.run(
                            ['ssocr', '--number-digits=2', '-d', '2', tmp.name],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        if result.returncode == 0:
                            temp_str = result.stdout.strip()
                            if temp_str.isdigit() and len(temp_str) == 2:
                                temp = int(temp_str)
                                if 30 <= temp <= 90:
                                    valid_temps.append(temp)
                                    threshold_results.append({'thresh': thresh, 'temp': temp, 'success': True})
                                else:
                                    threshold_results.append({'thresh': thresh, 'temp': temp, 'success': False, 'reason': 'out_of_range'})
                            else:
                                threshold_results.append({'thresh': thresh, 'success': False, 'reason': 'invalid_format'})
                        else:
                            threshold_results.append({'thresh': thresh, 'success': False, 'reason': 'ssocr_failed'})
                    except:
                        threshold_results.append({'thresh': thresh, 'success': False, 'reason': 'timeout'})
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
            except:
                threshold_results.append({'thresh': thresh, 'success': False, 'reason': 'processing_error'})

        result = None
        if valid_temps:
            most_common = max(set(valid_temps), key=valid_temps.count)
            count = valid_temps.count(most_common)
            if count >= 2 or len(valid_temps) == 1:
                result = most_common

        return result, threshold_results
    except:
        return None, []

@app.route('/')
def index():
    return render_template('calibrator.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    global CURRENT_IMAGE, CURRENT_IMAGE_PATH

    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'calibration_image_{timestamp}.jpg'
    file.save(filename)
    CURRENT_IMAGE_PATH = filename
    CURRENT_IMAGE = cv2.imread(filename)

    if CURRENT_IMAGE is None:
        return jsonify({'error': 'Failed to load image'}), 400

    h, w = CURRENT_IMAGE.shape[:2]
    return jsonify({'success': True, 'filename': filename, 'width': w, 'height': h})

@app.route('/load_camera', methods=['POST'])
def load_camera():
    global CURRENT_IMAGE, CURRENT_IMAGE_PATH

    try:
        response = requests.get(CAMERA_URL, timeout=10)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        CURRENT_IMAGE = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if CURRENT_IMAGE is None:
            return jsonify({'error': 'Failed to decode camera image'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'camera_snapshot_{timestamp}.jpg'
        cv2.imwrite(filename, CURRENT_IMAGE)
        CURRENT_IMAGE_PATH = filename

        h, w = CURRENT_IMAGE.shape[:2]
        return jsonify({'success': True, 'filename': filename, 'width': w, 'height': h})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/raw_image/<filename>')
def raw_image(filename):
    if os.path.exists(filename):
        return send_file(filename, mimetype='image/jpeg')
    return jsonify({'error': 'Image not found'}), 404

@app.route('/config', methods=['GET'])
def get_config():
    config = load_config()
    return jsonify(config)

@app.route('/config', methods=['POST'])
def update_config():
    config = request.json
    save_config(config)
    return jsonify({'success': True})

@app.route('/preview')
def preview():
    global CURRENT_IMAGE
    if CURRENT_IMAGE is None:
        return jsonify({'error': 'No image loaded'}), 400

    config = load_config()
    preview_img = draw_rois(CURRENT_IMAGE, config)
    preview_path = 'preview.jpg'
    cv2.imwrite(preview_path, preview_img)
    return send_file(preview_path, mimetype='image/jpeg')

@app.route('/detect', methods=['POST'])
def detect():
    """Run detection on current image with given config and thresholds"""
    global CURRENT_IMAGE

    if CURRENT_IMAGE is None:
        return jsonify({'error': 'No image loaded'}), 400

    data = request.json
    config = data.get('config', {})
    thresholds = data.get('thresholds', DEFAULT_CONFIG['THRESHOLDS'])

    results = {}

    if 'ROI_FLAME_ICON' in config:
        active, ratio = detect_icon_active(
            CURRENT_IMAGE,
            config['ROI_FLAME_ICON'],
            thresholds.get('ICON_THRESHOLD', 100),
            thresholds.get('FLAME_ICON_RATIO', 0.08)
        )
        results['flame_icon'] = {
            'active': active,
            'dark_ratio': round(ratio, 4),
            'threshold': thresholds.get('FLAME_ICON_RATIO', 0.08)
        }

    if 'ROI_BAR_GRAPH' in config:
        level = detect_flame_level(CURRENT_IMAGE, config['ROI_BAR_GRAPH'])
        results['flame_level'] = {'level': level}

    if 'ROI_HEATING_ICON' in config:
        active, ratio = detect_icon_active(
            CURRENT_IMAGE,
            config['ROI_HEATING_ICON'],
            thresholds.get('ICON_THRESHOLD', 100),
            thresholds.get('HEATING_ICON_RATIO', 0.05)
        )
        results['heating_icon'] = {
            'active': active,
            'dark_ratio': round(ratio, 4),
            'threshold': thresholds.get('HEATING_ICON_RATIO', 0.05)
        }

    if 'ROI_HOT_WATER_ICON' in config:
        active, ratio = detect_icon_active(
            CURRENT_IMAGE,
            config['ROI_HOT_WATER_ICON'],
            thresholds.get('ICON_THRESHOLD', 100),
            thresholds.get('HOT_WATER_ICON_RATIO', 0.05)
        )
        results['hot_water_icon'] = {
            'active': active,
            'dark_ratio': round(ratio, 4),
            'threshold': thresholds.get('HOT_WATER_ICON_RATIO', 0.05)
        }

    if 'ROI_TEMPERATURE' in config:
        temp, threshold_results = read_temperature_ssocr(
            CURRENT_IMAGE,
            config['ROI_TEMPERATURE'],
            thresholds.get('TEMP_THRESHOLDS', [80, 100, 60, 90, 110])
        )
        results['temperature'] = {
            'value': temp,
            'threshold_results': threshold_results
        }

    return jsonify(results)

@app.route('/export')
def export_config():
    config = load_config()
    python_code = "# ROI Configuration\n"
    python_code += "# Generated by calibration tool\n\n"

    for name, roi in config.items():
        if name.startswith('ROI_'):
            python_code += f"{name} = {roi}\n"

    if 'THRESHOLDS' in config:
        python_code += "\n# Detection Thresholds\n"
        thresholds = config['THRESHOLDS']
        python_code += f"ICON_THRESHOLD = {thresholds.get('ICON_THRESHOLD', 100)}\n"
        python_code += f"FLAME_ICON_RATIO = {thresholds.get('FLAME_ICON_RATIO', 0.08)}\n"
        python_code += f"HEATING_ICON_RATIO = {thresholds.get('HEATING_ICON_RATIO', 0.05)}\n"
        python_code += f"HOT_WATER_ICON_RATIO = {thresholds.get('HOT_WATER_ICON_RATIO', 0.05)}\n"
        python_code += f"TEMP_THRESHOLDS = {thresholds.get('TEMP_THRESHOLDS', [80, 100, 60, 90, 110])}\n"

    return jsonify({'python': python_code, 'json': json.dumps(config, indent=2)})

if __name__ == '__main__':
    print("="*70)
    print("BOILER ROI CALIBRATION TOOL")
    print("="*70)
    print("\nOpen browser: http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("="*70)
    app.run(host='0.0.0.0', port=5001, debug=False)
