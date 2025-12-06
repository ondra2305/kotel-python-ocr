#!/usr/bin/env python3
"""
Boiler Display ROI Calibration Tool
Simple Bootstrap UI with camera feed support
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import json
import os
import requests
from datetime import datetime

app = Flask(__name__)

# Default ROI values
DEFAULT_CONFIG = {
    'ROI_FLAME_ICON': [350, 530, 500, 540],
    'ROI_BAR_GRAPH': [350, 530, 540, 580],
    'ROI_HEATING_ICON': [370, 440, 650, 730],
    'ROI_HOT_WATER_ICON': [360, 440, 590, 650],
    'ROI_TEMPERATURE': [140, 260, 520, 650],
}

CONFIG_FILE = 'roi_config.json'
CAMERA_URL = 'http://127.0.0.1:5000/photo_feed'
CURRENT_IMAGE = None
CURRENT_IMAGE_PATH = None

def load_config():
    """Load ROI configuration from file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
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
        'ROI_FLAME_ICON': (0, 255, 0),      # Green
        'ROI_BAR_GRAPH': (255, 0, 0),       # Blue
        'ROI_HEATING_ICON': (0, 165, 255),  # Orange
        'ROI_HOT_WATER_ICON': (0, 255, 255),# Yellow
        'ROI_TEMPERATURE': (255, 0, 255),   # Magenta
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

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = labels.get(name, name)
            cv2.putText(img_copy, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_copy

@app.route('/')
def index():
    """Main calibration page"""
    return render_template('calibrator.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload and load image for calibration"""
    global CURRENT_IMAGE, CURRENT_IMAGE_PATH

    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'calibration_image_{timestamp}.jpg'
    file.save(filename)
    CURRENT_IMAGE_PATH = filename

    # Load image
    CURRENT_IMAGE = cv2.imread(filename)

    if CURRENT_IMAGE is None:
        return jsonify({'error': 'Failed to load image'}), 400

    h, w = CURRENT_IMAGE.shape[:2]

    return jsonify({
        'success': True,
        'filename': filename,
        'width': w,
        'height': h
    })

@app.route('/load_camera', methods=['POST'])
def load_camera():
    """Load image from camera feed"""
    global CURRENT_IMAGE, CURRENT_IMAGE_PATH

    try:
        response = requests.get(CAMERA_URL, timeout=10)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        CURRENT_IMAGE = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if CURRENT_IMAGE is None:
            return jsonify({'error': 'Failed to decode camera image'}), 400

        # Save for reference
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'camera_snapshot_{timestamp}.jpg'
        cv2.imwrite(filename, CURRENT_IMAGE)
        CURRENT_IMAGE_PATH = filename

        h, w = CURRENT_IMAGE.shape[:2]

        return jsonify({
            'success': True,
            'filename': filename,
            'width': w,
            'height': h
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/raw_image/<filename>')
def raw_image(filename):
    """Serve the raw image without ROI boxes"""
    if os.path.exists(filename):
        return send_file(filename, mimetype='image/jpeg')
    return jsonify({'error': 'Image not found'}), 404

@app.route('/config', methods=['GET'])
def get_config():
    """Get current ROI configuration"""
    config = load_config()
    return jsonify(config)

@app.route('/config', methods=['POST'])
def update_config():
    """Update ROI configuration"""
    config = request.json
    save_config(config)
    return jsonify({'success': True})

@app.route('/preview')
def preview():
    """Generate preview image with ROI boxes"""
    global CURRENT_IMAGE

    if CURRENT_IMAGE is None:
        return jsonify({'error': 'No image loaded'}), 400

    config = load_config()

    # Draw ROIs on image
    preview_img = draw_rois(CURRENT_IMAGE, config)

    # Save to temp file
    preview_path = 'preview.jpg'
    cv2.imwrite(preview_path, preview_img)

    return send_file(preview_path, mimetype='image/jpeg')

@app.route('/export')
def export_config():
    """Export configuration as Python code"""
    config = load_config()

    python_code = "# ROI Configuration\n"
    python_code += "# Generated by calibration tool\n\n"

    for name, roi in config.items():
        if name.startswith('ROI_'):
            python_code += f"{name} = {roi}\n"

    return jsonify({
        'python': python_code,
        'json': json.dumps(config, indent=2)
    })

if __name__ == '__main__':
    print("="*70)
    print("BOILER ROI CALIBRATION TOOL")
    print("="*70)
    print("\nOpen browser: http://localhost:5001")
    print("Press Ctrl+C to stop")
    print("="*70)

    app.run(host='0.0.0.0', port=5001, debug=True)
