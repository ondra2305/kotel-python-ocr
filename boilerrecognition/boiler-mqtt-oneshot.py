#!/usr/bin/env python3
"""
Boiler Display Recognition with MQTT
Version: 2.5 - ROI Calibrator config loading
"""

import cv2
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import subprocess
import tempfile
import os
import requests
from datetime import datetime
import sys

CONFIG_FILE = '../roicalibrator/roi_config.json'

# Default values (used as fallback)
DEFAULT_ROI_FLAME_ICON = [350, 530, 500, 540]
DEFAULT_ROI_BAR_GRAPH = [350, 530, 540, 580]
DEFAULT_ROI_HEATING_ICON = [370, 440, 650, 730]
DEFAULT_ROI_HOT_WATER_ICON = [360, 440, 590, 650]
DEFAULT_ROI_TEMPERATURE = [140, 260, 520, 650]

DEFAULT_ICON_THRESHOLD = 100
DEFAULT_FLAME_ICON_RATIO = 0.08
DEFAULT_HEATING_ICON_RATIO = 0.05
DEFAULT_HOT_WATER_ICON_RATIO = 0.05
DEFAULT_TEMP_THRESHOLDS = [80, 100, 60, 90, 110]

def load_config():
    """Load ROI and threshold configuration"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Error loading config file: {e}")
            return None
    return None

# Load configuration
config = load_config()

if config:
    print("✓ Loaded configuration from roi_config.json")
    # Load ROIs
    ROI_FLAME_ICON = config.get('ROI_FLAME_ICON', DEFAULT_ROI_FLAME_ICON)
    ROI_BAR_GRAPH = config.get('ROI_BAR_GRAPH', DEFAULT_ROI_BAR_GRAPH)
    ROI_HEATING_ICON = config.get('ROI_HEATING_ICON', DEFAULT_ROI_HEATING_ICON)
    ROI_HOT_WATER_ICON = config.get('ROI_HOT_WATER_ICON', DEFAULT_ROI_HOT_WATER_ICON)
    ROI_TEMPERATURE = config.get('ROI_TEMPERATURE', DEFAULT_ROI_TEMPERATURE)

    # Load thresholds
    thresholds = config.get('THRESHOLDS', {})
    ICON_THRESHOLD = thresholds.get('ICON_THRESHOLD', DEFAULT_ICON_THRESHOLD)
    FLAME_ICON_RATIO = thresholds.get('FLAME_ICON_RATIO', DEFAULT_FLAME_ICON_RATIO)
    HEATING_ICON_RATIO = thresholds.get('HEATING_ICON_RATIO', DEFAULT_HEATING_ICON_RATIO)
    HOT_WATER_ICON_RATIO = thresholds.get('HOT_WATER_ICON_RATIO', DEFAULT_HOT_WATER_ICON_RATIO)
    TEMP_THRESHOLDS = thresholds.get('TEMP_THRESHOLDS', DEFAULT_TEMP_THRESHOLDS)
else:
    # Use defaults if no config file or error loading
    print("⚠ Using default configuration")
    ROI_FLAME_ICON = DEFAULT_ROI_FLAME_ICON
    ROI_BAR_GRAPH = DEFAULT_ROI_BAR_GRAPH
    ROI_HEATING_ICON = DEFAULT_ROI_HEATING_ICON
    ROI_HOT_WATER_ICON = DEFAULT_ROI_HOT_WATER_ICON
    ROI_TEMPERATURE = DEFAULT_ROI_TEMPERATURE

    ICON_THRESHOLD = DEFAULT_ICON_THRESHOLD
    FLAME_ICON_RATIO = DEFAULT_FLAME_ICON_RATIO
    HEATING_ICON_RATIO = DEFAULT_HEATING_ICON_RATIO
    HOT_WATER_ICON_RATIO = DEFAULT_HOT_WATER_ICON_RATIO
    TEMP_THRESHOLDS = DEFAULT_TEMP_THRESHOLDS

# ============= MAIN CONFIGURATION =============
MQTT_BROKER = "192.168.0.13"
MQTT_PORT = 1883
MQTT_USER = "pi"
MQTT_PASS = "mydlinka"

# Camera feed configuration
CAMERA_URL = "http://127.0.0.1:5000/photo_feed"
CAMERA_TIMEOUT = 10  # seconds

# Validation ranges
TEMP_MIN = 30
TEMP_MAX = 90
FLAME_LEVEL_MAX = 6

BAR_SEGMENTS = 6
BAR_FILL_THRESHOLD = 0.3

# MQTT topics
DISCOVERY_PREFIX = "homeassistant"
TOPIC_PREFIX = "home/zero2w/boiler"
DEVICE_ID = "zero2w_boiler_display"
DISCOVERY_STATE_FILE = "/tmp/boiler_mqtt_discovery_sent"

# ============= IMAGE ACQUISITION =============

def download_image(url, timeout=CAMERA_TIMEOUT):
    """Download image from camera feed"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None or image.shape[0] < 100 or image.shape[1] < 100:
            return None

        return image

    except Exception as e:
        print(f"ERROR downloading image: {e}")
        return None

# ============= OPTIMIZED TEMPERATURE READING =============

def read_temperature_ssocr_optimized(image, roi, debug=False):
    """
    Read temperature using only BINARY thresholds (proven to work)
    Based on real testing: only binary 60, 80, 100 work consistently
    Adaptive and inverted methods don't work for this display
    """
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        temp_roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(temp_roi, cv2.COLOR_BGR2GRAY)

        # Only binary thresholds work for this display
        # Test results showed: 80 and 100 most reliable, 60 sometimes, 120 rarely
        thresholds = [80, 100, 60, 90, 110]

        valid_temps = []

        for thresh in thresholds:
            try:
                # Binary threshold (NOT inverted - display has orange backlight)
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
                                if TEMP_MIN <= temp <= TEMP_MAX:
                                    valid_temps.append(temp)
                                    if debug:
                                        print(f"  thresh_{thresh}: {temp}°C")

                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass

            except Exception:
                pass

        # Use consensus
        if valid_temps:
            most_common = max(set(valid_temps), key=valid_temps.count)
            count = valid_temps.count(most_common)

            # Require at least 2 thresholds to agree (or only 1 succeeded)
            if count >= 2 or len(valid_temps) == 1:
                if debug:
                    print(f"Temp: {most_common}°C ({count}/{len(valid_temps)} agree)")
                return most_common

        if debug:
            print(f"Temp: Failed (results: {valid_temps})")
        return None

    except Exception as e:
        if debug:
            print(f"Temp ERROR: {e}")
        return None

# ============= ICON DETECTION =============

def detect_icon_active(image, roi, threshold=ICON_THRESHOLD, min_ratio=0.15, debug=False):
    """Detect if an icon is active"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            if debug:
                print(f"  ROI out of bounds")
            return None

        roi_image = image[y1:y2, x1:x2]

        # Convert to grayscale if needed
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

        # Count dark pixels (below threshold)
        dark_pixels = np.sum(roi_gray < threshold)
        total_pixels = roi_gray.size

        if total_pixels == 0:
            if debug:
                print(f"  Empty ROI")
            return None

        dark_ratio = dark_pixels / total_pixels

        if debug:
            print(f"  Dark ratio: {dark_ratio:.3f} (threshold: {min_ratio})")

        return bool(dark_ratio > min_ratio)

    except Exception as e:
        if debug:
            print(f"  ERROR: {e}")
        return None

# ============= BAR GRAPH DETECTION =============

def detect_flame_level(image, roi, num_segments=BAR_SEGMENTS, debug=False):
    """Detect flame level from bar graph"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]

        # Convert to grayscale if needed
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

        # Binary threshold to detect filled segments
        _, binary = cv2.threshold(roi_gray, ICON_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        height = roi_gray.shape[0]
        segment_height = height // num_segments

        if segment_height == 0:
            return None

        active_segments = 0

        # Count from bottom up
        for i in range(num_segments):
            seg_start = height - (i + 1) * segment_height
            seg_end = height - i * segment_height
            segment = binary[seg_start:seg_end, :]

            if segment.size == 0:
                break

            filled_ratio = np.sum(segment > 0) / segment.size

            if filled_ratio > BAR_FILL_THRESHOLD:
                active_segments = i + 1
            else:
                break

        if 0 <= active_segments <= FLAME_LEVEL_MAX:
            return int(active_segments)

        return None

    except Exception:
        return None

# ============= VALIDATION LOGIC =============

def validate_detections(results):
    """Validate detection results for logical consistency"""
    warnings = []
    corrected = results.copy()

    # Rule 1: Heating and Hot Water cannot both be ON
    if results['heating_active'] and results['hot_water_active']:
        corrected['heating_active'] = None
        corrected['hot_water_active'] = None
        warnings.append("Heating+HotWater conflict")

    # Rule 2: Flame level and flame icon consistency (use elif!)
    flame_level = results['flame_level']
    flame_active = results['flame_active']

    if flame_level is not None and flame_active is not None:
        if flame_level > 0 and not flame_active:
            corrected['flame_active'] = True
            warnings.append(f"Level={flame_level} but icon OFF")

        elif flame_active and flame_level == 0:
            corrected['flame_level'] = 1
            warnings.append("Icon ON but level=0")

        elif not flame_active and flame_level > 0:
            corrected['flame_level'] = 0
            warnings.append(f"Icon OFF but level={flame_level}")

    return corrected, warnings

# ============= MAIN PROCESSING =============

def process_boiler_display(image, debug=False):
    """Process entire boiler display"""
    results = {
        'temperature': None,
        'flame_active': None,
        'flame_level': None,
        'heating_active': None,
        'hot_water_active': None,
        'timestamp': datetime.now().isoformat(),
        'error': None
    }

    try:
        if image is None:
            results['error'] = "No image"
            return results

        if debug:
            print("Detecting flame icon...")
        results['flame_active'] = detect_icon_active(image, ROI_FLAME_ICON,
                                                     threshold=ICON_THRESHOLD,
                                                     min_ratio=FLAME_ICON_RATIO,
                                                     debug=debug)

        if debug:
            print("Detecting flame level...")
        results['flame_level'] = detect_flame_level(image, ROI_BAR_GRAPH, debug=debug)

        if debug:
            print("Reading temperature...")
        results['temperature'] = read_temperature_ssocr_optimized(image, ROI_TEMPERATURE,
                                                                  debug=debug)

        if debug:
            print("Detecting heating icon...")
        results['heating_active'] = detect_icon_active(image, ROI_HEATING_ICON,
                                                       threshold=ICON_THRESHOLD,
                                                       min_ratio=HEATING_ICON_RATIO,
                                                       debug=debug)

        if debug:
            print("Detecting hot water icon...")
        results['hot_water_active'] = detect_icon_active(image, ROI_HOT_WATER_ICON,
                                                         threshold=ICON_THRESHOLD,
                                                         min_ratio=HOT_WATER_ICON_RATIO,
                                                         debug=debug)

        # Validate and correct
        results, warnings = validate_detections(results)

        if warnings:
            print(f"Validation: {', '.join(warnings)}")

        valid_count = sum(1 for k, v in results.items()
                         if k not in ['timestamp', 'error'] and v is not None)
        if valid_count == 0:
            results['error'] = "All sensors failed"

    except Exception as e:
        results['error'] = str(e)
        print(f"ERROR: {e}")

    return results

# ============= MQTT DISCOVERY =============

def send_discovery(client):
    """Send Home Assistant MQTT discovery messages (once)"""

    device_info = {
        "identifiers": [DEVICE_ID],
        "name": "Kotel Display",
        "model": "Boiler Display Recognition v2.4",
        "manufacturer": "DIY"
    }

    sensors = [
        {
            "id": "temperature",
            "name": "Kotel Teplota",
            "type": "sensor",
            "unit": "°C",
            "device_class": "temperature",
            "state_class": "measurement",
            "icon": "mdi:thermometer",
            "value_template": "{{ value_json.temperature if value_json.temperature is not none else 'unavailable' }}"
        },
        {
            "id": "flame_active",
            "name": "Kotel Plamen",
            "type": "binary_sensor",
            "device_class": "heat",
            "icon": "mdi:fire",
            "value_template": "{{ 'ON' if value_json.flame_active else 'OFF' }}",
            "payload_on": "ON",
            "payload_off": "OFF"
        },
        {
            "id": "flame_level",
            "name": "Kotel Plamen Úroveň",
            "type": "sensor",
            "unit": "",
            "icon": "mdi:fire",
            "state_class": "measurement",
            "value_template": "{{ value_json.flame_level if value_json.flame_level is not none else 0 }}"
        },
        {
            "id": "heating_active",
            "name": "Kotel Vytápění",
            "type": "binary_sensor",
            "device_class": "heat",
            "icon": "mdi:radiator",
            "value_template": "{{ 'ON' if value_json.heating_active else 'OFF' }}",
            "payload_on": "ON",
            "payload_off": "OFF"
        },
        {
            "id": "hot_water_active",
            "name": "Kotel Teplá Voda",
            "type": "binary_sensor",
            "device_class": "heat",
            "icon": "mdi:water-pump",
            "value_template": "{{ 'ON' if value_json.hot_water_active else 'OFF' }}",
            "payload_on": "ON",
            "payload_off": "OFF"
        }
    ]

    for sensor in sensors:
        sensor_id = sensor['id']
        sensor_type = sensor['type']

        config_topic = f"{DISCOVERY_PREFIX}/{sensor_type}/{DEVICE_ID}_{sensor_id}/config"
        state_topic = f"{TOPIC_PREFIX}/state"
        availability_topic = f"{TOPIC_PREFIX}/status"

        payload = {
            "name": sensor['name'],
            "unique_id": f"{DEVICE_ID}_{sensor_id}",
            "state_topic": state_topic,
            "availability_topic": availability_topic,
            "value_template": sensor['value_template'],
            "icon": sensor['icon'],
            "device": device_info
        }

        if 'unit' in sensor:
            payload['unit_of_measurement'] = sensor['unit']
        if 'device_class' in sensor:
            payload['device_class'] = sensor['device_class']
        if 'state_class' in sensor:
            payload['state_class'] = sensor['state_class']
        if 'payload_on' in sensor:
            payload['payload_on'] = sensor['payload_on']
        if 'payload_off' in sensor:
            payload['payload_off'] = sensor['payload_off']

        client.publish(config_topic, json.dumps(payload), retain=True)

# ============= MAIN =============

def main():
    """Main function - runs once per timer trigger"""
    start_time = time.time()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting...")

    # Setup MQTT
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        time.sleep(0.5)
    except Exception as e:
        print(f"ERROR: Cannot connect to MQTT: {e}")
        return 1

    # Send discovery once per day
    try:
        if not os.path.exists(DISCOVERY_STATE_FILE) or            (time.time() - os.path.getmtime(DISCOVERY_STATE_FILE)) > 86400:
            send_discovery(client)
            with open(DISCOVERY_STATE_FILE, 'w') as f:
                f.write(str(time.time()))
    except Exception:
        pass

    # Download and process image
    image = download_image(CAMERA_URL)

    if image is None:
        error_data = {
            'temperature': None,
            'flame_active': None,
            'flame_level': None,
            'heating_active': None,
            'hot_water_active': None,
            'timestamp': datetime.now().isoformat(),
            'error': 'Camera unavailable'
        }
        client.publish(f"{TOPIC_PREFIX}/state", json.dumps(error_data))
        client.publish(f"{TOPIC_PREFIX}/status", "offline", retain=True)
        print("ERROR: Failed to download image")
        client.loop_stop()
        client.disconnect()
        return 1

    # Process (debug=True to see temp detection details)
    results = process_boiler_display(image, debug=True)

    # Publish
    client.publish(f"{TOPIC_PREFIX}/state", json.dumps(results))
    client.publish(f"{TOPIC_PREFIX}/status", "online", retain=True)

    elapsed = time.time() - start_time

    # Print results
    temp_str = f"{results['temperature']}°C" if results['temperature'] else "N/A"
    flame_str = "ON" if results['flame_active'] else "OFF"
    level_str = f"{results['flame_level']}/6" if results['flame_level'] is not None else "N/A"
    heat_str = "ON" if results['heating_active'] else "OFF"
    water_str = "ON" if results['hot_water_active'] else "OFF"

    print(f"T:{temp_str} F:{flame_str}({level_str}) H:{heat_str} W:{water_str} [{elapsed:.1f}s]")

    client.loop_stop()
    client.disconnect()

    return 0

if __name__ == '__main__':
    sys.exit(main())
