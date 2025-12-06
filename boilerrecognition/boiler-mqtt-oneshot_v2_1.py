#!/usr/bin/env python3
"""
Boiler Display Recognition with MQTT
Single-shot version for systemd timer
Version: 2.2 - Oneshot mode for accurate timing via systemd timer
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

# ============= ROI CONFIGURATION =============
ROI_FLAME_ICON = [350, 530, 500, 540]
ROI_BAR_GRAPH = [350, 530, 540, 580]
ROI_HEATING_ICON = [370, 440, 650, 730]
ROI_HOT_WATER_ICON = [360, 440, 590, 650]
ROI_TEMPERATURE = [140, 260, 520, 650]

# ============= DETECTION THRESHOLDS =============
ICON_THRESHOLD = 100
FLAME_ICON_RATIO = 0.08
HEATING_ICON_RATIO = 0.05
HOT_WATER_ICON_RATIO = 0.05
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

# ============= TEMPERATURE READING =============

def read_temperature_ssocr(image, roi, flame_is_on=None, debug=False):
    """Read temperature using ssocr with adaptive preprocessing"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        temp_roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(temp_roi, cv2.COLOR_BGR2GRAY)

        # Determine preprocessing method based on flame state
        if flame_is_on is True:
            methods_to_try = [('inverted', cv2.THRESH_BINARY)]
        elif flame_is_on is False:
            methods_to_try = [('normal', cv2.THRESH_BINARY_INV)]
        else:
            methods_to_try = [('inverted', cv2.THRESH_BINARY), ('normal', cv2.THRESH_BINARY_INV)]

        # Try each preprocessing method
        for method_name, thresh_type in methods_to_try:
            _, processed = cv2.threshold(gray, 100, 255, thresh_type)

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
                                return temp

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass

    except Exception:
        pass

    return None

# ============= ICON DETECTION =============

def detect_icon_active(image, roi, threshold=ICON_THRESHOLD, min_ratio=0.15):
    """Detect if an icon is active"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image

        dark_pixels = np.sum(roi_gray < threshold)
        total_pixels = roi_gray.size

        if total_pixels == 0:
            return None

        dark_ratio = dark_pixels / total_pixels
        return bool(dark_ratio > min_ratio)

    except Exception:
        return None

# ============= BAR GRAPH DETECTION =============

def detect_flame_level(image, roi, num_segments=BAR_SEGMENTS):
    """Detect flame level"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) if len(roi_image.shape) == 3 else roi_image

        _, binary = cv2.threshold(roi_gray, ICON_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

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

    # Rule 2: Flame level and flame icon consistency
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

def process_boiler_display(image):
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

        # Detect flame first (for temperature preprocessing)
        results['flame_active'] = detect_icon_active(image, ROI_FLAME_ICON, 
                                                     threshold=ICON_THRESHOLD, 
                                                     min_ratio=FLAME_ICON_RATIO)
        results['flame_level'] = detect_flame_level(image, ROI_BAR_GRAPH)

        # Temperature with adaptive preprocessing
        results['temperature'] = read_temperature_ssocr(image, ROI_TEMPERATURE, 
                                                        flame_is_on=results['flame_active'])

        # Other icons
        results['heating_active'] = detect_icon_active(image, ROI_HEATING_ICON,
                                                       threshold=ICON_THRESHOLD,
                                                       min_ratio=HEATING_ICON_RATIO)
        results['hot_water_active'] = detect_icon_active(image, ROI_HOT_WATER_ICON,
                                                         threshold=ICON_THRESHOLD,
                                                         min_ratio=HOT_WATER_ICON_RATIO)

        # Validate and correct
        results, warnings = validate_detections(results)

        if warnings:
            print(f"Validation warnings: {', '.join(warnings)}")

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
        "model": "Boiler Display Recognition v2.2",
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

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting capture...")

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
        time.sleep(0.5)  # Brief wait for connection
    except Exception as e:
        print(f"ERROR: Cannot connect to MQTT: {e}")
        return 1

    # Send discovery once per day (check state file)
    try:
        if not os.path.exists(DISCOVERY_STATE_FILE) or            (time.time() - os.path.getmtime(DISCOVERY_STATE_FILE)) > 86400:
            print("Sending discovery messages...")
            send_discovery(client)
            with open(DISCOVERY_STATE_FILE, 'w') as f:
                f.write(str(time.time()))
    except Exception as e:
        print(f"Discovery error: {e}")

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

    # Process
    results = process_boiler_display(image)

    # Publish
    client.publish(f"{TOPIC_PREFIX}/state", json.dumps(results))
    client.publish(f"{TOPIC_PREFIX}/status", "online", retain=True)

    elapsed = time.time() - start_time

    # Print results
    print(f"Temp: {results['temperature']}°C | " +
          f"Flame: {'ON' if results['flame_active'] else 'OFF'} ({results['flame_level']}/6) | " +
          f"Heat: {'ON' if results['heating_active'] else 'OFF'} | " +
          f"Water: {'ON' if results['hot_water_active'] else 'OFF'} | " +
          f"Time: {elapsed:.1f}s")

    client.loop_stop()
    client.disconnect()

    return 0

if __name__ == '__main__':
    sys.exit(main())
