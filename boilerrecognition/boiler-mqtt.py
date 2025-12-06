#!/usr/bin/env python3
"""
Boiler Display Recognition with MQTT
Reads boiler display via camera and publishes to Home Assistant
Version: 2.1 - With adaptive temperature preprocessing based on flame state
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

# ============= MAIN CONFIGURATION =============
MQTT_BROKER = "192.168.0.13"
MQTT_PORT = 1883
MQTT_USER = "pi"
MQTT_PASS = "mydlinka"

# Camera feed configuration
CAMERA_URL = "http://127.0.0.1:5000/photo_feed"
CAMERA_TIMEOUT = 10  # seconds

# Startup delay configuration
STARTUP_DELAY_AFTER_BOOT = 180  # Wait 3 minutes after fresh boot
UPTIME_THRESHOLD = 5  # Consider it a fresh boot if uptime < 5 minutes

# Update intervals
UPDATE_INTERVAL = 60  # seconds - how often to capture and process

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

# ============= DETECTION THRESHOLDS (DATA-DRIVEN) =============
ICON_THRESHOLD = 100  # Pixel darkness threshold (0-255)

# Individual sensitivity for each icon type (% of dark pixels needed to activate)
# These values are based on real-world testing:
FLAME_ICON_RATIO = 0.08      # 8% - Flame ON=14.5%, OFF=5% (outline visible when off!)
HEATING_ICON_RATIO = 0.05    # 5% - Heating ON=26%, OFF=0%
HOT_WATER_ICON_RATIO = 0.05  # 5% - Hot water ON=22%, OFF=0%

BAR_SEGMENTS = 6
BAR_FILL_THRESHOLD = 0.3

# MQTT topics
DISCOVERY_PREFIX = "homeassistant"
TOPIC_PREFIX = "home/zero2w/boiler"
DEVICE_ID = "zero2w_boiler_display"

# ============= STARTUP DELAY =============

def check_startup_delay():
    """
    Wait if system just booted to ensure camera stream is ready
    Only delays if system uptime < UPTIME_THRESHOLD minutes
    Manual restarts will not trigger delay
    """
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.read().split()[0])

        uptime_minutes = uptime_seconds / 60

        print(f"System uptime: {uptime_minutes:.1f} minutes")

        if uptime_minutes < UPTIME_THRESHOLD:
            print(f"Fresh boot detected (uptime < {UPTIME_THRESHOLD} min)")
            print(f"Waiting {STARTUP_DELAY_AFTER_BOOT} seconds for camera stream...")

            # Show countdown every 30 seconds
            remaining = STARTUP_DELAY_AFTER_BOOT
            while remaining > 0:
                if remaining == STARTUP_DELAY_AFTER_BOOT or remaining % 30 == 0:
                    print(f"  Starting in {remaining} seconds...")
                time.sleep(1)
                remaining -= 1

            print("Startup delay complete!")
        else:
            print(f"System already running (uptime > {UPTIME_THRESHOLD} min), starting immediately")

    except Exception as e:
        print(f"Could not check uptime: {e}")
        print("Starting immediately")

# ============= IMAGE ACQUISITION =============

def download_image(url, timeout=CAMERA_TIMEOUT):
    """Download image from camera feed"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print(f"ERROR: Failed to decode image from {url}")
            return None

        if image.shape[0] < 100 or image.shape[1] < 100:
            print(f"ERROR: Image too small: {image.shape}")
            return None

        return image

    except requests.exceptions.Timeout:
        print(f"ERROR: Timeout downloading from {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download image: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error downloading image: {e}")
        return None

# ============= TEMPERATURE READING WITH ADAPTIVE PREPROCESSING =============

def read_temperature_ssocr(image, roi, flame_is_on=None, debug=False):
    """
    Read temperature using ssocr with adaptive preprocessing

    Args:
        image: Input image
        roi: Region of interest for temperature
        flame_is_on: Boolean indicating flame state (affects backlight)
                    True = orange backlight (dark bg, light digits) → inverted
                    False = no backlight (light bg, dark digits) → normal
                    None = try both methods
        debug: Print debug info
    """
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            if debug:
                print(f"  ERROR: ROI out of bounds")
            return None

        temp_roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(temp_roi, cv2.COLOR_BGR2GRAY)

        # Determine preprocessing method based on flame state
        methods_to_try = []

        if flame_is_on is True:
            # Flame ON → Orange backlight → Dark background, light digits
            methods_to_try = [('inverted', cv2.THRESH_BINARY)]
            if debug:
                print(f"  Using inverted preprocessing (flame ON, orange backlight)")
        elif flame_is_on is False:
            # Flame OFF → No backlight → Light background, dark digits
            methods_to_try = [('normal', cv2.THRESH_BINARY_INV)]
            if debug:
                print(f"  Using normal preprocessing (flame OFF, no backlight)")
        else:
            # Unknown state → Try both (inverted first as flame is usually on)
            methods_to_try = [
                ('inverted', cv2.THRESH_BINARY),
                ('normal', cv2.THRESH_BINARY_INV)
            ]
            if debug:
                print(f"  Flame state unknown, trying both preprocessing methods")

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
                                if debug:
                                    print(f"  ✓ Temperature: {temp}°C (method: {method_name})")
                                return temp
                            else:
                                if debug:
                                    print(f"  Temperature {temp}°C out of range with {method_name}")

                    if debug and method_name == methods_to_try[-1][0]:
                        print(f"  ssocr failed with {method_name}: {result.stderr.strip()}")

                except subprocess.TimeoutExpired:
                    if debug:
                        print(f"  ERROR: ssocr timeout with {method_name}")
                except FileNotFoundError:
                    if debug:
                        print(f"  ERROR: ssocr not installed")
                    return None
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass

    except Exception as e:
        if debug:
            print(f"  ERROR: {e}")

    return None

# ============= ICON DETECTION =============

def detect_icon_active(image, roi, threshold=ICON_THRESHOLD, min_ratio=0.15):
    """Detect if an icon is active - returns Python bool"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]

        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

        dark_pixels = np.sum(roi_gray < threshold)
        total_pixels = roi_gray.size

        if total_pixels == 0:
            return None

        dark_ratio = dark_pixels / total_pixels
        # Convert numpy.bool_ to Python bool
        return bool(dark_ratio > min_ratio)

    except Exception as e:
        print(f"ERROR detecting icon: {e}")
        return None

# ============= BAR GRAPH DETECTION =============

def detect_flame_level(image, roi, num_segments=BAR_SEGMENTS):
    """Detect flame level - returns Python int"""
    try:
        y1, y2, x1, x2 = roi

        if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
            return None

        roi_image = image[y1:y2, x1:x2]

        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi_image

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

        # Convert to Python int
        if 0 <= active_segments <= FLAME_LEVEL_MAX:
            return int(active_segments)

        return None

    except Exception as e:
        print(f"ERROR detecting flame level: {e}")
        return None

# ============= VALIDATION LOGIC =============

def validate_detections(results, debug=False):
    """
    Validate detection results for logical consistency
    Returns: (is_valid, corrected_results, warnings)
    """
    warnings = []
    corrected = results.copy()
    is_valid = True

    # Rule 1: Heating and Hot Water cannot both be ON simultaneously
    if results['heating_active'] and results['hot_water_active']:
        warnings.append("CONFLICT: Both heating and hot water detected as ON (mutually exclusive)")
        # Assume false positive, set both to None to be safe
        corrected['heating_active'] = None
        corrected['hot_water_active'] = None
        is_valid = False
        if debug:
            print("  ⚠️  Validation failed: Heating AND hot water both ON")

    # Rule 2: Flame level and flame icon must be consistent
    flame_level = results['flame_level']
    flame_active = results['flame_active']

    if flame_level is not None and flame_active is not None:
        # If flame level > 0, flame icon must be ON
        if flame_level > 0 and not flame_active:
            warnings.append(f"CONFLICT: Flame level is {flame_level} but flame icon is OFF")
            # Trust the flame level (more granular), set icon to ON
            corrected['flame_active'] = True
            is_valid = False
            if debug:
                print(f"  ⚠️  Validation failed: Flame level {flame_level} but icon OFF → Corrected to ON")

        # If flame icon is ON, flame level must be > 0
        elif flame_active and flame_level == 0:
            warnings.append("CONFLICT: Flame icon is ON but flame level is 0")
            # Trust the icon, set minimum level
            corrected['flame_level'] = 1
            is_valid = False
            if debug:
                print(f"  ⚠️  Validation failed: Flame icon ON but level 0 → Corrected to 1")

        # If flame icon is OFF, flame level must be 0
        elif not flame_active and flame_level > 0:
            warnings.append(f"CONFLICT: Flame icon is OFF but flame level is {flame_level}")
            # Trust the icon, clear the level
            corrected['flame_level'] = 0
            is_valid = False
            if debug:
                print(f"  ⚠️  Validation failed: Flame icon OFF but level {flame_level} → Corrected to 0")

    return is_valid, corrected, warnings

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
        'error': None,
        'validation_warnings': []
    }

    try:
        if image is None:
            results['error'] = "No image"
            return results

        # STEP 1: Detect flame icon first (needed for temperature preprocessing)
        if debug:
            print("Detecting flame icon...")
        results['flame_active'] = detect_icon_active(
            image, ROI_FLAME_ICON,
            threshold=ICON_THRESHOLD,
            min_ratio=FLAME_ICON_RATIO
        )

        # STEP 2: Detect flame level
        if debug:
            print("Detecting flame level...")
        results['flame_level'] = detect_flame_level(image, ROI_BAR_GRAPH)

        # STEP 3: Read temperature with adaptive preprocessing based on flame state
        if debug:
            print("Reading temperature...")
        results['temperature'] = read_temperature_ssocr(
            image, ROI_TEMPERATURE,
            flame_is_on=results['flame_active'],
            debug=debug
        )

        # STEP 4: Detect heating icon
        if debug:
            print("Detecting heating icon...")
        results['heating_active'] = detect_icon_active(
            image, ROI_HEATING_ICON,
            threshold=ICON_THRESHOLD,
            min_ratio=HEATING_ICON_RATIO
        )

        # STEP 5: Detect hot water icon
        if debug:
            print("Detecting hot water icon...")
        results['hot_water_active'] = detect_icon_active(
            image, ROI_HOT_WATER_ICON,
            threshold=ICON_THRESHOLD,
            min_ratio=HOT_WATER_ICON_RATIO
        )

        # STEP 6: Validate detection results
        if debug:
            print("Validating detections...")
        is_valid, corrected_results, warnings = validate_detections(results, debug=debug)

        # Apply corrections
        if not is_valid:
            results.update(corrected_results)
            results['validation_warnings'] = warnings

        valid_count = sum(1 for k, v in results.items()
                         if k not in ['timestamp', 'error', 'validation_warnings'] and v is not None)
        if valid_count == 0:
            results['error'] = "All sensors failed"

    except Exception as e:
        results['error'] = str(e)
        print(f"ERROR processing display: {e}")

    return results

# ============= MQTT DISCOVERY =============

def send_discovery(client):
    """Send Home Assistant MQTT discovery messages"""

    device_info = {
        "identifiers": [DEVICE_ID],
        "name": "Kotel Display",
        "model": "Boiler Display Recognition v2.1",
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
        print(f"Discovery sent: {sensor['name']}")

    print("All discovery messages sent")

# ============= MQTT CALLBACKS =============

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback when connected to MQTT broker"""
    if rc == 0:
        print("Connected to MQTT Broker")
        client.publish(f"{TOPIC_PREFIX}/status", "online", retain=True)
        send_discovery(client)
    else:
        print(f"Connection failed, code: {rc}")

# ============= MAIN LOOP =============

def main():
    """Main application loop"""

    print("="*70)
    print("Boiler Display Recognition Service v2.1")
    print("="*70)
    print(f"Camera URL: {CAMERA_URL}")
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Update interval: {UPDATE_INTERVAL}s")
    print(f"Temperature range: {TEMP_MIN}-{TEMP_MAX}°C")
    print(f"Detection thresholds:")
    print(f"  Flame icon:     {FLAME_ICON_RATIO:.0%} (ON=14.5%, OFF=5%)")
    print(f"  Heating icon:   {HEATING_ICON_RATIO:.0%}")
    print(f"  Hot water icon: {HOT_WATER_ICON_RATIO:.0%}")
    print(f"Temperature preprocessing:")
    print(f"  Flame ON  → Inverted (orange backlight)")
    print(f"  Flame OFF → Normal (no backlight)")
    print("="*70)

    # Smart startup delay - only after fresh boot
    check_startup_delay()

    print("\nInitializing MQTT connection...")

    # Setup MQTT client with v1/v2 compatibility
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    if MQTT_USER and MQTT_PASS:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    client.will_set(f"{TOPIC_PREFIX}/status", "offline", retain=True)
    client.on_connect = on_connect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"Cannot connect to MQTT: {e}")
        return

    print("Service started, processing images...")

    consecutive_failures = 0
    max_failures = 5

    while True:
        try:
            start_time = time.time()

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading image...")
            image = download_image(CAMERA_URL)

            if image is None:
                consecutive_failures += 1
                print(f"Failed to download image ({consecutive_failures}/{max_failures})")

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

                if consecutive_failures >= max_failures:
                    print(f"Too many consecutive failures, marking offline")
                    client.publish(f"{TOPIC_PREFIX}/status", "offline", retain=True)

            else:
                print("Processing image...")
                results = process_boiler_display(image, debug=True)

                print("\nResults:")
                print(f"  Temperature: {results['temperature']}°C" if results['temperature'] else "  Temperature: N/A")
                print(f"  Flame: {'ON' if results['flame_active'] else 'OFF'}")
                print(f"  Flame Level: {results['flame_level']}/{FLAME_LEVEL_MAX}")
                print(f"  Heating: {'ON' if results['heating_active'] else 'OFF'}")
                print(f"  Hot Water: {'ON' if results['hot_water_active'] else 'OFF'}")
                if results.get('validation_warnings'):
                    print(f"  ⚠️  Validation warnings:")
                    for warning in results['validation_warnings']:
                        print(f"     {warning}")
                if results['error']:
                    print(f"  Error: {results['error']}")

                # Remove validation_warnings before publishing (internal only)
                publish_data = {k: v for k, v in results.items() if k != 'validation_warnings'}

                client.publish(f"{TOPIC_PREFIX}/state", json.dumps(publish_data))
                client.publish(f"{TOPIC_PREFIX}/status", "online", retain=True)

                print("Published to MQTT")
                consecutive_failures = 0

            elapsed = time.time() - start_time
            sleep_time = max(0, UPDATE_INTERVAL - elapsed)
            if sleep_time > 0:
                print(f"Waiting {sleep_time:.1f}s until next update...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"ERROR in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)

    client.publish(f"{TOPIC_PREFIX}/status", "offline", retain=True)
    client.loop_stop()
    client.disconnect()
    print("Service stopped")

if __name__ == '__main__':
    main()
