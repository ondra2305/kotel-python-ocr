#!/usr/bin/env python3
"""
Boiler Display Recognition with MQTT
Version: 3.0 - automatic screen registration + health checks

Downloads a photo of the boiler's LCD, locates the screen automatically (so it
survives the camera being bumped or moved), reads every feature, and publishes
to Home Assistant over MQTT. When the display cannot be read (camera down, lens
covered, screen not found) the sensors are marked Unavailable instead of
reporting wrong values, and a diagnostic "Detection Status" sensor says why.

All the vision lives in boiler_vision.py; this file is just I/O + MQTT.
"""

import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import paho.mqtt.client as mqtt

# Work both as a module (python -m boiler_ocr.mqtt_oneshot) and as a plain
# script (python boiler_ocr/mqtt_oneshot.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from boiler_ocr import boiler_vision

# ============= CONFIGURATION =============
MQTT_BROKER = "192.168.0.13"
MQTT_PORT = 1883
MQTT_USER = "pi"
MQTT_PASS = "mydlinka"

CAMERA_URL = "http://127.0.0.1:5000/photo_feed"
CAMERA_TIMEOUT = 10  # seconds

DISCOVERY_PREFIX = "homeassistant"
TOPIC_PREFIX = "home/zero2w/boiler"
DEVICE_ID = "zero2w_boiler_display"
DISCOVERY_STATE_FILE = "/tmp/boiler_mqtt_discovery_sent"

STATE_TOPIC = f"{TOPIC_PREFIX}/state"          # JSON with all readings
AVAILABILITY_TOPIC = f"{TOPIC_PREFIX}/status"  # online / offline
DETECTION_TOPIC = f"{TOPIC_PREFIX}/detection"  # ok / no_image / too_dark / no_screen

# Human-readable text for each detection status.
STATUS_TEXT = {
    "ok": "OK",
    "no_image": "Camera unavailable",
    "too_dark": "Too dark / lens covered",
    "no_screen": "Screen not found",
}


# ============= IMAGE ACQUISITION =============

def download_image(url, timeout=CAMERA_TIMEOUT):
    """Fetch the camera frame; return a BGR image or None."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None or image.shape[0] < 100 or image.shape[1] < 100:
            return None
        return image
    except Exception as e:
        print(f"ERROR downloading image: {e}")
        return None


# ============= MQTT DISCOVERY =============

def send_discovery(client):
    """Publish Home Assistant MQTT discovery for every sensor (retained)."""
    device_info = {
        "identifiers": [DEVICE_ID],
        "name": "Kotel Display",
        "model": "Boiler Display Recognition v3.0",
        "manufacturer": "DIY",
    }

    # Measurement sensors: share the availability topic, so they all show
    # "Unavailable" together when the display cannot be read.
    sensors = [
        {
            "id": "temperature", "name": "Kotel Teplota", "type": "sensor",
            "unit": "°C", "device_class": "temperature",
            "state_class": "measurement", "icon": "mdi:thermometer",
            "value_template": "{{ value_json.temperature if value_json.temperature is not none else 'unavailable' }}",
        },
        {
            "id": "flame_active", "name": "Kotel Plamen", "type": "binary_sensor",
            "device_class": "heat", "icon": "mdi:fire",
            "value_template": "{{ 'ON' if value_json.flame_active else 'OFF' }}",
            "payload_on": "ON", "payload_off": "OFF",
        },
        {
            "id": "flame_level", "name": "Kotel Plamen Úroveň", "type": "sensor",
            "icon": "mdi:fire", "state_class": "measurement",
            "value_template": "{{ value_json.flame_level if value_json.flame_level is not none else 'unavailable' }}",
        },
        {
            "id": "heating_active", "name": "Kotel Vytápění", "type": "binary_sensor",
            "device_class": "heat", "icon": "mdi:radiator",
            "value_template": "{{ 'ON' if value_json.heating_active else 'OFF' }}",
            "payload_on": "ON", "payload_off": "OFF",
        },
        {
            "id": "hot_water_active", "name": "Kotel Teplá Voda", "type": "binary_sensor",
            "device_class": "heat", "icon": "mdi:water-pump",
            "value_template": "{{ 'ON' if value_json.hot_water_active else 'OFF' }}",
            "payload_on": "ON", "payload_off": "OFF",
        },
        {
            "id": "mode", "name": "Kotel Režim", "type": "sensor",
            "icon": "mdi:sun-snowflake-variant",
            "value_template": "{{ value_json.mode if value_json.mode is not none else 'unavailable' }}",
        },
    ]

    for s in sensors:
        config_topic = f"{DISCOVERY_PREFIX}/{s['type']}/{DEVICE_ID}_{s['id']}/config"
        payload = {
            "name": s["name"],
            "unique_id": f"{DEVICE_ID}_{s['id']}",
            "state_topic": STATE_TOPIC,
            "availability_topic": AVAILABILITY_TOPIC,
            "value_template": s["value_template"],
            "icon": s["icon"],
            "device": device_info,
        }
        for k in ("unit", "device_class", "state_class", "payload_on", "payload_off"):
            if k in s:
                payload[{"unit": "unit_of_measurement"}.get(k, k)] = s[k]
        client.publish(config_topic, json.dumps(payload), retain=True)

    # Diagnostic status sensor: its OWN (no) availability so it stays visible
    # to explain WHY the others are unavailable.
    status_cfg = {
        "name": "Kotel Detekce",
        "unique_id": f"{DEVICE_ID}_detection_status",
        "state_topic": DETECTION_TOPIC,
        "icon": "mdi:cctv",
        "entity_category": "diagnostic",
        "device": device_info,
    }
    client.publish(
        f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_detection_status/config",
        json.dumps(status_cfg), retain=True)


# ============= MAIN =============

def main():
    start = time.time()
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Starting...")

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

    # Re-send discovery at most once per day.
    try:
        if (not os.path.exists(DISCOVERY_STATE_FILE)
                or time.time() - os.path.getmtime(DISCOVERY_STATE_FILE) > 86400):
            send_discovery(client)
            with open(DISCOVERY_STATE_FILE, "w") as f:
                f.write(str(time.time()))
    except Exception:
        pass

    # Acquire and analyze.
    image = download_image(CAMERA_URL)
    results = boiler_vision.analyze(image)
    results["timestamp"] = datetime.now().isoformat()
    status = results["status"]
    online = status == "ok"

    # Publish: full state, availability (drives Unavailable in HA), and the
    # always-visible diagnostic status.
    client.publish(STATE_TOPIC, json.dumps(results))
    client.publish(AVAILABILITY_TOPIC, "online" if online else "offline", retain=True)
    client.publish(DETECTION_TOPIC, STATUS_TEXT.get(status, status), retain=True)

    elapsed = time.time() - start
    if online:
        temp = f"{results['temperature']}°C" if results["temperature"] is not None else "N/A"
        lvl = f"{results['flame_level']}/6" if results["flame_level"] is not None else "N/A"
        print(f"T:{temp} F:{'ON' if results['flame_active'] else 'OFF'}({lvl}) "
              f"H:{'ON' if results['heating_active'] else 'OFF'} "
              f"W:{'ON' if results['hot_water_active'] else 'OFF'} "
              f"Mode:{results['mode']} [{elapsed:.1f}s]")
        if results["error"]:
            print(f"Validation: {results['error']}")
    else:
        print(f"UNAVAILABLE: {STATUS_TEXT.get(status, status)} [{elapsed:.1f}s]")

    client.loop_stop()
    client.disconnect()
    return 0 if online else 2


if __name__ == "__main__":
    sys.exit(main())
