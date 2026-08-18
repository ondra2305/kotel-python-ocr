# kotel-python-ocr

Reads a gas-boiler LCD from a camera photo and publishes the values to Home
Assistant over MQTT. It locates the screen automatically on every frame and
reads ROIs relative to it, so a bumped or repositioned camera keeps working
without re-calibration.

## Layout

```
boiler_ocr/                  the package
├── screen_detection.py      find the LCD glass, flatten it to a 400x640 canonical image
├── boiler_vision.py         read every feature from screen-relative ROIs; analyze() + health checks
├── mqtt_oneshot.py          download a frame, analyze, publish to MQTT (runs on a timer)
└── calibrator/              Flask ROI editor (edit on the full image, saved screen-relative)
    ├── app.py
    └── templates/calibrator.html
debug/detection_preview.py   draw detection results on the sample images
samples/                     calibration + feed sample images
systemd/                     service / timer unit templates
```

## Running

From the repo root, using the project venv (`.venv`, uv-managed):

```bash
# one detection cycle -> MQTT
.venv/bin/python -m boiler_ocr.mqtt_oneshot

# ROI calibrator web UI -> http://localhost:5001
.venv/bin/python -m boiler_ocr.calibrator.app

# debug: run detection on the sample images -> debug/screen_debug/*.jpg
.venv/bin/python debug/detection_preview.py
```

## Configuration

ROIs and detection parameters have working defaults in `boiler_vision.py`
(`DEFAULT_ROIS`, `DEFAULT_PARAMS`). The calibrator writes overrides to
`boiler_ocr/canonical_roi_config.json` (git-ignored); if present it is loaded on
top of the defaults. ROIs are stored as fractions of the screen.

Temperature OCR uses the external `ssocr` binary; install it on the device that
runs `mqtt_oneshot` (it is not required for icon/level/mode detection).

## Home Assistant

Sensors: temperature, flame (on/off + level), heating, hot water, mode
(winter/summer). When the display can't be read (camera down, lens covered,
screen not found) the measurement sensors go **Unavailable** and a diagnostic
**Detection Status** sensor reports why.

## Deploy

Copy the repo to the Pi, create the venv (`uv sync`), install `ssocr`, then
adjust the paths/user in `systemd/*.service` and enable the timer:

```bash
sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/
sudo systemctl enable --now boilerocr-mqtt-oneshot.timer
sudo systemctl enable --now roi-calibrator.service   # optional, for calibration
```