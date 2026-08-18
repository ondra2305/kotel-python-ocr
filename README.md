# kotel-python-ocr

Reads a gas-boiler LCD from a camera photo and publishes the values to Home
Assistant over MQTT. It locates the screen automatically on every frame and
reads ROIs relative to it, so a bumped or repositioned camera keeps working
without re-calibration.

## How it works

1. Detect the LCD and its four corners on the raw photo:

   ![screen detection](docs/1-detect.jpg)

2. Flatten it to a fixed canonical image and read each region — temperature via
   ssocr, the icons / bar-graph / mode by dark-ink coverage:

   ![readout](docs/2-readout.jpg)

3. Because the ROIs are screen-relative, it still lines up when the camera is
   moved to a new angle (here the display is also in summer mode):

   ![moved camera](docs/3-moved.jpg)

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
scripts/install.sh           Raspberry Pi installer (venv, ssocr, systemd)
```

## Running

Needs Python 3.11+ and the project venv (`.venv`, uv-managed). From the repo root:

```bash
# one detection cycle -> MQTT
.venv/bin/python -m boiler_ocr.mqtt_oneshot

# ROI calibrator web UI -> http://localhost:5001
.venv/bin/python -m boiler_ocr.calibrator.app

# debug: run detection on the sample images -> debug/screen_debug/*.jpg
.venv/bin/python debug/detection_preview.py
```

The two entry points also run as plain scripts (e.g.
`python boiler_ocr/calibrator/app.py`), which is handy from an IDE. Dependency
majors are pinned in `pyproject.toml`.

## Configuration

ROIs and detection parameters have working defaults in `boiler_vision.py`
(`DEFAULT_ROIS`, `DEFAULT_PARAMS`). The calibrator writes overrides to
`boiler_ocr/canonical_roi_config.json`; if present it is loaded on top of the
defaults. ROIs are stored as fractions of the screen.

Temperature OCR uses the external `ssocr` binary; install it on the device that
runs `mqtt_oneshot` (it is not required for icon/level/mode detection).

## Home Assistant

Sensors: temperature, flame (on/off + level), heating, hot water, mode
(winter/summer). When the display can't be read (camera down, lens covered,
screen not found) the measurement sensors go **Unavailable** and a diagnostic
**Detection Status** sensor reports why.

## Deploy

Copy the repo to the Pi and run the installer. It creates the venv, installs
`ssocr`, renders the systemd units with this repo's path and the chosen user,
removes any previous install of these units, and enables the timer and
calibrator:

```bash
sudo scripts/install.sh                       # user = repo owner
sudo scripts/install.sh --user pi             # run services as 'pi'
sudo scripts/install.sh --no-calibrator       # skip the calibrator web UI
```

Check it: `systemctl status boilerocr-mqtt-oneshot.timer` and
`journalctl -u boilerocr-mqtt-oneshot.service -f`.

The `systemd/*.service` / `*.timer` files are templates (defaults point at
`/home/admin/boilerocr` and `User=admin`); the installer substitutes the real
path and user, or you can copy and edit them by hand.

---

Created with Claude Opus.