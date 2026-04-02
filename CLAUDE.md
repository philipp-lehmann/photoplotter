# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A photobooth art installation that captures visitor photos, converts them into abstract line-art SVG portraits using depth-map-based contour extraction, and renders them onto Post-it notes using an AxiDraw NextDraw pen plotter. Runs on Raspberry Pi with a 128×128 LCD display and physical buttons.

## Running the Application

Two processes must run concurrently (each in its own terminal):

```bash
# Terminal 1: LCD display service (requires sudo for SPI/GPIO)
sudo python lcd/lcd.py

# Terminal 2: Main photobooth logic
source photoplotter-env/bin/activate
python main.py
```

On non-Raspberry Pi systems, the app automatically enters test mode and processes images from `photos/test/`.

**Plot a single SVG file:**
```bash
python plotfile.py /absolute/path/to/file.svg
```

**Systemd services (Raspberry Pi):**
```bash
sudo systemctl start photoplotter_lcd.service
sudo systemctl start photoplotter_main.service
```

## Architecture

The system uses an event-driven state machine with two processes communicating via MQTT (localhost).

### Process Communication

- `lcd/lcd.py` — runs as a separate process; reads hardware buttons and publishes to `lcd/buttons` MQTT topic; subscribes to `state_engine/*` for display updates
- `photobooth/stateengine.py` — acts as MQTT broker interface; drives state transitions and publishes state changes to LCD

### Image Processing Pipeline

1. `camera.py` — captures JPEG via libcam, auto-crops to square → `photos/snapped/`
2. `imageparser.py` — face detection (dlib 68-point landmarks) → depth estimation (MiDaS/PyTorch) → contour extraction (OpenCV) → SVG generation → `photos/traced/`
3. `plotter.py` — loads SVG, configures pen, plots to Post-it via NextDraw API

### State Machine (11 states)

`Startup` → `Waiting` ↔ `Tracking` → `Snapping` → `Processing` → `Drawing` → `Redrawing`/`Waiting`/`ResetPending` → `Template`

State logic lives in `photobooth/photobooth.py`; transitions are defined in `photobooth/stateengine.py`.

### Stress Level System

A float (0.0–1.0) computed from the time interval between drawing sessions. Shorter intervals → higher stress. Controls:
- SVG complexity: `min_paths` (20–40), `max_paths` (80–140), `min_contour_area` (10–20)
- Plotter speed: range 40–100

### Photo Grid Layout

Portraits are placed in a 5×3 grid (15 positions) tracked by `stateengine.py`. Photo IDs are shuffled in triplets. Grid position, borders, and gutters are configurable in `stateengine.py`.

## Key Files

| File | Role |
|------|------|
| `main.py` | Entry point; instantiates and starts `Photobooth` |
| `photobooth/photobooth.py` | State action orchestrator |
| `photobooth/stateengine.py` | State machine + MQTT server |
| `photobooth/imageparser.py` | Core image→SVG pipeline |
| `photobooth/camera.py` | Camera capture + image processing |
| `photobooth/plotter.py` | NextDraw plotter interface |
| `photobooth/nextdraw_conf.py` | Hardware config (pen positions, speeds) |
| `utils.py` | Raspberry Pi detection, CPU temp monitoring, profiling decorators |
| `lcd/lcd.py` | LCD display + button input handler |

## Hardware & Platform Notes

- `utils.py:is_raspberry_pi()` gates all hardware-specific code paths
- CPU temperature is monitored to throttle operations and prevent overheating
- The NextDraw API is installed from Bantam Tools' private download URL (not on PyPI)
- dlib requires the `shape_predictor_68_face_landmarks.dat` model in `photobooth/shape_predictor/`
- MiDaS model files live in `photobooth/midas/`
