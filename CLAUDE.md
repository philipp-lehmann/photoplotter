# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A photobooth art installation that captures visitor photos, converts them into abstract line-art SVG portraits using depth-map-based contour extraction, and renders them onto Post-it notes using an AxiDraw NextDraw pen plotter. Runs on Raspberry Pi with a 128×128 LCD display and physical buttons.

## Running the Application

Two processes must run concurrently (each in its own terminal):

```bash
# Terminal 1: LCD display service (own venv, requires sudo for SPI/GPIO)
sudo lcd-env/bin/python lcd/lcd.py

# Terminal 2: Main photobooth logic
source photoplotter-env/bin/activate
python main.py
```

On non-Raspberry Pi systems, the app automatically enters test mode and processes images from `photos/test/`.

**Plot a single SVG file:**
```bash
python plotfile.py /absolute/path/to/file.svg
```

**Trace a single image to SVG (drawing-style test harness, no plotter needed):**
```bash
python parsefile.py photos/test/1.jpg --style oneline --seed 1 --open
```
Styles combine with `+`: `oneline` (one continuous line), `features` (only strokes near the 68-point facial landmarks), `outline` (person silhouette only; with `features` the silhouette stays whole while image strokes are feature-filtered), `shade` (parallel hatching of dark person areas; `--shades` sets the tone count incl. paper white, `--spacing` the line spacing, `--stress` 0-1 scales its randomness from calm to wild, darker tones stack rotated hatch families so cross-hatching emerges at `--shades 3+`), `hair` (brush strokes following the hair flow, simulated where the photo shows no strand texture; `--hair-strokes` sets the count), `landmarks` (a random subset of the 68-point feature lines drawn directly into the SVG), `dynamic_grid`/`poisson_disk` (feature-weighted point-snap: dense at eyes/nose/mouth, thinning outwards; placed after a drawing token it snaps only that token's lines, e.g. `hair+poisson_disk+shade` snaps just the hair strokes — placed first it snaps the whole drawing), e.g. `features+outline+shade+oneline`. Style tokens may be joined with `+` or `-`. `--snap dynamic_grid|poisson_disk` forces the global point-snap; `--radius` tunes the feature-overlap distance; `--seed` makes runs reproducible. Outputs land in `photos/parsefile/`.

The live installation's global style is set via `StateEngine.DRAWING_STYLE` in `photobooth/stateengine.py` (same `+`-combinable tokens; `None` = classic contour tracing), with `FEATURE_RADIUS`, `SHADES`, `HATCH_SPACING` alongside it.

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
2. `imageparser.py` — face detection (dlib 68-point landmarks) → person segmentation (MediaPipe selfie segmenter) → depth estimation (MiDaS/PyTorch) → contour extraction (OpenCV) → SVG generation → `photos/traced/`
3. `plotter.py` — loads SVG, configures pen, plots to Post-it via NextDraw API

### State Machine (11 states)

`Startup` → `Waiting` ↔ `Tracking` (→ `Working` to trace an idle-time portrait into the featured slot) → `Snapping` → `Processing` → `Drawing` → `Redrawing`/`Waiting`/`ResetPending` → `Template`. A separate `Test` state drives the non-Raspberry-Pi test mode.

State logic lives in `photobooth/photobooth.py`; transitions are defined in `photobooth/stateengine.py`.

### Stress Level System

A float (0.0–1.0) computed from the time interval between drawing sessions. Shorter intervals → higher stress. Controls:
- SVG complexity: `min_paths` (20–40), `max_paths` (80–140), `min_contour_area` (10–20)
- Plotter speed: range 40–100

### Photo Grid Layout

Portraits are placed on a 5×3 physical grid: 11 standard single-cell slots plus one 2×2 "featured" slot in the bottom-right, defined in `stateengine.py` (`SLOT_LAYOUT`). The featured slot is excluded from the normal visitor-photo rotation and is instead reprinted on demand (see `KEY2` handling in `photobooth.py`) or during idle-time `Working` cycles. Photo IDs are shuffled in blocks (`SHUFFLE_BLOCKS`). Grid position, borders, and gutters are configurable in `stateengine.py`.

### Idle-Time Featured Collage

When the booth has been idle for a while (`Tracking` sees 20+ consecutive faceless frames), it enters the `Working` state, which traces one photo from `photos/work/` and plots it into the featured slot (`process_working` in `photobooth.py`). Drop curated portrait photos (`.jpg`/`.jpeg`/`.png`) into `photos/work/` — only photos with **exactly one detected face** are eligible; others are skipped. Eligible photos are drawn from a shuffled queue that cycles through the whole folder before repeating, and is rescanned (picking up newly added photos) once exhausted. Because each cycle plots onto the same physical featured slot, repeated idle triggers layer additional linework there over time, building up a composite artwork. If `photos/work/` is missing, empty, or has no eligible photos, the state is skipped and the booth returns to `Tracking`.

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
- The MediaPipe selfie segmenter model (`selfie_segmenter.tflite`) is auto-downloaded into `photobooth/models/` on first run if missing
