# photoplotter
The photoplotter project is a Python-based application designed to create a digital photobooth experience. It consists of several modules organized within a package structure. Here's a general project description:

**Description:**
The photobooth application allows users to take photos, process them, and print them with an axidraw penplotter. It simulates the experience of a traditional photobooth but with added digital features. 

**Classes:**

1. `photobooth.py:` This module contains the main logic for the photobooth application, coordinating interactions between different components.
2. `stateengine.py:` Manages the state of the photobooth, controlling transitions between different states such as idle, capturing, processing, printing and completed.
Should also track the current id of the current portrait drawing. Communicates with broker messages to output the current application state on `lcd.py`
3. `camera.py:` Handles camera functionality, including facetracking, capturing photos and providing them for processing. Saves snapped image temporary and source images for vectorization separately.
4. `imageparser.py:` Processes the snapped images captured by the camera, converting them to traced SVGs for plotting.
5. `plotter.py:` Manages the plotter connection and functionality to draw the portraits, which could be used for printing svg images with a axidraw penplotter.
6. `lcd.py:` Display interface of the photobooth, showing instructions and the current state of the application to users on a 128x128 lcd display. Located in a separate directory and communicates with broker to submit button inputs. Runs as a separate task.

**File Structure:**
The project is structured as follows:

```
photoplotter/
│
├── start.sh
├── main.py
│
├── lcd/
│   └── lcd.py
│
└── photobooth/
│   ├── __init__.py
│   ├── photobooth.py
│   ├── stateengine.py
│   ├── display.py
│   ├── camera.py
│   ├── plotter.py
│   └── imageparser.py
│   
└── photos/
	└── output/
	└── collection/
	└── snapped/
	└── test/
	└── traced/

```

**Functionality:**

- Users interact with the photobooth by following on-screen instructions.
- They can capture photos using the integrated camera.
- The captured photos are processed, saved and plotted.
- When not connected to Pi the app goes into test mode and collects all images in the test folder.

**Goals:**

- Create an intuitive and engaging photobooth experience.
- Ensure smooth transitions between different states of operation.
- Provide options for different target outputs, paper sizes or drawing complexity.

**Ideas:**

- Add reprint functionality
- Style picker logic
- Configuration for output sizes

This project aims to combine the nostalgic charm of traditional photobooths with modern digital technology, offering a fun and interactive experience for users of all ages.

**State-Engine**
- The project uses a state-engine.
![State engine for this branch](doc/state-engine-stadtfest.jpg)


## Startup

```bash
# Startup lcd display first
cd photoplotter
sudo python lcd/lcd.py
```

```bash
# Startup activate venv and run main second
cd photoplotter
source photoplotter-env/bin/activate
python main.py
```

<aside>
☝ Alternatively run `start.sh`
</aside>

## Setup repo

```bash
# Create venv
python -m venv photoplotter-env
source photoplotter-env/bin/activate

# Install dependencies
pip install paho-mqtt==1.5.1
pip install opencv-python
pip install svgwrite==1.4.1
pip install Pillow
pip install lxml
pip install numpy
pip install paho-mqtt
pip install dlib
pip install scipy
pip install torch
pip install torch torchvision
pip install timm

python -m pip install https://software-download.bantamtools.com/nd/api/nextdraw_api.zip
```


## Services
Create .service files to launch on startup
```

# Disable autostart
sudo systemctl stop photoplotter_lcd.service
sudo systemctl stop photoplotter_main.service
```
```
sudo systemctl disable photoplotter_lcd.service
sudo systemctl disable photoplotter_main.service
```
```
# Enable autostart
sudo systemctl start photoplotter_lcd.service
sudo systemctl start photoplotter_main.service
```
```
sudo systemctl enable photoplotter_lcd.service
sudo systemctl enable photoplotter_main.service
```
```
# Check status
sudo systemctl status photoplotter_lcd.service
sudo systemctl status photoplotter_main.service
```
```
# Zip & download with right click in vs code
zip -r photos.zip photos
```

<aside>
⚠️ After reinstallation make sure plotter sizes are correct in `axidraw_conf.py`
</aside>