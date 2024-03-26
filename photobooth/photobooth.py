from .stateengine import StateEngine
from .camera import Camera
from .plotter import Plotter
from .imageparser import ImageParser
import time
import os


class PhotoBooth:
    def __init__(self):
        print("Starting PhotoBooth ...")
        self.state_engine = StateEngine()
        self.camera = Camera()
        self.plotter = Plotter()
        self.image_parser = ImageParser()

    # Handling states
    # ------------------------------------------------------------------------
    def process_startup(self):
        # Logic for "Startup" state
        self.state_engine.change_state("Waiting")
        pass
    
    def process_Waiting(self):
        # Logic for "Waiting" state
        image_path = self.camera.snap_image(output_dir="photos/current", filename="temp")
        if image_path:
            print(f"Waiting: Photo snapped {image_path}")
            # Check if face detected
            if self.image_parser.detect_faces(image_path):
                # If face detected set state to "Tracking"
                self.state_engine.change_state("Tracking")
            else:
                print("No faces detected, staying in Waiting state.")
        else:
            print("Failed to snap photo.")
        time.sleep(3)
    
    def process_tracking(self):
        # Logic for "Tracking" state
        image_path = self.camera.snap_image()
        if image_path:
            print(f"Tracking: Photo snapped and saved at {image_path}")
            self.state_engine.currentPhotoPath = image_path
            self.state_engine.change_state("Processing")
        else:
            print("Failed to snap photo.")
        time.sleep(3)
        pass

    def process_processing(self):
        print(f"Processing: Converting photo to SVG")
        self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath)
        self.state_engine.change_state("Drawing")
        # Logic for "Drawing" state
        pass
    
    def process_drawing(self):
        print(f"Drawing: Connecting with penplotter {self.state_engine.currentPhotoPath}")
        self.state_engine.change_state("Waiting")
        time.sleep(10)
        # Logic for "Drawing" state
        pass

    def process_reset_pending(self):
        # Logic for "ResetPending" state
        pass
    
    # Listen
    def listen_for_keys(self):
        key_input_lines = []
        try:
            # Open the FIFO in non-blocking mode
            fd = os.open(self.state_engine.key_fifo_path, os.O_RDONLY | os.O_NONBLOCK)
            while True:
                # Try to read a chunk of data
                data = os.read(fd, 4096)  # Read up to 4096 bytes
                if not data:
                    break  # No more data available
                # Decode bytes to string and split into lines
                lines = data.decode('utf-8').strip().split('\n')
                key_input_lines.extend(lines)
            os.close(fd)
        except FileNotFoundError:
            print(f"Error: Key FIFO at {self.state_engine.key_fifo_path} not found.")
        except Exception as e:
            print(f"Error reading from key FIFO: {e}")
        
        return key_input_lines

    # Main loop
    # ------------------------------------------------------------------------
    def start(self):
        state_actions = {
            "Startup": self.process_startup,
            "Waiting": self.process_Waiting,
            "Tracking": self.process_tracking,
            "Processing": self.process_processing,
            "Drawing": self.process_drawing,
            "ResetPending": self.process_reset_pending,
        }

        try:
            while True:
                # Update state engine
                current_state = self.state_engine.get_state()
                action = state_actions.get(current_state, lambda: print(f"Unhandled state: {current_state}"))
                action()  # Execute the function associated with the current state
                
                # Listen for key inputs
                key_inputs = self.listen_for_keys()
                for key_input in key_inputs:
                    print(f"Key pressed: {key_input}")
                    # Handle each key input


        except KeyboardInterrupt:
            print("\nExiting PhotoBooth due to keyboard interrupt...")