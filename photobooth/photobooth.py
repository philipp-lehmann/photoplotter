from .stateengine import StateEngine
from .camera import Camera
from .plotter import Plotter
from .imageparser import ImageParser
import time


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
        self.state_engine.change_state("Ready")
        pass
    
    def process_ready(self):
        # Logic for "Ready" state
        image_path = self.camera.snap_image(output_dir="photos/current", filename="temp")
        if image_path:
            print(f"Ready: Photo snapped {image_path}")
            # Todo: If face detected set state to "Tracking"
            self.state_engine.change_state("Tracking")
        else:
            print("Failed to snap photo.")
        time.sleep(3)
        pass
    
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
        self.state_engine.change_state("Ready")
        time.sleep(10)
        # Logic for "Drawing" state
        pass

    def process_reset_pending(self):
        # Logic for "ResetPending" state
        pass

    # Main loop
    # ------------------------------------------------------------------------
    def start(self):
        state_actions = {
            "Startup": self.process_startup,
            "Ready": self.process_ready,
            "Tracking": self.process_tracking,
            "Processing": self.process_processing,
            "Drawing": self.process_drawing,
            "ResetPending": self.process_reset_pending,
            # Add more states and their corresponding methods as needed
        }

        try:
            while True:
                current_state = self.state_engine.get_state()
                action = state_actions.get(current_state, lambda: print(f"Unhandled state: {current_state}"))
                action()  # Execute the function associated with the current state

        except KeyboardInterrupt:
            print("\nExiting PhotoBooth due to keyboard interrupt...")