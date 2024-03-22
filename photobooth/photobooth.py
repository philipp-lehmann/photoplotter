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
        image_path = self.camera.capture_image()
        if image_path:
            print(f"Photo captured and saved at {image_path}")
            self.state_engine.change_state("Processing")
            # Processing logic here...
            self.state_engine.change_state("Drawing")
        else:
            print("Failed to capture photo.")
        time.sleep(2)  # Adjust based on your capture frequency needs
        pass

    def process_drawing(self):
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