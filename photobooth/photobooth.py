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
        self.state_engine.change_state("Waiting")
        pass
    
    def process_waiting(self):
        # Logic for "Waiting" state
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
    

    # Main loop
    # ------------------------------------------------------------------------
    def start(self):
        state_actions = {
            "Startup": self.process_startup,
            "Waiting": self.process_waiting,
            "Tracking": self.process_tracking,
            "Processing": self.process_processing,
            "Drawing": self.process_drawing,
            "ResetPending": self.process_reset_pending,
        }
        
        # self.state_engine.client.subscribe("#")
        self.state_engine.client.on_message = self.state_engine.on_message
        
        try:
            while True:
                # Update state engine
                current_state = self.state_engine.get_state()
                action = state_actions.get(current_state, lambda: print(f"Unhandled state: {current_state}"))
                action()  # Execute the function associated with the current state
            
        except KeyboardInterrupt:
            print("\nExiting PhotoBooth due to keyboard interrupt...")
