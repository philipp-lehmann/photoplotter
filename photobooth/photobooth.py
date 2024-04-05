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
        self.state_engine.change_state("Test")
        pass
    
    def process_test(self):
        # Logic for "Waiting" state
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_engine.currentPhotoPath = os.path.join(parent_dir, f"photos/snapped/test.jpg")
        self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath)
        
        # Calc test positions
        for id in range(15): 
            startX, startY = self.state_engine.get_image_params_by_id(id)
            print(f"ID {id}: Position X {startX} / {startY}")
            self.image_parser.create_output_svg(self.state_engine.currentSVGPath, 1.0, startX, startY, id)
        
        # Create test image
        print(f"Positioning SVG: {self.state_engine.currentSVGPath}")
        #self.state_engine.currentSVGPath = self.image_parser.create_output_svg(self.state_engine.currentSVGPath, 1.0, 300, 300, 1)
        
        time.sleep(1)
        # self.state_engine.change_state("Drawing")
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
        # Logic for "Drawing" state
        self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath)
        print(f"Converted to SVG: {self.state_engine.currentSVGPath}")
        self.state_engine.change_state("Drawing")
        pass
    
    def process_drawing(self):
        print(f"Drawing: Connecting with penplotter {self.state_engine.currentSVGPath}")
        self.state_engine.update_image_id()
        self.plotter.plot_image(self.state_engine.currentSVGPath)
        #self.plotter.plot_image(self.state_engine.currentSVGPath, self.state_engine.photoID, self.state_engine.imagesPerRow, 15)
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
            "Test": self.process_test,
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
