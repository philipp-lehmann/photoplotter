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
        time.sleep(4)
        if self.plotter.plotter_found:
            self.state_engine.change_state("ResetPending")
            print("Waiting for reset")
        else:
            self.state_engine.change_state("Waiting")
            
        pass
    
    def process_waiting(self):
        # Logic for "Waiting" state
        time.sleep(4)
        self.state_engine.change_state("Working")
        pass
    
    def process_working(self):
        # Logic to retrieve work pattern and create output svg
        time.sleep(2)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_engine.currentWorkPath = os.path.join(parent_dir, f"assets/work/work-01.svg")
        startX, startY = self.state_engine.get_image_params_by_id(self.state_engine.photoID-1)
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(self.state_engine.currentWorkPath, "work-", 1.0, startX, startY, self.state_engine.photoID)
        
        print(f"Converted Work pattern to SVG: {self.state_engine.currentSVGPath}")
        self.plotter.plot_image(self.state_engine.currentSVGPath)
        
        self.state_engine.change_state("Tracking")
        pass
    
    def process_tracking(self):
        # Logic for "Tracking" state
        time.sleep(1)
        image_path = self.camera.snap_image()
        if image_path:
            print(f"Tracking: Photo snapped and saved at {image_path}")
            self.state_engine.currentPhotoPath = image_path
            
            if self.image_parser.detect_faces(self.state_engine.currentPhotoPath):
                self.state_engine.change_state("Processing")
            else:
                os.remove(self.state_engine.currentPhotoPath)
                self.state_engine.workID += 1
                self.state_engine.change_state("Working")
        else:
            print("Failed to snap photo.")
        pass

    def process_processing(self):
        # Logic for "Drawing" state
        if self.plotter.connect_to_plotter == False: 
            time.sleep(2)
        tempSVG = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath)
        startX, startY = self.state_engine.get_image_params_by_id(self.state_engine.photoID-1)
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(tempSVG, "image-", 1.0, startX, startY, self.state_engine.photoID)
        print(f"Converted to SVG: {self.state_engine.currentSVGPath}")
        self.state_engine.change_state("Drawing")
        pass
    
    def process_drawing(self):
        if self.plotter.connect_to_plotter == False: 
            time.sleep(2)
        print(f"Drawing: Connecting with penplotter {self.state_engine.currentSVGPath}")
        self.plotter.plot_image(self.state_engine.currentSVGPath)
        self.state_engine.update_image_id()
        
        # Check if all available spots for images have been drawn
        if self.state_engine.photoID > 0:
            self.state_engine.change_state("Waiting")
        else:
            self.state_engine.change_state("ResetPending") 
            print(f"All photos printed, changing state to 'ResetPending'.")
        

    def process_reset_pending(self):
        # Logic for "ResetPending" state
        pass
    
    def process_test(self):
        # Logic for "Waiting" state
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_engine.currentPhotoPath = os.path.join(parent_dir, f"photos/snapped/image_20240906_214205.jpg")
        self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath, min_contour_area=5, suffix='-5')
        time.sleep(10)
        pass
    

    # Main loop
    # ------------------------------------------------------------------------
    def start(self):
        state_actions = {
            "Startup": self.process_startup,
            "Waiting": self.process_waiting,
            "Working": self.process_working,
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
