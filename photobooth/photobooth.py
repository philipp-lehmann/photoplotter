from photobooth.stateengine import StateEngine
from photobooth.camera import Camera
from photobooth.plotter import Plotter
from photobooth.imageparser import ImageParser
import time
import os
import sys
import random

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
        time.sleep(1)
        self.state_engine.change_state("Tracking")
        pass
    
    def process_tracking(self):
        # Logic for "Tracking" state
        random_delay = random.randint(1, 3)
        time.sleep(random_delay)
        image_path = self.camera.snap_image()
        if image_path:
            print(f"Tracking: Photo snapped and saved at {image_path}")
            self.state_engine.currentPhotoPath = image_path
            
            if self.image_parser.detect_faces(self.state_engine.currentPhotoPath):
                os.remove(self.state_engine.currentPhotoPath)  
                self.state_engine.change_state("Snapping")
            else:
                os.remove(self.state_engine.currentPhotoPath)                
                self.state_engine.workID += 1
                
                if self.state_engine.workID < 2:
                    self.state_engine.change_state("Working")
                elif self.state_engine.workID > 15:
                    print("Working skipped: Creating fake portrait")
                    random_fakephoto_number = random.randint(1, 50)
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    self.state_engine.currentPhotoPath = os.path.join(parent_dir, f"assets/fake/fake-{random_fakephoto_number}.jpg")
                    self.state_engine.change_state("Processing")
                    self.state_engine.reset_work_id()
                else: 
                    print(f"Working skipped: {self.state_engine.workID}")
                    time.sleep(random.randint(1,2))
                    self.state_engine.change_state("Tracking")
        else:
            print("Failed to snap photo.")
        pass
    
    def process_snapping(self):
        # Logic for "Snapping" state
        random_delay = random.randint(1, 3)
        time.sleep(random_delay)
        image_path = self.camera.snap_image()
        
        if image_path:
            print(f"Snapping: Photo snapped and saved at {image_path}")
            self.state_engine.currentPhotoPath = image_path
            self.state_engine.change_state("Processing")
        else:
            print("Failed to snap photo.")
            self.state_engine.change_state("Tracking")
            pass
    
    def process_working(self):        
        print(f"Working started: {self.state_engine.workID}")
        # Logic to retrieve work pattern and create output SVG
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        random_svg_number = random.randint(1, 36)
        self.state_engine.currentWorkPath = os.path.join(parent_dir, f"assets/work/work-{random_svg_number}.svg")
            
        # Randomly pick one photo ID from the remaining list without removing it
        random_photo_id = random.choice(self.state_engine.photoID)
        startX, startY = self.state_engine.get_image_params_by_id(random_photo_id - 1)
        
        # Create output SVG using the randomly chosen photo ID
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            self.state_engine.currentWorkPath, "work-output-", 0.35, startX, startY, random_photo_id
        )
        
        print(f"Converted Work pattern to SVG: {self.state_engine.currentSVGPath}, random: {random_photo_id}, from {self.state_engine.photoID}")
        self.plotter.plot_image(self.state_engine.currentSVGPath)
       
        # Change state to Tracking after the work is done
        self.state_engine.change_state("Tracking")
        pass

    def process_processing(self):
        # Logic for "Drawing" state
        if not self.plotter.connect_to_plotter:
            time.sleep(2)
        
        # Convert image to SVG
        tempSVG = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath)
        
        # Check if the SVG file was generated
        if not tempSVG or not os.path.isfile(tempSVG):
            print("Error: SVG file was not created successfully.")
            self.state_engine.change_state("Waiting")
            return
        
        # Get the starting coordinates
        startX, startY = self.state_engine.get_image_params_by_id(self.state_engine.photoID[-1] - 1)

        # Create the final output SVG file
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            tempSVG, "photo-output-", 0.35, startX, startY, self.state_engine.photoID[-1] - 1
        )
        
        # Check if the output SVG was created successfully
        if not self.state_engine.currentSVGPath or not os.path.isfile(self.state_engine.currentSVGPath):
            print("Error: Output SVG file was not created successfully.")
            return

        # Output success message and proceed to the next state
        print(f"Converted to SVG: {self.state_engine.currentSVGPath}")
        self.state_engine.change_state("Drawing")

    def process_drawing(self):
        if self.plotter.connect_to_plotter == False: 
            time.sleep(2)
        print(f"Drawing: Connecting with penplotter {self.state_engine.currentSVGPath}")
        self.plotter.plot_image(self.state_engine.currentSVGPath)
        self.state_engine.update_photo_id()

        # Check if all spots for images have been drawn
        if self.state_engine.photoID:
            self.state_engine.change_state("Waiting")
        else:
            self.state_engine.change_state("ResetPending")
            print(f"All photos printed, changing state to 'ResetPending'.")
        

    def process_reset_pending(self):
        # Logic for "ResetPending" state
        pass
    
    def process_test(self):
        # Base directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        photos_dir = os.path.join(parent_dir, "photos/test")
        
        # Find all .jpg files in the directory
        jpg_files = [f for f in os.listdir(photos_dir) if f.endswith('.jpg') and not f.endswith('_optimized.jpg')]
        
        # Initialize the rolling ID
        # Initialize the array of IDs
        id_array = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
        id_index = 0  # Index to track the current position in the array

        # Process each .jpg file
        for jpg_file in jpg_files:
            self.state_engine.currentPhotoPath = os.path.join(photos_dir, jpg_file)

            # Convert to SVG with various methods
            # self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(
            #     self.state_engine.currentPhotoPath, scale_x=1.0, scale_y=1.0, min_contour_area=5, suffix='-edge', method=1)
            # self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(
            #     self.state_engine.currentPhotoPath, scale_x=1.0, scale_y=1.0, min_contour_area=5, suffix='-binary', method=2)
            self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(
                self.state_engine.currentPhotoPath, scale_x=1.0, scale_y=1.0, min_contour_area=5, suffix='-both', method=3)
            
            # Create the final output SVG file using the rolling ID
            current_id = id_array[id_index]
            startX, startY = self.state_engine.get_image_params_by_id(current_id)
            self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                self.state_engine.currentSVGPath, "photo-output-", 0.35, startX, startY, current_id
            )

            # Update rolling ID, ensuring it wraps between 0 and 15
            id_index = (id_index + 1) % len(id_array)


        output_directory = os.path.join(parent_dir, "photos/current")
        combined_file_path = os.path.join(parent_dir, "photos/output/photo-collection.svg")
        self.image_parser.collect_all_paths(output_directory, combined_file_path)
                    
        print("All JPG files processed.")
        sys.exit()
        pass
    

    # Main loop
    # ------------------------------------------------------------------------
    def start(self):
        state_actions = {
            "Startup": self.process_startup,
            "Waiting": self.process_waiting,
            "Working": self.process_working,
            "Tracking": self.process_tracking,
            "Snapping": self.process_snapping,
            "Processing": self.process_processing,
            "Drawing": self.process_drawing,
            "ResetPending": self.process_reset_pending,
            "Test": self.process_test,
        }
        
        # self.state_engine.client.subscribe("#")
        # self.state_engine.client.on_message = self.state_engine.on_message
        
        try:
            while True:
                # Update state engine
                current_state = self.state_engine.get_state()
                action = state_actions.get(current_state, lambda: print(f"Unhandled state: {current_state}"))
                action()  # Execute the function associated with the current state
            
        except KeyboardInterrupt:
            print("\nExiting PhotoBooth due to keyboard interrupt...")
