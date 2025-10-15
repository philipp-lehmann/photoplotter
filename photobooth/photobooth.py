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
        time.sleep(0.5)
        if self.plotter.plotter_found:
            self.state_engine.change_state("ResetPending")
            print("Waiting for reset")
        else:
            self.state_engine.change_state("Waiting")
            print("No plotter found")
        pass
    
    def process_waiting(self):
        # Logic for "Waiting" state
        time.sleep(1)
        self.state_engine.change_state("Tracking")
        pass
    
    def process_tracking(self):
        # Logic for "Tracking" state
        time.sleep(2)
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
                
                if self.state_engine.workID > 20:
                    self.state_engine.change_state("Working")
                    self.state_engine.reset_work_id()
                else: 
                    print(f"Working skipped: {self.state_engine.workID}")
                    time.sleep(2)
        else:
            print("Failed to snap photo.")
        pass
    
    def process_snapping(self):
        # Logic for "Snapping" state
        time.sleep(3)
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
        self.state_engine.currentWorkPath = os.path.join(parent_dir, f"assets/work/work-event.svg")
            
        # Randomly pick one photo ID from the remaining list without removing it
        random_photo_id = random.choice(self.state_engine.photoID)
        startX, startY = self.state_engine.get_image_params_by_id(random_photo_id - 1)
        
        # Create output SVG using the randomly chosen photo ID
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            self.state_engine.currentWorkPath, "work-output-", offset_x=startX, offset_y=startY, id=random_photo_id, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
        )
        
        print(f"Converted Work pattern to SVG: {self.state_engine.currentSVGPath}, random: {random_photo_id}, from {self.state_engine.photoID}")
        stress = self.state_engine.update_stresslevel_from_interval()
        self.plotter.plot_image(self.state_engine.currentSVGPath, stresslevel=stress)
        self.state_engine.last_draw_end_time = time.time()
       
        # Change state to Tracking after the work is done
        self.state_engine.change_state("Tracking")
        pass

    def process_processing(self):
        # Logic for "Drawing" state
        if not self.plotter.connect_to_plotter:
            time.sleep(0.5)
        
        # Calc current stresslevel and convert image to SVG
        params = self.state_engine.get_stress_scaled_params()
        tempSVG = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath, **params)
        
        # Check if the SVG file was generated
        if not tempSVG or not os.path.isfile(tempSVG):
            print("Error: SVG file was not created successfully.")
            self.state_engine.change_state("Waiting")
            return
        
        # Get the starting coordinates
        startX, startY = self.state_engine.get_image_params_by_id(self.state_engine.photoID[-1] - 1)

        # Create the final output SVG file
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            tempSVG, "photo-output-", offset_x=startX, offset_y=startY, id=self.state_engine.photoID[-1] - 1, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
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
            time.sleep(1)
        print(f"Drawing: Connecting with penplotter {self.state_engine.currentSVGPath}")
        stress = self.state_engine.update_stresslevel_from_interval()
        self.plotter.plot_image(self.state_engine.currentSVGPath, stresslevel=stress)
        self.state_engine.update_photo_id()
        self.state_engine.last_draw_end_time = time.time()

        # Check if all spots for images have been drawn
        if self.state_engine.photoID and not self.state_engine.state == "Redrawing":
            self.state_engine.change_state("Waiting")
        elif not self.state_engine.photoID:
            self.state_engine.change_state("ResetPending")
            print(f"All photos printed, changing state to 'ResetPending'.")

    def process_redrawing(self):
        # Go back to drawing or reset if no space left
        time.sleep(3)
        
        if self.state_engine.photoID:
            self.state_engine.change_state("Drawing")
        else:
            self.state_engine.change_state("ResetPending")
        
    def process_reset_pending(self):
        # Logic for "ResetPending" state
        timeout_s = self.state_engine.reset_timeout_s
        
        # Check for state entry to set start time and plot indicator
        if not hasattr(self.state_engine, 'reset_pending_start_time'):
            print(f"🚩 Reset pending: Auto-restart in {timeout_s}s.")
            self.state_engine.reset_pending_start_time = time.time()

            # Draw work-pointer.svg indicator with max speed
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            input_svg_path = os.path.join(parent_dir, f"assets/work/work-pointer.svg")
            
            # Get the parameters for Position 1 (id=1, which is index 0)
            target_id = 1
            startX, startY = self.state_engine.get_image_params_by_id(target_id - 1)
            self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                input_svg_path, 
                "work-pointer-output-", # Use a distinct prefix for the output file
                offset_x=startX, 
                offset_y=startY, 
                id=target_id, 
                paper_width=self.state_engine.paperSizeX, 
                paper_height=self.state_engine.paperSizeY
            )
            
            print(f"Generated work-pointer SVG: {self.state_engine.currentSVGPath}")
            self.plotter.plot_image(self.state_engine.currentSVGPath, is_pointing_motion=True)
            
        # Check if timeout has passed
        if time.time() - self.state_engine.reset_pending_start_time >= timeout_s:
            print(f"Timeout reached. Restarting to Waiting.")
            
            # Change state and cleanup
            self.state_engine.change_state("Waiting")
            del self.state_engine.reset_pending_start_time
        else:
            time.sleep(1) # Prevent busy loop
        pass
    
    def process_template(self, dynamic_grid=False):
        print("🚩 Generate template")
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if (dynamic_grid):
            # Dynamic grid generation (skipped)
            self.currentDebugPath = os.path.join(parent_dir, "assets/work/work-template.svg")
            # Logic to retrieve work pattern and create output SVG
            
            for i in range(1, 16):
                startX, startY = self.state_engine.get_image_params_by_id(i - 1)
                self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                    self.currentDebugPath, "work-output-", offset_x=startX, offset_y=startY, id=i, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
                )
            
            output_directory = os.path.join(parent_dir, "photos/output")
            combined_file_path = os.path.join(parent_dir, "photos/collection/photo-collection.svg")
            self.image_parser.collect_all_paths(output_directory, combined_file_path, "work")
            self.plotter.plot_image(combined_file_path)
        
        else:   
            instructions_file_path = os.path.join(parent_dir, "assets/work/work-instructions.svg")
            self.plotter.plot_image(instructions_file_path, stresslevel=0.65)
            
        self.state_engine.change_state("ResetPending")
        pass
    
    
    def process_test(self):
        
        print("🚩 Starting test")
        # Base directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        photos_dir = os.path.join(parent_dir, "photos/test")
        
        # Find all .jpg files in the directory
        jpg_files = [f for f in os.listdir(photos_dir) if f.endswith('.jpg') and not f.endswith('_optimized.jpg')]
        
        # Initialize the array of IDs
        id_array = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]
        id_index = 0  # Index to track the current position in the array

        # Process each .jpg file
        for jpg_file in jpg_files:
            self.state_engine.currentPhotoPath = os.path.join(photos_dir, jpg_file)

            # Test across 3 fixed stress levels: calm, medium, stressed
            for stress in [0.0, 0.5, 1.0]:
                print(f"\n🧠 Testing stress level {stress:.1f} for {jpg_file}")

                params = self.state_engine.get_stress_scaled_params()
                self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath, **params)

                # Create the final output SVG file using the rolling ID
                current_id = id_array[id_index]
                startX, startY = self.state_engine.get_image_params_by_id(current_id)
                self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                    self.state_engine.currentSVGPath,
                    f"photo-output-stress-{stress:.1f}-",
                    offset_x=startX,
                    offset_y=startY,
                    id=current_id,
                    paper_width=self.state_engine.paperSizeX,
                    paper_height=self.state_engine.paperSizeY
                )

                # Update rolling ID, ensuring it wraps between 0 and 15
                id_index = (id_index + 1) % len(id_array)



        output_directory = os.path.join(parent_dir, "photos/output")
        combined_file_path = os.path.join(parent_dir, "photos/collection/photo-collection.svg")
        self.image_parser.collect_all_paths(output_directory, combined_file_path, "photo")
                            
        print("All SVGs files processed.")
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
            "Redrawing": self.process_redrawing,
            "ResetPending": self.process_reset_pending,
            "Template": self.process_template,
            "Test": self.process_test,
        }
        
        try:
            self.state_engine.client.subscribe("#")
            self.state_engine.client.on_message = self.state_engine.on_message
        except AttributeError:
            pass 
        
        try:
            while True:
                # Update state engine
                current_state = self.state_engine.get_state()
                action = state_actions.get(current_state, lambda: print(f"Unhandled state: {current_state}"))
                action()  # Execute the function associated with the current state
            
        except KeyboardInterrupt:
            print("\nExiting PhotoBooth due to keyboard interrupt...")
