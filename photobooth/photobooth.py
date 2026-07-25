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
        self.test_mode = "slots"  # "slots" (verify slot geometry, no plotter needed) | "photos" (original stress-level test loop)

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
        time.sleep(1)
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
        geom = self.state_engine.get_slot_geometry(random_photo_id)
        scale_factor = self.state_engine.compute_scale_factor(
            geom.width, geom.height, source_size=self.state_engine.DEFAULT_TARGET_SIZE
        )

        # Create output SVG using the randomly chosen photo ID
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            self.state_engine.currentWorkPath, "work-output-", offset_x=geom.x, offset_y=geom.y, scale_factor=scale_factor, id=random_photo_id, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
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
        
        # Calc current stresslevel and convert image to SVG, scaled for the target slot's size
        next_slot_id = self.state_engine.photoID[-1]
        params = self.state_engine.get_render_params(next_slot_id)
        tempSVG = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath, **params)

        # Check if the SVG file was generated
        if not tempSVG or not os.path.isfile(tempSVG):
            print("Error: SVG file was not created successfully.")
            self.state_engine.change_state("Waiting")
            return

        # Keep the traced (pre-positioned) SVG + its native size so a redraw can
        # later re-place the same tracing into Slot 1 without recalculating it.
        self.state_engine.currentTracedSVGPath = tempSVG
        self.state_engine.currentTraceWidth = params["target_width"]

        # Get slot geometry and the scale factor needed to fill it
        geom = self.state_engine.get_slot_geometry(next_slot_id)
        scale_factor = self.state_engine.compute_scale_factor(
            geom.width, geom.height, source_size=params["target_width"]
        )

        # Create the final output SVG file
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            tempSVG, "photo-output-", offset_x=geom.x, offset_y=geom.y, scale_factor=scale_factor, id=next_slot_id, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
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
        # KEY2 always reprints the current tracing into Slot 1 (the featured slot),
        # no recalculation and no photo slot consumed (Slot 1 isn't part of photoID).
        target_id = 1
        geom = self.state_engine.get_slot_geometry(target_id)
        scale_factor = self.state_engine.compute_scale_factor(
            geom.width, geom.height, source_size=self.state_engine.currentTraceWidth
        )
        self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
            self.state_engine.currentTracedSVGPath, "photo-output-", offset_x=geom.x, offset_y=geom.y, scale_factor=scale_factor, id=target_id, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
        )

        print(f"Redrawing: Reprinting {self.state_engine.currentSVGPath} into Slot {target_id}")
        stress = self.state_engine.update_stresslevel_from_interval()
        self.plotter.plot_image(self.state_engine.currentSVGPath, stresslevel=stress)
        self.state_engine.last_draw_end_time = time.time()

        if self.state_engine.photoID:
            self.state_engine.change_state("Waiting")
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
            
            # Get the parameters for Position 1 (the featured slot)
            target_id = 1
            geom = self.state_engine.get_slot_geometry(target_id)
            scale_factor = self.state_engine.compute_scale_factor(
                geom.width, geom.height, source_size=self.state_engine.DEFAULT_TARGET_SIZE
            )
            self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                input_svg_path,
                "work-pointer-output-", # Use a distinct prefix for the output file
                offset_x=geom.x,
                offset_y=geom.y,
                scale_factor=scale_factor,
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
    
    def process_template(self, dynamic_grid=False, output_filename="photo-collection.svg", change_state=True):
        print("🚩 Generate template")
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if (dynamic_grid):
            # Dynamic grid generation (skipped)
            self.currentDebugPath = os.path.join(parent_dir, "assets/work/work-template-featured.svg")
            # Logic to retrieve work pattern and create output SVG

            for slot_id in self.state_engine.slot_ids:
                geom = self.state_engine.get_slot_geometry(slot_id)
                # fill_ratio=1.0: these are alignment/crop-mark guides, not photo content — they
                # should trace the true cell boundary exactly rather than leave the usual ink margin.
                scale_factor = self.state_engine.compute_scale_factor(
                    geom.width, geom.height, source_size=self.state_engine.DEFAULT_TARGET_SIZE, fill_ratio=1.0
                )
                self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                    self.currentDebugPath, "work-output-", offset_x=geom.x, offset_y=geom.y, scale_factor=scale_factor, id=slot_id, paper_width=self.state_engine.paperSizeX, paper_height=self.state_engine.paperSizeY
                )

            output_directory = os.path.join(parent_dir, "photos/output")
            combined_file_path = os.path.join(parent_dir, "photos/collection", output_filename)
            self.image_parser.collect_all_paths(output_directory, combined_file_path, "work")
            self.plotter.plot_image(combined_file_path)

        else:
            instructions_file_path = os.path.join(parent_dir, "assets/work/work-template-featured.svg")
            self.plotter.plot_image(instructions_file_path, stresslevel=0.65)

        if change_state:
            self.state_engine.change_state("ResetPending")
        pass
    
    
    def process_test(self):
        """Dispatches to a test scenario based on self.test_mode:
        - "layout": verify slot geometry (positions/sizes), no plotter or real photos needed.
        - "photos": exercise the real image-tracing pipeline across stress levels (original behavior).
        """
        if self.test_mode == "slots":
            self.process_test_slots()
        else:
            self.process_test_photos()

    def process_test_slots(self):
        print("🚩 Starting test (slots)")

        # Reuses the dynamic-grid template generation to visually verify slot geometry.
        self.process_template(dynamic_grid=True, change_state=False)

        print("Slot layout preview saved to photos/collection/photo-collection.svg")
        sys.exit()

    def process_test_photos(self):
        print("🚩 Starting test (photos)")
        # Base directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        photos_dir = os.path.join(parent_dir, "photos/test")

        # Find all .jpg files in the directory
        jpg_files = [f for f in os.listdir(photos_dir) if f.endswith('.jpg') and not f.endswith('_optimized.jpg')]

        # Initialize the array of IDs (all 12 physical slots, including the featured one,
        # for full debug coverage of the new layout)
        id_array = list(self.state_engine.slot_ids)
        id_index = 0  # Index to track the current position in the array

        # Process each .jpg file
        for jpg_file in jpg_files:
            self.state_engine.currentPhotoPath = os.path.join(photos_dir, jpg_file)

            # Test across 3 fixed stress levels: calm, medium, stressed
            for stress in [0.0, 0.5, 1.0]:
                print(f"\n🧠 Testing stress level {stress:.1f} for {jpg_file}")

                current_id = id_array[id_index]
                params = self.state_engine.get_render_params(current_id)
                self.state_engine.currentSVGPath = self.image_parser.convert_to_svg(self.state_engine.currentPhotoPath, **params)

                # Create the final output SVG file using the rolling ID
                geom = self.state_engine.get_slot_geometry(current_id)
                scale_factor = self.state_engine.compute_scale_factor(
                    geom.width, geom.height, source_size=params["target_width"]
                )
                self.state_engine.currentSVGPath = self.image_parser.create_output_svg(
                    self.state_engine.currentSVGPath,
                    f"photo-output-stress-{stress:.1f}-",
                    offset_x=geom.x,
                    offset_y=geom.y,
                    scale_factor=scale_factor,
                    id=current_id,
                    paper_width=self.state_engine.paperSizeX,
                    paper_height=self.state_engine.paperSizeY
                )

                # Update rolling ID, ensuring it wraps within id_array
                id_index = (id_index + 1) % len(id_array)



        output_directory = os.path.join(parent_dir, "photos/output")
        combined_file_path = os.path.join(parent_dir, "photos/collection/photo-collection.svg")
        self.image_parser.collect_all_paths(output_directory, combined_file_path, "photo")

        print("All SVGs files processed.")
        sys.exit()


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
