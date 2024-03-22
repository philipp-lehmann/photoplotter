from .stateengine import StateEngine
from .camera import Camera
from .plotter import Plotter
from .imageparser import ImageParser


class PhotoBooth:
    def __init__(self):
        print("Starting PhotoBooth ...")
        self.state_engine = StateEngine()
        self.camera = Camera()
        self.plotter = Plotter()
        self.image_parser = ImageParser()

    def start(self):
        # Initialize the state engine
        
        self.state_engine.change_state("Ready")
        
        # Snap a photo
        print("Snapping a photo...")
        image_path = self.camera.capture_image()
        if image_path:
            print(f"Photo captured and saved at {image_path}")
            # If you want to process the image right after capturing it
            # svg_filepath, num_paths = self.image_parser.convert_to_svg(image_path)
            # if svg_filepath:
            #     print(f"Image processed successfully. SVG saved at: {svg_filepath}")
            # Proceed with any additional steps, like sending the image to the plotter
            # self.plotter.plot(svg_filepath)
        else:
            print("Failed to capture photo.")
        
        # Draw a test on a position
        # self.state_engine.change_state("Drawing")
        # self.plotter.plot_image("./assets/test.svg", 0, 5, 3)
        # self.plotter.return_home()

        # try:
        #     while True:
        #         # Your application's main loop
        #         print(f"Current state: {self.state_engine.state}")
        #         while True:
                    
        #             pass 

                
        # except KeyboardInterrupt:
        #     # Handle graceful exit upon Ctrl+C
        #     print("\nExiting PhotoBooth due to keyboard interrupt...")