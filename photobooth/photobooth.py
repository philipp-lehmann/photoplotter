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

        # try:
        #     while True:
        #         # Your application's main loop
        #         print(f"Current state: {self.state_engine.state}")
        #         while True:
                    
        #             pass 

                
        # except KeyboardInterrupt:
        #     # Handle graceful exit upon Ctrl+C
        #     print("\nExiting PhotoBooth due to keyboard interrupt...")