from .stateengine import StateEngine
from .camera import Camera
from .plotter import Plotter
from .imageparser import ImageParser


class PhotoBooth:
    def __init__(self):
        print(f"Starting PhotoBooth ...")
        self.state_engine = StateEngine()
        self.camera = Camera()
        self.plotter = Plotter()
        self.image_parser = ImageParser()

    def start(self):
        # Init state engine
        self.state_engine.change_state("Ready")