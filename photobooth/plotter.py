import os
import time
from pyaxidraw import axidraw

class Plotter:
    def __init__(self):
        print("Starting Plotter ...")
        self.startPositionX = 0
        self.startPositionY = 0
        self.ad = axidraw.AxiDraw()
        self.ad.interactive()
        self.plotter_found = self.connect_to_plotter()

    def connect_to_plotter(self):
        """Attempt to connect to the AxiDraw plotter."""
        if self.ad.connect():
            print("AxiDraw connected.")
            self.ad.options.model = 2
            self.ad.update()
            return True
        else:
            print("AxiDraw not found. Entering simulation mode.")
            return False

    def return_home(self):
        if self.plotter_found:
            # New code to return home
            print("Plotter: Returning home.")
        else:
            print("Simulation Mode: Returning home.")

    def plot_image(self, svg_path):
        svg_path = os.path.abspath(svg_path)
        
        if os.path.exists(svg_path):
            pass
        else:
            print("SVG file not found.")
        
        if self.plotter_found:
            print("Plotter: Plotting image.")

            self.ad.plot_setup(svg_path)
            self.ad.plot_run()
            
            print("Plotting complete.")
        else:
            print("AxiDraw not found.")