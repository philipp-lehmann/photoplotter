import os
import time
from pyaxidraw import axidraw

class Plotter:
    def __init__(self):
        print("Starting Plotter ...")
        self.ad = axidraw.AxiDraw()
        self.ad.interactive()
        self.plotter_found = self.connect_to_plotter()

    def connect_to_plotter(self):
        """Attempt to connect to the AxiDraw plotter."""
        if self.ad.connect():
            print("AxiDraw connected.")
            self.ad.options.units = 2
            self.ad.update()
            self.ad.moveto(0, 0)
            return True
        else:
            print("AxiDraw not found. Entering simulation mode.")
            return False

    def return_home(self):
        if self.plotter_found:
            self.ad.moveto(1, 0)
        else:
            print("Simulation Mode: Returning home.")

    def plot_image(self, svg_path, photoID, imagesPerRow, totalImages):
        
        svg_path = os.path.abspath(svg_path)
        
        print(f"Attempting to access SVG file at: {svg_path}")
        if os.path.exists(svg_path):
            print("SVG file found.")
        else:
            print("SVG file not found.")
        
        if self.plotter_found:
            print("Plotting image.")

            # Define border and gutter sizes (mm)
            borderSize = 10  # Border size around the entire drawing area
            gutterSize = 5   # Space between images

            # Adjusted maximum dimensions to account for border
            maxX = 420 - (borderSize * 2)  # Maximum X dimension in mm for A3 paper
            maxY = 297 - (borderSize * 2)  # Maximum Y dimension in mm for A3 paper

            # Calculate the number of gutters and subtract their total size from the drawable area
            totalRows = (totalImages + imagesPerRow - 1) // imagesPerRow
            drawableWidth = maxX - (gutterSize * (imagesPerRow - 1))
            drawableHeight = maxY - (gutterSize * (totalRows - 1))

            # Calculate offset for each image, including gutters
            offsetX = drawableWidth / imagesPerRow
            offsetY = drawableHeight / totalRows

            # Calculate starting position for the drawing, including border
            col = photoID % imagesPerRow
            row = photoID // imagesPerRow
            startPositionX = (col * offsetX) + (col * gutterSize) + borderSize
            startPositionY = (row * offsetY) + (row * gutterSize) + borderSize

            print(f"Plotting image from {svg_path} at position ({startPositionX}, {startPositionY})")

            self.ad.moveto(startPositionX, startPositionY)    
            self.ad.plot_setup(svg_path)
            self.ad.plot_run()

            # Move back to the "home" position after plotting
            print("Returning to home position.")
            self.ad.moveto(0, 0)

            print("Plotting complete.")

            # Disconnect is commented out; uncomment if ending the session
            # self.ad.disconnect()  # Disconnect from the AxiDraw plotter
        else:
            print("AxiDraw not found.")

# Usage example
# plotter = Plotter()
# photoID = some_value_from_state_engine
# imagesPerRow = some_value_from_state_engine
# totalImages = some_calculated_or_defined_value
# svg_path = "path/to/your/svg/file.svg"
# plotter.plot_image(svg_path, photoID, imagesPerRow, totalImages)
