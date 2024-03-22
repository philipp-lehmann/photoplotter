import os
from pyaxidraw import axidraw  # Make sure this is installed

class Plotter:
    def __init__(self):
        # Initialize plotter connection or any other setup
        print("Starting Plotter ...")
        self.ad = axidraw.AxiDraw()  # Create an AxiDraw instance

    def plot_image(self, svg_path, imageID, imagesPerRow, totalImages):
        # Plot an image based on the SVG file at svg_path
        if os.path.isfile(svg_path):
            # Attempt to connect to the AxiDraw plotter
            try:
                self.ad.connect()
            except Exception as e:
                print("Failed to connect to AxiDraw: {e}")
                return  # Stop execution if connection fails

            # Define border and gutter sizes (mm)
            borderSize = 10  # Border size around the entire drawing area
            gutterSize = 5   # Space between images

            # Adjusted maximum dimensions to account for border
            maxX = 420 - (borderSize * 2)  # Maximum X dimension in mm for A3 paper
            maxY = 297 - (borderSize * 2)  # Maximum Y dimension in mm for A3 paper

            # Calculate the number of gutters and subtract their total size from the drawable area
            totalRows = (totalImages + imagesPerRow - 1) // imagesPerRow  # Calculate total rows needed
            drawableWidth = maxX - (gutterSize * (imagesPerRow - 1))
            drawableHeight = maxY - (gutterSize * (totalRows - 1))

            # Calculate offset for each image, including gutters
            offsetX = drawableWidth / imagesPerRow
            offsetY = drawableHeight / totalRows

            # Calculate starting position for the drawing, including border
            col = imageID % imagesPerRow
            row = imageID // imagesPerRow
            startPositionX = (col * offsetX) + (col * gutterSize) + borderSize
            startPositionY = (row * offsetY) + (row * gutterSize) + borderSize

            print("Plotting image from {svg_path} at position ({startPositionX}, {startPositionY})")

            # Perform the plotting operation
            self.ad.plot_setup(svg_path)
            self.ad.plot_run()

            # Provide feedback to the user
            print("Plotting complete.")

            # Disable the motors and disconnect from the AxiDraw plotter
            self.ad.motor_disable()  # Disable the motors
            self.ad.disconnect()  # Disconnect from the AxiDraw plotter
        else:
            print("SVG file not found.")

# Usage example
# plotter = Plotter()
# imageID = some_value_from_state_engine
# imagesPerRow = some_value_from_state_engine
# totalImages = some_calculated_or_defined_value
# svg_path = "path/to/your/svg/file.svg"
# plotter.plot_image(svg_path, imageID, imagesPerRow, totalImages)
