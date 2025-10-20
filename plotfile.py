import os
import argparse
from photobooth.plotter import Plotter

# Initialize the plotter
plotter = Plotter()

# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot an image using the Plotter.")
parser.add_argument(
    "file", 
    help="Absolute path to the SVG file to be plotted."
)

# Parse the arguments
args = parser.parse_args()
file_path = args.file

# Plot the image
print(file_path)
plotter.plot_image(file_path)
