import os
import argparse
from photobooth.plotter import Plotter

# Initialize the plotter
plotter = Plotter()

# Set up argument parsing
parser = argparse.ArgumentParser(description="Plot an image using the Plotter.")
parser.add_argument(
    "--file", 
    required=True, 
    help="Path to the SVG file to be plotted."
)

# Parse the arguments
args = parser.parse_args()
file_name = args.file

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "photos", "current", file_name)

# Plot the image
print(file_path)
plotter.plot_image(file_path)