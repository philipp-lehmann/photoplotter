import os
from photobooth.plotter import Plotter

plotter = Plotter()

# File
file_name = "photo-output-3.svg"

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "photos", "current", file_name)
config_path = os.path.join(current_dir, "photobooth", "nextdraw_conf.py")

# Pass the constructed path
print(file_path)
plotter.plot_image(file_path)