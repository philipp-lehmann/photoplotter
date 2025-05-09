import os
from nextdraw import NextDraw

# File
file_name = "photo-output-1.svg"

# Construct the path
base_path = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(base_path, "photos", "current")
file_path = os.path.join(base_dir, file_name)

# Pass the constructed path
print(file_path)
nd1 = NextDraw()

            
nd1.plot_setup(file_path)
nd1.plot_run()

