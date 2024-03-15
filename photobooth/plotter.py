import os
#from pyaxidraw import axidraw

class Plotter:
    def __init__(self):
        # Initialize plotter connection or any other setup
        pass

    def plot_image(self, svg_path):
        # Plot an image based on the SVG file at svg_path
        if os.path.isfile(svg_path):
            # Connect to the AxiDraw plotter
            # axidraw_instance = axidraw.AxiDraw()
            # axidraw_instance.connect() 

            # Perform the plotting operation
            # axidraw_instance.plot_setup(svg_path)
            # axidraw_instance.plot_run()

            # Provide feedback to the user
            print("Plotting image from", svg_path)
            # flash("AxiDraw plot started successfully.", 'success')

            # Disconnect from the AxiDraw plotter
            # axidraw_instance.motor_disable()  # Disable the motors
            # axidraw_instance.disconnect()  # Disconnect from the AxiDraw plotter
        else:
            print("SVG file not found.")

# Usage example:
# plotter = Plotter()
# plotter.plot_image(svg_path)
