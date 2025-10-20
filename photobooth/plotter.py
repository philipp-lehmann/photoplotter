import time
import os
from nextdraw import NextDraw

class Plotter:
    def __init__(self):
        print("Starting Plotter ...")
        self.nd1 = NextDraw()
        self.nd1.interactive() 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "nextdraw_conf.py")
        self.nd1.load_config(config_path)
 
        self.plotter_found = self.connect_to_plotter()

        if self.nd1.connect():
            print("Plotter connected successfully.")
            print(f"NextDraw connected. Model: {self.nd1.options.model}" )

        else:
            print("Failed to connect to the plotter.")
            
        self.nd1.plot_setup()
        self.nd1.plot_run()

    # def plot(self):
    #     """Test simple plot movements."""
    #     self.nd1.moveto(1, 1)
    #     self.nd1.lineto(2, 1)
    #     self.nd1.moveto(0, 0)
    #     self.nd1.disconnect()
        
    def connect_to_plotter(self, speed=60):
        """Attempt to connect to the NextDraw plotter."""
        if self.nd1.connect():
            self.nd1.options.model = 2
            self.nd1.options.auto_rotate = True
            self.nd1.options.pen_rate_lower = speed
            self.nd1.options.pen_rate_raise = speed
            self.nd1.options.speed_pendown = speed
            self.nd1.options.speed_pendown = speed
            self.nd1.options.speed_penup = speed
            self.nd1.options.penlift = 3
            
            self.nd1.update()
            print(f"NextDraw connected. Model: {self.nd1.options.model}")
            return True
        else:
            print("NextDraw not found. Entering simulation mode.")
            return False

    def return_home(self):
        if self.plotter_found:
            # New code to return home
            print("Plotter: Returning home.")
        else:
            print("Simulation Mode: Returning home.")

    def plot_image(self, svg_path, stresslevel=1.0, is_pointing_motion=False):
        """
        Plot an SVG image with optional stress level adjustment.

        stresslevel: float (0.0 - 1.0)
            Controls plotting speed or pen pressure — higher = more stress, faster.
        is_pointing_motion: bool
            If True, overrides stress-based speed and sets max speed for quick pointing/reset indicator.
        """
        svg_path = os.path.abspath(svg_path)

        if not os.path.exists(svg_path):
            print("SVG file not found.")
            return

        if not self.plotter_found:
            print("NextDraw not found. Plotting 'work-pointer' skipped in simulation mode.")
            return

        # Ensure stresslevel is clamped between 0 and 1
        stresslevel = max(0.0, min(1.0, stresslevel))

        if is_pointing_motion:
            # Set maximum speed for quick motion
            adjusted_speed = 50 
            print(f"Plotting reset indicator with max speed ({adjusted_speed}).")
        else:
            # Derive speed or pressure from stresslevel (base_speed 40, scaled range 60-100)
            base_speed = 40
            adjusted_speed = int(base_speed + stresslevel * 60) 
            print(f"Plotting with stress level {stresslevel} (speed={adjusted_speed})")

        self.nd1.interactive()
        self.nd1.plot_setup(svg_path)

        # Update options 
        self.nd1.options.reordering = 2
        self.nd1.options.speed_pendown = adjusted_speed
        self.nd1.options.speed_penup = adjusted_speed
        self.nd1.options.pen_rate_lower = adjusted_speed
        self.nd1.options.pen_rate_raise = adjusted_speed

        self.nd1.update()
        
        # Run plotting
        start_time = time.time() # Record the time before plotting starts
        self.nd1.plot_run(True)

        # --- Duration Calculation and Print ---
        end_time = time.time() # Record the time after plotting finishes
        plot_duration = end_time - start_time
        # --- Duration Calculation and Print ---

        print("Plotting complete.")
        print(f"⏱️ Plot duration: {plot_duration:.2f} seconds.") # Print the duration